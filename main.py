import dataclasses
import time
from datetime import datetime

import numpy as np

from config import Config
from src.instance import Instance
from src.problem import PortfolioProblem
from src.algorithm import (
    RandomFeasibleAlgorithm, GreedyAlgorithm, GRASPAlgorithm, OneOverNAlgorithm,
)
from src.evofolio import EvoFolioAlgorithm
from src.grasp_pareto import GRASPParetoAlgorithm
from src.local_search import swap_local_search
from src.backtest import run_backtest, period_length
from src.io_utils import ensure_dir, save_solutions_to_csv
from src.visualization import (
    plot_risk_return, plot_weights, plot_market_day_grid,
    plot_evolution, plot_target_period, plot_risk_return_evolution,
    plot_evofolio_evolution, plot_sharpe_evolution,
)


def _plot_path(cfg, name: str) -> str:
    return f"{cfg.plots_dir}/{name}.{cfg.plot_format}"


def _setup_run_dir(cfg: Config) -> Config:
    ensure_dir(cfg.plots_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algos = []
    if cfg.run_greedy:                              algos.append("greedy")
    if cfg.run_grasp:                               algos.append("grasp")
    if cfg.run_random:                              algos.append("random")
    if cfg.run_oneoverN:                            algos.append("1N")
    if cfg.run_evofolio:                            algos.append("evofolio")
    if cfg.run_grasp_pareto:                        algos.append("graspPareto")
    if getattr(cfg, "run_full_oneoverN", False):    algos.append("full1N")
    if getattr(cfg, "run_buy_and_hold", False):     algos.append("buyhold")
    algo_tag = "+".join(algos) if algos else "none"

    schedule_tag = cfg.rebalancing if cfg.mode == "multi_period" else f"t{cfg.single_period_t}"
    run_name = (
        f"{timestamp}__{algo_tag}__{schedule_tag}"
        f"__{cfg.weight_method}__{cfg.risk_method}__seed{cfg.seed}"
    )
    run_dir = f"{cfg.plots_dir}/{run_name}"
    ensure_dir(run_dir)

    new_cfg = dataclasses.replace(cfg, plots_dir=run_dir)
    _write_run_info(new_cfg, run_dir)
    print(f"Run dir: {run_dir}/")
    return new_cfg


def _write_run_info(cfg: Config, run_dir: str) -> None:
    algos_enabled = [name for name, flag in [
        ("Greedy",       cfg.run_greedy),
        ("GRASP",        cfg.run_grasp),
        ("Random",       cfg.run_random),
        ("1/N",          cfg.run_oneoverN),
        ("EvoFolio",     cfg.run_evofolio),
        ("GRASP_Pareto", cfg.run_grasp_pareto),
    ] if flag]

    L = []
    sep = "═" * 67
    sub = "─" * 67
    L += [sep, f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sep, ""]

    L += ["── Algorithms run " + sub[18:],
          f"  {', '.join(algos_enabled) if algos_enabled else '(none)'}",
          ""]

    L += ["── Dataset " + sub[11:],
          f"  data_root:        {cfg.data_root}",
          f"  frequency:        {cfg.frequency}",
          f"  price_col:        {cfg.price_col}",
          ""]

    L += ["── Portfolio constraints " + sub[25:],
          f"  budget:           {cfg.budget:,}",
          f"  cardinality:      [{cfg.min_cardinality}, {cfg.max_cardinality}]",
          f"  weight bounds:    [{cfg.min_weight}, {cfg.max_weight}]",
          f"  SLSQP bounds:     [{cfg.slsqp_min_weight}, {cfg.slsqp_max_weight}]",
          f"  risk-free rate:   {cfg.risk_free_rate}",
          ""]

    L += ["── Execution mode " + sub[18:],
          f"  mode:             {cfg.mode}"]
    if cfg.mode == "single_period":
        L += [f"  single_period_t:  {cfg.single_period_t}",
              f"  use_time_limit:   {cfg.use_time_limit}",
              f"  time_limit:       {cfg.time_limit}",
              f"  population_size:  {cfg.population_size}",
              f"  grasp_iterations: {cfg.grasp_iterations}"]
    else:
        L += [f"  rebalancing:      {cfg.rebalancing}",
              f"  max_periods:      {cfg.max_periods}",
              f"  warmup_periods:   {cfg.warmup_periods}",
              f"  lookback_window:  {cfg.lookback_window}",
              f"  use_time_budget:  {cfg.use_time_budget}",
              f"  time_budget_s:    {cfg.time_budget_seconds}",
              f"  GRASP iters/per.: {cfg.backtest_grasp_iterations}",
              f"  Random sols/per.: {cfg.backtest_random_solutions}",
              f"  return_signal:    {cfg.return_signal}"]
    L += [""]

    L += ["── Methods " + sub[11:],
          f"  risk_method:      {cfg.risk_method}",
          f"  weight_method:    {cfg.weight_method}",
          f"  use_local_search: {cfg.use_local_search}",
          ""]

    L += ["── Hyperparameters " + sub[19:],
          f"  seed:             {cfg.seed}",
          f"  grasp_alpha:      {cfg.grasp_alpha}"]
    if cfg.run_evofolio:
        L += ["",
              "  EvoFolio NSGA-II (defaults = paper's best):",
              f"    MU (population):  {cfg.evofolio_population_size}",
              f"    NGEN:             {cfg.evofolio_generations}",
              f"    CXPB:             {cfg.evofolio_crossover_rate}",
              f"    indpb (per-gene): {cfg.evofolio_indpb}",
              f"    TOURNPARAM:       {cfg.evofolio_tournparam}",
              f"    SELPARAM:         {cfg.evofolio_selparam}",
              f"    ELITEPARAM:       {cfg.evofolio_elite_cxpb}"]
    if cfg.run_grasp_pareto:
        L += ["",
              "  Pareto-GRASP:",
              f"    alpha:            {cfg.pareto_grasp_alpha}",
              f"    iters/period:     {cfg.backtest_pareto_grasp_iterations}"]
    L += [""]

    L += ["── Visualisation " + sub[17:],
          f"  target_period_index: {cfg.target_period_index}",
          f"  plot_format:         {cfg.plot_format}",
          "",
          sep]

    with open(f"{run_dir}/run_info.txt", "w") as f:
        f.write("\n".join(L))


def main(cfg: Config | None = None):
    """Entry point: load data, run the backtest for all enabled algorithms and save metrics/plots.

    In: cfg (Config | None, defaults to Config()).
    Out: None (writes results to disk).
    """
    if cfg is None:
        cfg = Config()
    ensure_dir(cfg.results_dir)
    cfg = _setup_run_dir(cfg)

    instance = Instance.from_folder(
        data_root=cfg.data_root,
        frequency=cfg.frequency,
        price_col=cfg.price_col,
    )
    print(f"Loaded {instance.frequency} data: {len(instance.assets)} assets, "
          f"{len(instance.returns)} return observations.")

    if cfg.mode == "single_period":
        _run_single_period(cfg, instance)
    elif cfg.mode == "multi_period":
        _run_multi_period(cfg, instance)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. "
                         f"Use 'single_period' or 'multi_period'.")


def _run_single_period(cfg, instance):
    t = cfg.single_period_t

    problem = PortfolioProblem(
        instance=instance,
        t=t,
        period=1,
        budget=cfg.budget,
        min_cardinality=cfg.min_cardinality,
        max_cardinality=cfg.max_cardinality,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
        slsqp_min_weight=cfg.slsqp_min_weight,
        slsqp_max_weight=cfg.slsqp_max_weight,
        risk_free_rate=cfg.risk_free_rate,
        weight_method=cfg.weight_method,
        risk_method=cfg.risk_method,
    )

    if cfg.use_time_limit:
        tl          = cfg.time_limit
        pop_size    = None
        grasp_iters = None
        mode_label  = f"time limit = {cfg.time_limit}s"
    else:
        tl          = None
        pop_size    = cfg.population_size
        grasp_iters = cfg.grasp_iterations
        mode_label  = f"iterations (random={pop_size}, grasp={grasp_iters})"

    print(f"\nStopping criterion: {mode_label}")
    print(f"Optimising for returns.iloc[{t}]  "
          f"({instance.returns.index[t].date()})\n")

    results = {}
    print(f"{'Algorithm':<12} {'Sharpe':>8} {'Return':>8} {'Risk':>8} "
          f"{'Card':>5} {'Time':>7} {'#Sol':>10}")
    print("-" * 65)

    if cfg.run_greedy:
        t0 = time.time()
        sol, _ = GreedyAlgorithm(problem).run()
        if cfg.use_local_search:
            sol = swap_local_search(sol, problem)
        results["Greedy"] = (sol, time.time() - t0, 1)
        print(f"{'Greedy':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['Greedy'][1]:>6.1f}s {1:>10}")

    if cfg.run_grasp:
        t0 = time.time()
        sol, _ = GRASPAlgorithm(problem, n_iterations=grasp_iters, time_limit=tl,
                                alpha=cfg.grasp_alpha, seed=cfg.seed).run()
        if cfg.use_local_search:
            sol = swap_local_search(sol, problem)
        results["GRASP"] = (sol, time.time() - t0, 1)
        print(f"{'GRASP':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['GRASP'][1]:>6.1f}s {1:>10}")

    if cfg.run_oneoverN:
        t0 = time.time()
        sol, oneoverN_pop = OneOverNAlgorithm(
            problem, population_size=pop_size, time_limit=tl, seed=cfg.seed
        ).run()
        results["1/N"] = (sol, time.time() - t0, len(oneoverN_pop))
        print(f"{'1/N':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['1/N'][1]:>6.1f}s {len(oneoverN_pop):>10}")

    if cfg.run_grasp_pareto:
        t0 = time.time()
        sol, pgrasp_pop = GRASPParetoAlgorithm(
            problem,
            n_iterations=grasp_iters,
            time_limit=tl,
            alpha=cfg.pareto_grasp_alpha,
            seed=cfg.seed,
        ).run()
        results["GRASP_Pareto"] = (sol, time.time() - t0, len(pgrasp_pop))
        print(f"{'GRASP_Pareto':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['GRASP_Pareto'][1]:>6.1f}s {len(pgrasp_pop):>10}")

    if cfg.run_evofolio:
        t0 = time.time()
        n_gens = None if cfg.use_time_limit else cfg.evofolio_generations
        sol, evo_pop = EvoFolioAlgorithm(
            problem,
            population_size=cfg.evofolio_population_size,
            n_generations=n_gens,
            crossover_rate=cfg.evofolio_crossover_rate,
            indpb=cfg.evofolio_indpb,
            tournparam=cfg.evofolio_tournparam,
            selparam=cfg.evofolio_selparam,
            elite_cxpb=cfg.evofolio_elite_cxpb,
            time_limit=tl,
            seed=cfg.seed,
        ).run()
        results["EvoFolio"] = (sol, time.time() - t0, len(evo_pop))
        print(f"{'EvoFolio':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['EvoFolio'][1]:>6.1f}s {len(evo_pop):>10}")

    random_solutions = []
    if cfg.run_random:
        t0 = time.time()
        _, random_solutions = RandomFeasibleAlgorithm(
            problem, population_size=pop_size, time_limit=tl, seed=cfg.seed
        ).run()
        sol = random_solutions[0]
        results["Random"] = (sol, time.time() - t0, len(random_solutions))
        print(f"{'Random':<12} {sol.sharpe:>8.4f} {sol.expected_return:>8.4f} "
              f"{sol.risk:>8.4f} {sol.cardinality():>5} "
              f"{results['Random'][1]:>6.1f}s {len(random_solutions):>10}")

    if random_solutions:
        save_solutions_to_csv(random_solutions, instance.assets,
                              cfg.solutions_output_path)

    base_solutions = random_solutions or []
    overlays = []
    if "Greedy" in results:
        overlays.append((results["Greedy"][0], "Greedy", "steelblue", "D"))
    if "GRASP"  in results:
        overlays.append((results["GRASP"][0],  "GRASP",  "darkorange", "*"))
    if "Random" in results and not random_solutions:
        overlays.append((results["Random"][0], "Random", "seagreen", "^"))

    if base_solutions:
        plot_risk_return(base_solutions, overlay_solutions=overlays or None,
                         save_path=_plot_path(cfg, f"risk_return_t{t}"),
                         show=cfg.show_plots)

    plot_market_day_grid(instance, t=t,
                         save_path=_plot_path(cfg, f"market_grid_t{t}"),
                         show=cfg.show_plots)

    best_sol = (results.get("GRASP") or results.get("GRASP_Pareto")
                or results.get("Greedy") or results.get("EvoFolio")
                or results.get("Random") or results.get("1/N"))[0]
    plot_weights(best_sol, instance.assets, instance, t=t,
                 save_path=_plot_path(cfg, f"weights_t{t}"),
                 show=cfg.show_plots)


def _run_multi_period(cfg, instance):
    period = period_length(cfg, instance)
    print(f"\nBacktest: {cfg.rebalancing} rebalancing  "
          f"({period} {instance.frequency.lower()} rows per period)")
    print(f"Cardinality: [{cfg.min_cardinality}, {cfg.max_cardinality}]  "
          f"|  Budget: {cfg.budget:,.0f}  |  Max periods: {cfg.max_periods}")
    print(f"Weight method: {cfg.weight_method}  "
          f"|  Local search: {cfg.use_local_search}  "
          f"|  Risk metric: {cfg.risk_method}")
    if cfg.run_evofolio:
        print(f"EvoFolio NSGA-II: MU={cfg.evofolio_population_size}, "
              f"NGEN={cfg.evofolio_generations}, "
              f"CXPB={cfg.evofolio_crossover_rate}, "
              f"indpb={cfg.evofolio_indpb}, "
              f"TOURNPARAM={cfg.evofolio_tournparam}, "
              f"SELPARAM={cfg.evofolio_selparam}, "
              f"ELITEPARAM={cfg.evofolio_elite_cxpb}")
    if cfg.run_grasp_pareto:
        print(f"Pareto-GRASP: iters/period={cfg.backtest_pareto_grasp_iterations}, "
              f"alpha={cfg.pareto_grasp_alpha}")
    if cfg.use_time_budget:
        print(f"Return signal: {cfg.return_signal}  |  "
              f"Time budget: {cfg.time_budget_seconds:.0f}s total per algorithm")
    else:
        print(f"Return signal: {cfg.return_signal}  |  "
              f"GRASP iters/period: {cfg.backtest_grasp_iterations}  |  "
              f"Random sols/period: {cfg.backtest_random_solutions}")

    t0 = time.time()
    results = run_backtest(instance, cfg, period)
    elapsed = time.time() - t0

    first_result = next(iter(results.values()))
    print(f"\nBacktest complete in {elapsed:.1f}s  "
          f"({len(first_result.periods)} periods)\n")

    periods_per_year = instance.periods_per_year / period

    print(f"{'Algorithm':<12} {'Final value':>15} {'Gain':>10} "
          f"{'Ann. return':>12} {'Ann. vol':>10} {'Ann. Sharpe':>12}")
    print("-" * 75)
    for name, res in results.items():
        rets = res.realized_log_returns
        ann_ret = rets.mean() * periods_per_year
        ann_vol = rets.std()  * np.sqrt(periods_per_year)
        ann_sh  = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        final_v = res.portfolio_values[-1]
        gain    = final_v / cfg.budget
        print(f"{name:<12} {final_v:>15,.2f} {gain:>9.3f}× "
              f"{ann_ret:>12.2%} {ann_vol:>10.2%} {ann_sh:>12.4f}")

    suffix = cfg.rebalancing
    rolling_window = max(2, round(periods_per_year))
    plot_evolution(
        results, rebalancing_label=cfg.rebalancing,
        rolling_window=rolling_window, initial_value=cfg.budget,
        save_path=_plot_path(cfg, f"evolution_{suffix}"),
        show=cfg.show_plots,
    )
    plot_risk_return_evolution(
        results, instance, cfg, period,
        save_path=_plot_path(cfg, f"risk_return_evolution_{suffix}"),
        show=cfg.show_plots,
    )
    plot_evofolio_evolution(
        results, instance, cfg, period,
        save_path=_plot_path(cfg, f"evofolio_evolution_{suffix}"),
        show=cfg.show_plots,
    )
    plot_sharpe_evolution(
        results, cfg, periods_per_year=periods_per_year,
        rolling_window=rolling_window,
        save_path=_plot_path(cfg, f"sharpe_evolution_{suffix}"),
        show=cfg.show_plots,
    )

    if cfg.target_period_index is not None:
        plot_target_period(
            results, cfg,
            save_path=_plot_path(cfg, f"target_period_{suffix}_p{cfg.target_period_index}"),
            show=cfg.show_plots,
        )

    print(f"\nPlots saved to {cfg.plots_dir}/")


if __name__ == "__main__":
    main()
