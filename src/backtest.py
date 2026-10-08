import numpy as np
from dataclasses import dataclass, field

from src.problem import PortfolioProblem
from src.algorithm import (
    RandomFeasibleAlgorithm, GreedyAlgorithm, GRASPAlgorithm, OneOverNAlgorithm,
)
from src.evofolio import EvoFolioAlgorithm
from src.grasp_pareto import GRASPParetoAlgorithm
from src.local_search import swap_local_search


PERIOD_BY_FREQ = {
    "WEEK": {"monthly": 4, "quarterly": 12},
    "DAY":  {"monthly": 21, "quarterly": 63},
}


@dataclass
class PeriodResult:
    date: object
    t: int
    weights: np.ndarray
    shares: np.ndarray
    cash_left: float
    expected_sharpe: float
    realized_log_return: float
    portfolio_value: float
    avg_risk: float = 0.0
    avg_return: float = 0.0
    n_candidates: int = 0
    trace: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


@dataclass
class BacktestResult:
    name: str
    periods: list[PeriodResult] = field(default_factory=list)
    target_candidates: list = field(default_factory=list)
    target_period_idx: int | None = None

    @property
    def dates(self) -> list:
        return [p.date for p in self.periods]

    @property
    def portfolio_values(self) -> np.ndarray:
        return np.array([p.portfolio_value for p in self.periods])

    @property
    def realized_log_returns(self) -> np.ndarray:
        return np.array([p.realized_log_return for p in self.periods])

    @property
    def expected_sharpes(self) -> np.ndarray:
        return np.array([p.expected_sharpe for p in self.periods])

    @property
    def avg_risks(self) -> np.ndarray:
        return np.array([p.avg_risk for p in self.periods])

    @property
    def avg_returns(self) -> np.ndarray:
        return np.array([p.avg_return for p in self.periods])


def period_length(cfg, instance) -> int:
    freq_map = PERIOD_BY_FREQ.get(instance.frequency)
    if freq_map is None:
        raise ValueError(f"No period mapping for frequency {instance.frequency!r}.")
    if cfg.rebalancing not in freq_map:
        raise ValueError(
            f"rebalancing={cfg.rebalancing!r} not supported for "
            f"{instance.frequency} (use one of {list(freq_map)})."
        )
    return freq_map[cfg.rebalancing]


def _rebalancing_indices(cfg, instance, period: int) -> list[int]:
    n_prices = len(instance.prices)
    max_tempo = min(cfg.max_periods, n_prices)

    tempos = list(range(period, max_tempo + 1, period))

    ts = [tempo - 2 for tempo in tempos if tempo - 2 >= 0]

    if cfg.warmup_periods is not None:
        ts = [t for t in ts if t + 1 >= cfg.warmup_periods]

    return ts


def run_backtest(instance, cfg, period: int | None = None) -> dict[str, BacktestResult]:
    """Rolling multi-period backtest: at each rebalancing date, estimate on the past window, run every enabled algorithm and hold until next date.

    In: instance (Instance), cfg (Config), period (int | None rows between rebalances).
    Out: dict[str, BacktestResult] keyed by algorithm name.
    """
    if period is None:
        period = period_length(cfg, instance)

    rebalancing_ts = _rebalancing_indices(cfg, instance, period)
    if not rebalancing_ts:
        raise ValueError(
            f"No rebalancing dates with period={period}, max_periods={cfg.max_periods}."
        )

    active = [n for n, flag in [("Greedy",       cfg.run_greedy),
                                ("GRASP",        cfg.run_grasp),
                                ("Random",       cfg.run_random),
                                ("1/N",          cfg.run_oneoverN),
                                ("EvoFolio",     cfg.run_evofolio),
                                ("GRASP_Pareto", cfg.run_grasp_pareto)] if flag]
    baselines_on = (getattr(cfg, "run_full_oneoverN", False)
                    or getattr(cfg, "run_buy_and_hold", False))
    if not active and not baselines_on:
        raise ValueError("At least one algorithm must be enabled.")

    time_per_period: float | None = None
    if cfg.use_time_budget:
        time_per_period = cfg.time_budget_seconds / len(rebalancing_ts)
        print(f"Time budget: {cfg.time_budget_seconds:.0f}s total per algorithm "
              f"→ {time_per_period:.2f}s per period over {len(rebalancing_ts)} rebalancings")

    results = {name: BacktestResult(name=name, target_period_idx=cfg.target_period_index)
               for name in active}

    cash = {name: float(cfg.budget) for name in active}
    rng = np.random.default_rng(cfg.seed)
    total = len(rebalancing_ts)

    close_prices = instance.prices.values

    if not active:
        rebalancing_ts_iter = []
    else:
        rebalancing_ts_iter = rebalancing_ts

    for i, t in enumerate(rebalancing_ts_iter):
        price_idx = t + 1
        next_price_idx = min(price_idx + period, len(instance.prices) - 1)
        date = instance.prices.index[price_idx]

        print(f"\r  [{i + 1}/{total}] Rebalancing at {date.date()} (t={t})",
              end="", flush=True)

        prices_now  = close_prices[price_idx]
        prices_next = close_prices[next_price_idx]

        if cfg.return_signal == "oracle":
            mean_returns = np.log(prices_next / prices_now)
        elif cfg.return_signal == "last_period":
            start = max(0, t - period + 1)
            mean_returns = instance.returns.iloc[start : t + 1].sum().values
        elif cfg.return_signal == "full_history":
            mean_returns = None
        elif cfg.return_signal == "ewma":
            hist_arr = instance.returns.iloc[: t + 1].values
            T_ewma = len(hist_arr)
            hl = getattr(cfg, "ewma_half_life_weeks", 12)
            lam = 0.5 ** (1.0 / hl)
            w = lam ** np.arange(T_ewma - 1, -1, -1)
            w /= w.sum()
            mean_returns = (hist_arr * w[:, None]).sum(axis=0)
        elif cfg.return_signal == "bl_views_ewma":
            hist_arr = instance.returns.iloc[: t + 1].values
            T_bl = len(hist_arr)
            N_assets = hist_arr.shape[1]
            delta = getattr(cfg, "bl_delta", 2.5)
            tau = getattr(cfg, "bl_tau", 0.05)
            hl = getattr(cfg, "ewma_half_life_weeks", 12)

            Sigma_bl = np.cov(hist_arr, rowvar=False)
            w_market = np.ones(N_assets) / N_assets
            Pi = delta * (Sigma_bl @ w_market)

            lam = 0.5 ** (1.0 / hl)
            w = lam ** np.arange(T_bl - 1, -1, -1)
            w /= w.sum()
            Q_views = (hist_arr * w[:, None]).sum(axis=0)

            Omega = tau * np.diag(np.diag(Sigma_bl))
            tau_Sigma = tau * Sigma_bl
            mid = np.linalg.solve(tau_Sigma + Omega, Q_views - Pi)
            mean_returns = Pi + tau_Sigma @ mid
        else:
            raise ValueError(f"Unknown return_signal: {cfg.return_signal!r}")

        problem = PortfolioProblem(
            instance=instance,
            t=t,
            period=period,
            budget=cfg.budget,
            min_cardinality=cfg.min_cardinality,
            max_cardinality=cfg.max_cardinality,
            min_weight=cfg.min_weight,
            max_weight=cfg.max_weight,
            slsqp_min_weight=cfg.slsqp_min_weight,
            slsqp_max_weight=cfg.slsqp_max_weight,
            risk_free_rate=cfg.risk_free_rate,
            lookback_window=cfg.lookback_window,
            mean_returns=mean_returns,
            weight_method=cfg.weight_method,
            risk_method=cfg.risk_method,
        )

        for name in active:
            sol, candidates, trace = _run_algorithm(
                name, problem, cfg, rng, time_per_period,
            )

            cash_before = cash[name]
            shares = np.floor(sol.weights * cash_before / prices_now)
            spent  = float(np.sum(shares * prices_now))
            cash_left = cash_before - spent

            value_next = float(np.sum(shares * prices_next) + cash_left)
            realized_log = (
                np.log(value_next / cash_before) if cash_before > 0 else 0.0
            )

            cash[name] = value_next

            risks   = np.array([c.risk for c in candidates if np.isfinite(c.risk)])
            returns = np.array([c.expected_return for c in candidates
                                if np.isfinite(c.expected_return)])
            avg_risk   = float(risks.mean())   if risks.size   else float(sol.risk)
            avg_return = float(returns.mean()) if returns.size else float(sol.expected_return)

            if trace is None:
                trace = np.array(
                    [[c.risk, c.expected_return] for c in candidates
                     if np.isfinite(c.risk) and np.isfinite(c.expected_return)],
                    dtype=float,
                )
                if trace.size == 0:
                    trace = np.array([[sol.risk, sol.expected_return]], dtype=float)

            results[name].periods.append(PeriodResult(
                date=date,
                t=t,
                weights=sol.weights.copy(),
                shares=shares,
                cash_left=cash_left,
                expected_sharpe=sol.sharpe,
                realized_log_return=float(realized_log),
                portfolio_value=value_next,
                avg_risk=avg_risk,
                avg_return=avg_return,
                n_candidates=len(candidates),
                trace=trace,
            ))

            if cfg.target_period_index is not None and i == cfg.target_period_index:
                results[name].target_candidates = candidates

    print()

    if getattr(cfg, "run_full_oneoverN", False) or getattr(cfg, "run_buy_and_hold", False):
        from src.baselines import (
            run_full_one_over_n, run_buy_and_hold_one_over_n,
        )
        if getattr(cfg, "run_full_oneoverN", False):
            results["Full 1/N"] = run_full_one_over_n(instance, cfg, period)
        if getattr(cfg, "run_buy_and_hold", False):
            results["Buy&Hold 1/N"] = run_buy_and_hold_one_over_n(instance, cfg, period)

    return results


def _run_algorithm(name: str, problem, cfg, rng: np.random.Generator,
                   time_per_period: float | None = None):
    n_iters_grasp = None if time_per_period is not None else cfg.backtest_grasp_iterations
    pop_size      = None if time_per_period is not None else cfg.backtest_random_solutions

    if name == "Greedy":
        sol, candidates = GreedyAlgorithm(problem).run()
        if cfg.use_local_search:
            sol = swap_local_search(sol, problem)
            candidates = [sol]
        return sol, candidates, None

    if name == "GRASP":
        sol, candidates = GRASPAlgorithm(
            problem,
            n_iterations=n_iters_grasp,
            time_limit=time_per_period,
            alpha=cfg.grasp_alpha,
            seed=int(rng.integers(0, 2**31)),
        ).run()
        if cfg.use_local_search:
            sol = swap_local_search(sol, problem)
        return sol, candidates, None

    if name == "1/N":
        sol, candidates = OneOverNAlgorithm(
            problem,
            population_size=pop_size,
            time_limit=time_per_period,
            seed=int(rng.integers(0, 2**31)),
        ).run()
        return sol, candidates, None

    if name == "GRASP_Pareto":
        n_iters_pgrasp = (None if time_per_period is not None
                          else cfg.backtest_pareto_grasp_iterations)
        sol, candidates = GRASPParetoAlgorithm(
            problem,
            n_iterations=n_iters_pgrasp,
            time_limit=time_per_period,
            alpha=cfg.pareto_grasp_alpha,
            seed=int(rng.integers(0, 2**31)),
        ).run()
        return sol, candidates, None

    if name == "EvoFolio":
        n_gens = None if time_per_period is not None else cfg.evofolio_generations
        algo = EvoFolioAlgorithm(
            problem,
            population_size=cfg.evofolio_population_size,
            n_generations=n_gens,
            crossover_rate=cfg.evofolio_crossover_rate,
            indpb=cfg.evofolio_indpb,
            tournparam=cfg.evofolio_tournparam,
            selparam=cfg.evofolio_selparam,
            elite_cxpb=cfg.evofolio_elite_cxpb,
            time_limit=time_per_period,
            seed=int(rng.integers(0, 2**31)),
        )
        sol, candidates = algo.run()
        trace = (np.asarray(algo.gen_history, dtype=float)
                 if algo.gen_history else None)
        return sol, candidates, trace

    sol, candidates = RandomFeasibleAlgorithm(
        problem,
        population_size=pop_size,
        time_limit=time_per_period,
        seed=int(rng.integers(0, 2**31)),
    ).run()
    return sol, candidates, None
