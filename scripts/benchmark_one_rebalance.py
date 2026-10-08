import sys, time, dataclasses
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from src.instance import Instance
from src.backtest import run_backtest, period_length, _rebalancing_indices


def main():
    cfg_full = Config()
    instance = Instance.from_folder(
        data_root=cfg_full.data_root,
        frequency=cfg_full.frequency,
        price_col=cfg_full.price_col,
    )
    period = period_length(cfg_full, instance)
    full_ts = _rebalancing_indices(cfg_full, instance, period)
    n_full = len(full_ts)

    cfg = dataclasses.replace(
        cfg_full,
        max_periods=period * 2,
        run_greedy=False, run_grasp=True,
        run_random=False, run_oneoverN=False,
        run_evofolio=True, run_grasp_pareto=True,
        run_full_oneoverN=False, run_buy_and_hold=False,
        show_plots=False,
    )

    bench_ts = _rebalancing_indices(cfg, instance, period)
    n_bench = len(bench_ts)
    print(f"Benchmark over {n_bench} rebalance(s); full horizon = {n_full}.")
    print(f"period={period} weeks, max_periods bench={cfg.max_periods}, "
          f"weight_method={cfg.weight_method}, return_signal={cfg.return_signal}")

    t0 = time.time()
    results = run_backtest(instance, cfg, period=period)
    elapsed = time.time() - t0

    per_rebalance = elapsed / max(n_bench, 1)
    projected = per_rebalance * n_full

    print()
    print(f"Wall-clock for {n_bench} rebalance(s): {elapsed:.2f} s")
    print(f"  → per-rebalance avg:                {per_rebalance:.2f} s")
    print(f"  → projected for {n_full} rebalances:  "
          f"{projected:.1f} s ≈ {projected/60:.1f} min ≈ {projected/3600:.2f} h")
    print()
    print("Algorithms executed:", list(results.keys()))


if __name__ == "__main__":
    main()
