import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import io
import pickle
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from config import Config
from src.instance import Instance
from src.backtest import run_backtest, period_length
from src.metrics import compute_metrics


N_SEEDS   = 5
SEEDS     = list(range(N_SEEDS))
N_WORKERS = 5

PRESETS = {
    "full_history": dict(rebalancing="monthly", out="multiseed_random_pareto_evofolio_monthly",
                         cfg=dict(return_signal="full_history")),
    "ewma":         dict(rebalancing="monthly", out="multiseed_ewma_monthly",
                         cfg=dict(return_signal="ewma", ewma_half_life_weeks=12)),
    "bl_quarterly": dict(rebalancing="quarterly", out="multiseed_bl_views_ewma_quarterly",
                         cfg=dict(return_signal="bl_views_ewma", ewma_half_life_weeks=12,
                                  bl_delta=2.5, bl_tau=0.05)),
}


def _run_one_job(args):
    seed, rebal, base_cfg, per_seed_dir = args
    cfg = dataclasses.replace(base_cfg, seed=int(seed), rebalancing=rebal)

    instance = Instance.from_folder(
        data_root=cfg.data_root,
        frequency=cfg.frequency,
        price_col=cfg.price_col,
    )

    t0 = _time.time()
    with redirect_stdout(io.StringIO()):
        results = run_backtest(instance, cfg)
    elapsed = _time.time() - t0

    period = period_length(cfg, instance)
    ppy = instance.periods_per_year / period

    payload = {
        "seed":              int(seed),
        "rebalancing":       rebal,
        "period":            period,
        "periods_per_year":  ppy,
        "budget":            cfg.budget,
        "return_signal":     cfg.return_signal,
        "ewma_half_life":    cfg.ewma_half_life_weeks,
        "bl_delta":          cfg.bl_delta,
        "bl_tau":            cfg.bl_tau,
        "elapsed_s":         elapsed,
        "assets":            list(instance.assets),
        "results":           results,
    }
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    with open(per_seed_dir / f"seed_{seed}_{rebal}.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for name, res in results.items():
        vals = np.concatenate([[cfg.budget], res.portfolio_values])
        m = compute_metrics(vals, ppy, cfg.risk_free_rate)
        m["algorithm"]   = name
        m["seed"]        = int(seed)
        m["rebalancing"] = rebal
        m["elapsed_s"]   = elapsed
        rows.append(m)
    return rows


def main() -> None:
    """Run N_SEEDS backtests in parallel for one preset and save long_form.csv, summary.csv and per-seed pickles.

    In: CLI arg `preset` (full_history | ewma | bl_quarterly).
    Out: None (writes to results/<preset output dir>/).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=PRESETS)
    preset = PRESETS[parser.parse_args().preset]
    REBALANCINGS = [preset["rebalancing"]]
    OUTPUT_DIR   = REPO_ROOT / "results" / preset["out"]
    PER_SEED_DIR = OUTPUT_DIR / "per_seed"
    PER_SEED_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = dataclasses.replace(
        Config(),
        run_evofolio         = True,
        run_grasp_pareto     = True,
        run_random           = True,
        run_greedy           = False,
        run_grasp            = False,
        run_oneoverN         = False,
        run_full_oneoverN    = False,
        run_buy_and_hold     = False,
        weight_method        = "sharpe_max",
        risk_method          = "markowitz",
        show_plots           = False,
        **preset["cfg"],
    )

    work = [(seed, rebal, base_cfg, PER_SEED_DIR)
            for rebal in REBALANCINGS for seed in SEEDS]

    print(f"Parallel multi-seed backtest — {preset['cfg']}")
    print(f"  seeds        : {SEEDS}")
    print(f"  rebalancings : {REBALANCINGS}")
    print("  algorithms   : EvoFolio, Pareto-GRASP, Random")
    print(f"  workers      : {N_WORKERS} (BLAS threads pinned to 1 each)")
    print(f"  total jobs   : {len(work)}")
    print(f"  output       : {OUTPUT_DIR}/")
    print(f"  per-seed pkl : {PER_SEED_DIR}/\n")

    t_start = _time.time()
    all_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_one_job, w): (w[0], w[1]) for w in work}

        for i, fut in enumerate(as_completed(futures)):
            seed, rebal = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                elap = rows[0]["elapsed_s"] / 60
                print(f"  ✓ [{i+1}/{len(work)}] seed={seed} rebal={rebal} "
                      f"finished ({elap:.1f} min) "
                      f"— pickle: seed_{seed}_{rebal}.pkl")
                pd.DataFrame(all_rows).to_csv(
                    OUTPUT_DIR / "long_form.csv", index=False,
                )
            except Exception as e:
                print(f"  ✗ [{i+1}/{len(work)}] seed={seed} rebal={rebal} "
                      f"FAILED: {e!r}")

    wall = _time.time() - t_start
    print(f"\nWall-clock total: {wall/60:.1f} min ({wall/3600:.2f} h)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "long_form.csv", index=False)

    numeric = ["final_value", "total_return", "annualised_return",
               "annualised_vol", "sharpe", "max_drawdown"]
    summary = (df.groupby(["rebalancing", "algorithm"])[numeric]
                 .agg(["mean", "std"]))
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n── Summary (mean ± std over seeds) ──")
    print(summary.to_string(index=False))
    print(f"\nAll {N_SEEDS} pickles saved under {PER_SEED_DIR}/")


if __name__ == "__main__":
    main()
