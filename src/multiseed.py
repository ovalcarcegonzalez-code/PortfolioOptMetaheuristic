from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import pandas as pd

from src.instance import Instance
from src.backtest import run_backtest, period_length
from src.metrics import compute_metrics


def _series_with_initial(result, initial_value: float) -> np.ndarray:
    return np.concatenate([[initial_value], result.portfolio_values])


def metrics_for_result(result, cfg, instance) -> dict:
    period = period_length(cfg, instance)
    periods_per_year = instance.periods_per_year / period
    values = _series_with_initial(result, cfg.budget)
    return compute_metrics(values, periods_per_year=periods_per_year,
                           rf=cfg.risk_free_rate)


def run_multiseed(
    cfg,
    seeds: Iterable[int],
    instance: Instance | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the backtest once per seed and collect metrics.

    In: cfg, seeds (Iterable[int]), instance (Instance | None).
    Out: long-form pd.DataFrame (seed, algorithm, metrics).
    """
    if instance is None:
        instance = Instance.from_folder(
            data_root=cfg.data_root,
            frequency=cfg.frequency,
            price_col=cfg.price_col,
        )

    rows: list[dict] = []
    seeds = list(seeds)

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n── Multi-seed run {i + 1}/{len(seeds)} (seed={seed}) ──")

        seeded_cfg = dataclasses.replace(cfg, seed=int(seed))
        results = run_backtest(instance, seeded_cfg)

        for name, res in results.items():
            m = metrics_for_result(res, seeded_cfg, instance)
            m["algorithm"] = name
            m["seed"]      = int(seed)
            rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["algorithm", "seed", "final_value", "total_return",
            "annualised_return", "annualised_vol", "sharpe",
            "max_drawdown", "n_periods"]
    return df[cols]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed results to mean/std per algorithm.

    In: pd.DataFrame from run_multiseed.
    Out: pd.DataFrame.
    """
    numeric = ["final_value", "total_return", "annualised_return",
               "annualised_vol", "sharpe", "max_drawdown"]
    grouped = df.groupby("algorithm")[numeric].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()
