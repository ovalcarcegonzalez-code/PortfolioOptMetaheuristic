from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def prediction_metrics(df: pd.DataFrame) -> dict:
    """Aggregate prediction metrics (IC, RankIC, directional accuracy, RMSE...).

    In: pd.DataFrame from evaluate_estimator.
    Out: dict.
    """
    p = df["predicted"].values
    r = df["realized"].values

    err = p - r
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    ic = float(np.corrcoef(p, r)[0, 1]) if len(p) > 1 else 0.0

    rho, _ = spearmanr(p, r)
    rank_ic = float(rho) if not np.isnan(rho) else 0.0

    dir_acc = float(np.mean(np.sign(p) == np.sign(r)))

    return {
        "rmse":    rmse,
        "mae":     mae,
        "r2":      r2,
        "ic":      ic,
        "rank_ic": rank_ic,
        "dir_acc": dir_acc,
    }


def cross_sectional_ic(df: pd.DataFrame) -> pd.Series:
    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return np.nan
        p = g["predicted"].values
        r = g["realized"].values
        if p.std() < 1e-12 or r.std() < 1e-12:
            return np.nan
        return float(np.corrcoef(p, r)[0, 1])
    return df.groupby("date").apply(_ic)
