from __future__ import annotations

import numpy as np
import pandas as pd


def rebalancing_indices(
    n_prices: int,
    period: int,
    warmup_weeks: int = 52,
) -> list[int]:
    tempos = list(range(period, n_prices + 1, period))
    ts = [tempo - 2 for tempo in tempos if tempo - 2 >= 0]
    ts = [t for t in ts if t + 1 + period <= n_prices - 1]
    ts = [t for t in ts if t + 1 >= warmup_weeks]
    return ts


def evaluate_estimator(
    instance,
    estimator_fn,
    estimator_name: str,
    period: int,
    warmup_weeks: int = 52,
    **estimator_kwargs,
) -> pd.DataFrame:
    """Apply an expected-return estimator at every rebalancing date and compare with realised returns.

    In: instance, estimator_fn, estimator_name, period, warmup_weeks, **kwargs.
    Out: pd.DataFrame (date, asset, predicted, realised).
    """
    prices = instance.prices.values
    returns_arr = instance.returns.values
    n_prices = len(prices)
    assets = list(instance.assets)

    ts = rebalancing_indices(n_prices, period, warmup_weeks)

    records = []
    for t in ts:
        returns_history = returns_arr[: t + 1]

        predicted = estimator_fn(returns_history, period, **estimator_kwargs)

        price_idx = t + 1
        next_price_idx = price_idx + period
        realized = np.log(prices[next_price_idx] / prices[price_idx])

        date = instance.prices.index[price_idx]

        for i, asset in enumerate(assets):
            records.append({
                "date":      date,
                "t":         int(t),
                "asset":     asset,
                "predicted": float(predicted[i]),
                "realized":  float(realized[i]),
                "estimator": estimator_name,
                "period":    period,
            })

    return pd.DataFrame(records)
