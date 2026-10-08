from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from scripts.expected_return_experiment.estimators import (
    ewma,
    shrinkage_toward_zero,
    black_litterman_with_views,
    black_litterman_with_momentum_views,
)
from scripts.expected_return_experiment.evaluate import evaluate_estimator
from scripts.expected_return_experiment.metrics import prediction_metrics


@dataclass
class SweepSpec:
    label: str
    estimator_fn: Callable
    param_name: str
    param_values: list
    fixed_kwargs: dict = field(default_factory=dict)


SWEEPS: list[SweepSpec] = [
    SweepSpec(
        label="EWMA — half_life_weeks",
        estimator_fn=ewma,
        param_name="half_life_weeks",
        param_values=[4, 8, 12, 26, 52, 104],
    ),
    SweepSpec(
        label="Shrinkage toward 0 — alpha",
        estimator_fn=shrinkage_toward_zero,
        param_name="alpha",
        param_values=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    ),
    SweepSpec(
        label="BL EWMA-views — tau",
        estimator_fn=black_litterman_with_views,
        param_name="tau",
        param_values=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
        fixed_kwargs={"delta": 2.5, "half_life_weeks": 12},
    ),
    SweepSpec(
        label="BL EWMA-views — half_life_weeks",
        estimator_fn=black_litterman_with_views,
        param_name="half_life_weeks",
        param_values=[4, 8, 12, 26, 52],
        fixed_kwargs={"delta": 2.5, "tau": 0.05},
    ),
    SweepSpec(
        label="BL MOM-views — tau",
        estimator_fn=black_litterman_with_momentum_views,
        param_name="tau",
        param_values=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
        fixed_kwargs={"delta": 2.5},
    ),
]


def run_sweep(
    instance,
    spec: SweepSpec,
    period: int,
    warmup_weeks: int = 52,
) -> pd.DataFrame:
    """Evaluate an estimator across all values of one hyperparameter.

    In: instance, spec (SweepSpec), period, warmup_weeks.
    Out: pd.DataFrame of metrics per value.
    """
    rows = []
    for value in spec.param_values:
        kwargs = {**spec.fixed_kwargs, spec.param_name: value}
        df = evaluate_estimator(
            instance,
            spec.estimator_fn,
            estimator_name=f"{spec.estimator_fn.__name__}_{spec.param_name}={value}",
            period=period,
            warmup_weeks=warmup_weeks,
            **kwargs,
        )
        m = prediction_metrics(df)
        rows.append({
            "param_value": value,
            "n_rows":      int(len(df)),
            **m,
        })
    return pd.DataFrame(rows)
