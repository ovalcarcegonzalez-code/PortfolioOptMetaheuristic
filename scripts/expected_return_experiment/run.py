from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.instance import Instance
from scripts.expected_return_experiment.estimators import ESTIMATORS
from scripts.expected_return_experiment.evaluate import evaluate_estimator
from scripts.expected_return_experiment.metrics import (
    prediction_metrics, cross_sectional_ic,
)
from scripts.expected_return_experiment.plots import (
    plot_scatter_grid,
    plot_metrics_bars,
    plot_cross_sectional_ic,
    plot_per_asset_series,
    plot_price_with_predictions,
)


DATA_ROOT      = "data/stock"
FREQUENCY      = "WEEK"
PRICE_COL      = "Close"
OUTPUT_DIR     = REPO_ROOT / "results" / "expected_return_study"

FREQ_CONFIGS = {
    "monthly":   {"period": 4,  "label": "Monthly (4-week horizon)"},
    "quarterly": {"period": 12, "label": "Quarterly (12-week horizon)"},
}

WARMUP_WEEKS = 52

SAMPLE_ASSETS_DEFAULT = ["AAPL", "TSLA", "MSFT", "GOOG"]


def _pick_sample_assets(universe: list[str]) -> list[str]:
    picked = [a for a in SAMPLE_ASSETS_DEFAULT if a in universe]
    for a in universe:
        if len(picked) >= 4:
            break
        if a not in picked:
            picked.append(a)
    return picked[:4]


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instance = Instance.from_folder(
        data_root=DATA_ROOT, frequency=FREQUENCY, price_col=PRICE_COL,
    )
    print(f"Loaded {instance.frequency} data: {len(instance.assets)} assets, "
          f"{len(instance.returns)} return observations.")

    sample_assets = _pick_sample_assets(list(instance.assets))
    print(f"Sample assets for per-asset plots: {sample_assets}")

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict] = []

    for freq_name, freq_cfg in FREQ_CONFIGS.items():
        period = freq_cfg["period"]
        label = freq_cfg["label"]

        print(f"\n{'=' * 72}")
        print(f"  {label} — period={period} weeks, warmup={WARMUP_WEEKS} weeks")
        print(f"{'=' * 72}")

        dfs_by_estimator: dict[str, pd.DataFrame] = {}
        metrics_by_estimator: dict[str, dict] = {}
        ic_by_estimator: dict[str, pd.Series] = {}

        for name, (fn, kwargs, _human) in ESTIMATORS.items():
            df = evaluate_estimator(
                instance, fn, name,
                period=period, warmup_weeks=WARMUP_WEEKS, **kwargs,
            )
            dfs_by_estimator[name] = df

            m = prediction_metrics(df)
            metrics_by_estimator[name] = m

            ic_by_estimator[name] = cross_sectional_ic(df)

            df_out = df.copy()
            df_out["frequency"] = freq_name
            all_predictions.append(df_out)

            row = {"estimator": name, "frequency": freq_name, **m}
            all_metrics.append(row)

        print(f"\n  Prediction quality — {label}")
        print(f"  {'Estimator':<20}{'RMSE':>8}{'MAE':>8}"
              f"{'IC':>10}{'RankIC':>10}{'DirAcc':>10}{'R²':>10}")
        print(f"  {'-' * 76}")
        for name, m in metrics_by_estimator.items():
            print(f"  {name:<20}"
                  f"{m['rmse']:>8.4f}{m['mae']:>8.4f}"
                  f"{m['ic']:>+10.3f}{m['rank_ic']:>+10.3f}"
                  f"{m['dir_acc']:>10.2%}{m['r2']:>+10.3f}")

        freq_dir = OUTPUT_DIR / freq_name
        freq_dir.mkdir(exist_ok=True)

        plot_scatter_grid(
            dfs_by_estimator, metrics_by_estimator,
            label, freq_dir / "scatter_grid.png",
        )
        plot_metrics_bars(
            metrics_by_estimator, label, freq_dir / "metrics_bars.png",
        )
        plot_cross_sectional_ic(
            ic_by_estimator, label, freq_dir / "ic_over_time.png",
        )
        plot_per_asset_series(
            dfs_by_estimator, sample_assets, label,
            freq_dir / "per_asset_series.png",
        )
        plot_price_with_predictions(
            instance, dfs_by_estimator, sample_assets, label,
            freq_dir / "price_with_predictions.png",
        )

        pd.DataFrame(metrics_by_estimator).T.to_csv(
            freq_dir / "metrics_summary.csv",
        )
        print(f"\n  Saved plots + summary to: {freq_dir}/")

    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions_full.csv", index=False,
    )
    pd.DataFrame(all_metrics).to_csv(
        OUTPUT_DIR / "summary_all.csv", index=False,
    )
    print(f"\nAll outputs under {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
