from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.instance import Instance
from scripts.expected_return_experiment.sensitivity import SWEEPS, run_sweep
from scripts.expected_return_experiment.plots import plot_sensitivity_grid


DATA_ROOT   = "data/stock"
FREQUENCY   = "WEEK"
PRICE_COL   = "Close"
OUTPUT_DIR  = REPO_ROOT / "results" / "expected_return_study" / "sensitivity"
WARMUP_WEEKS = 52

FREQ_CONFIGS = {
    "Monthly (4-week horizon)":   4,
    "Quarterly (12-week horizon)": 12,
}


def _slug(label: str) -> str:
    return (label.replace(" ", "_")
                 .replace("—", "_")
                 .replace("/", "_")
                 .replace("(", "").replace(")", "")
                 .replace(",", "")
                 .lower())


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instance = Instance.from_folder(
        data_root=DATA_ROOT, frequency=FREQUENCY, price_col=PRICE_COL,
    )
    print(f"Loaded {instance.frequency} data: {len(instance.assets)} assets, "
          f"{len(instance.returns)} return observations.")

    all_rows: list[pd.DataFrame] = []

    for spec in SWEEPS:
        print(f"\n{'=' * 72}")
        print(f"  Sweep: {spec.label}")
        print(f"  Param: {spec.param_name} ∈ {spec.param_values}")
        print(f"{'=' * 72}")

        results_by_freq: dict[str, pd.DataFrame] = {}

        for freq_label, period in FREQ_CONFIGS.items():
            df = run_sweep(instance, spec, period=period,
                           warmup_weeks=WARMUP_WEEKS)
            results_by_freq[freq_label] = df

            print(f"\n  {freq_label}")
            print(f"    {spec.param_name:>18} {'IC':>10} {'RankIC':>10} "
                  f"{'DirAcc':>10} {'RMSE':>10}")
            for _, row in df.iterrows():
                print(f"    {row['param_value']:>18} "
                      f"{row['ic']:>+10.3f} {row['rank_ic']:>+10.3f} "
                      f"{row['dir_acc']:>10.2%} {row['rmse']:>10.4f}")

            df_out = df.copy()
            df_out["sweep"] = spec.label
            df_out["frequency"] = freq_label
            all_rows.append(df_out)

        vals = spec.param_values
        logx = (
            len(vals) > 1
            and min(vals) > 0
            and (max(vals) / min(vals)) >= 10
        )

        out_path = OUTPUT_DIR / f"sensitivity_{_slug(spec.label)}.png"
        plot_sensitivity_grid(
            results_by_freq,
            sweep_label=spec.label,
            param_name=spec.param_name,
            save_path=out_path,
            logx=logx,
        )
        print(f"\n  Plot saved: {out_path}")

    merged = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "sensitivity_all.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\nFull sensitivity table: {csv_path}")


if __name__ == "__main__":
    run()
