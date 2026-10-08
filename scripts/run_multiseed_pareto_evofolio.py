import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from config import Config
from src.instance import Instance
from src.multiseed import run_multiseed, summary_table


N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
REBALANCINGS = ["monthly", "quarterly"]

OUTPUT_DIR = REPO_ROOT / "results" / "multiseed_pareto_evofolio"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = dataclasses.replace(
        Config(),
        run_evofolio=True,
        run_grasp_pareto=True,
        run_full_oneoverN=True,
        run_buy_and_hold=True,
        run_greedy=False,
        run_grasp=False,
        run_random=False,
        run_oneoverN=False,
        return_signal="full_history",
        weight_method="sharpe_max",
        risk_method="markowitz",
        show_plots=False,
    )

    instance = Instance.from_folder(
        data_root=base_cfg.data_root,
        frequency=base_cfg.frequency,
        price_col=base_cfg.price_col,
    )
    print(f"Loaded {instance.frequency}: {len(instance.assets)} assets, "
          f"{len(instance.returns)} return observations.")
    print(f"Seeds: {SEEDS}   Rebalancings: {REBALANCINGS}")

    all_rows: list[pd.DataFrame] = []

    for rebal in REBALANCINGS:
        print(f"\n{'=' * 72}")
        print(f"  {rebal.upper()} rebalancing — {N_SEEDS} seeds")
        print(f"{'=' * 72}")
        cfg = dataclasses.replace(base_cfg, rebalancing=rebal)

        df = run_multiseed(cfg, seeds=SEEDS, instance=instance, verbose=True)
        df["rebalancing"] = rebal
        all_rows.append(df)
        df.to_csv(OUTPUT_DIR / f"long_form_{rebal}.csv", index=False)

        summary = summary_table(df)
        summary["rebalancing"] = rebal
        summary.to_csv(OUTPUT_DIR / f"summary_{rebal}.csv", index=False)
        print(f"\n── Summary [{rebal}] ──")
        print(summary.to_string(index=False))

    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(OUTPUT_DIR / "master_table.csv", index=False)
    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
