import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import Config
from src.instance import Instance
from src.multiseed import run_multiseed


ALL_OFF = dict(
    run_greedy=False, run_grasp=False, run_random=False,
    run_oneoverN=False, run_evofolio=False, run_grasp_pareto=False,
    run_full_oneoverN=False, run_buy_and_hold=False,
)

EXPERIMENTS = [
    ("Full 1/N completo",         "Full 1/N",       1,
        {**ALL_OFF, "run_full_oneoverN": True}),

    ("Buy & Hold 1/N",            "Buy&Hold 1/N",   1,
        {**ALL_OFF, "run_buy_and_hold": True}),

    ("Random Dirichlet",          "Random",         10,
        {**ALL_OFF, "run_random": True}),

    ("Random 1/N (K∈[10,14])",    "1/N",            10,
        {**ALL_OFF, "run_oneoverN": True}),

    ("Greedy (equal)",            "Greedy",         1,
        {**ALL_OFF, "run_greedy": True, "weight_method": "equal"}),

    ("Greedy (sharpe_max)",       "Greedy",         1,
        {**ALL_OFF, "run_greedy": True, "weight_method": "sharpe_max"}),

    ("GRASP (equal)",             "GRASP",          10,
        {**ALL_OFF, "run_grasp": True, "weight_method": "equal"}),

    ("GRASP (sharpe_max)",        "GRASP",          10,
        {**ALL_OFF, "run_grasp": True, "weight_method": "sharpe_max"}),

    ("Pareto-GRASP",              "GRASP_Pareto",   10,
        {**ALL_OFF, "run_grasp_pareto": True, "weight_method": "sharpe_max",
         "risk_method": "markowitz"}),

    ("NSGA-II EvoFolio (riesgo EVF)",       "EvoFolio", 10,
        {**ALL_OFF, "run_evofolio": True, "risk_method": "evofolio"}),

    ("NSGA-II EvoFolio (riesgo Markowitz)", "EvoFolio", 10,
        {**ALL_OFF, "run_evofolio": True, "risk_method": "markowitz"}),
]

REBALANCINGS = ["monthly", "quarterly"]

BASE_SEEDS = list(range(10))


def run_one_experiment(base_cfg: Config, instance, display_name: str,
                       result_key: str, n_seeds: int, overrides: dict,
                       rebalancing: str) -> pd.DataFrame:
    """Run one (variant, rebalancing) experiment across n_seeds.

    Out: pd.DataFrame of per-seed metrics.
    """
    cfg = dataclasses.replace(base_cfg,
                              rebalancing=rebalancing,
                              show_plots=False,
                              **overrides)
    seeds = BASE_SEEDS[:n_seeds]
    print("\n========================================================")
    print(f"  {display_name}  ·  {rebalancing}  ·  {n_seeds} seed(s)")
    print("========================================================")
    df = run_multiseed(cfg, seeds=seeds, instance=instance, verbose=False)

    df = df[df["algorithm"] == result_key].copy()
    df["variant"]     = display_name
    df["rebalancing"] = rebalancing
    return df


def main():
    out_dir = Path("results") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Config()
    instance = Instance.from_folder(
        data_root=base_cfg.data_root,
        frequency=base_cfg.frequency,
        price_col=base_cfg.price_col,
    )
    print(f"Loaded {instance.frequency} data: {len(instance.assets)} assets, "
          f"{len(instance.returns)} return observations.")
    print(f"Total experiments: {len(EXPERIMENTS)} variants × {len(REBALANCINGS)} "
          f"rebalancings = {len(EXPERIMENTS) * len(REBALANCINGS)} runs.")

    all_rows: list[pd.DataFrame] = []

    for rebal in REBALANCINGS:
        per_rebal_rows: list[pd.DataFrame] = []
        for display_name, result_key, n_seeds, overrides in EXPERIMENTS:
            df = run_one_experiment(
                base_cfg, instance, display_name,
                result_key, n_seeds, overrides, rebal,
            )
            per_rebal_rows.append(df)
            all_rows.append(df)

        long_df = pd.concat(per_rebal_rows, ignore_index=True)
        long_df.to_csv(out_dir / f"long_form_{rebal}.csv", index=False)

        numeric = ["final_value", "total_return", "annualised_return",
                   "annualised_vol", "sharpe", "max_drawdown"]
        summary = long_df.groupby("variant")[numeric].agg(["mean", "std"])
        summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(out_dir / f"summary_{rebal}.csv", index=False)

        print(f"\n──── Summary [{rebal}] ────")
        print(summary.to_string(index=False))

    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(out_dir / "master_table.csv", index=False)
    print(f"\nMaster table written: {out_dir / 'master_table.csv'}")
    print(f"  → {len(master)} rows  ({len(EXPERIMENTS)} variants × "
          f"{len(REBALANCINGS)} rebalancings × seeds)")


if __name__ == "__main__":
    main()
