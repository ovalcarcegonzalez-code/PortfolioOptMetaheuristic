import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses
from config import Config
from main import main

cfg = dataclasses.replace(
    Config(),
    rebalancing="monthly",
    run_greedy=True,
    run_grasp=True,
    run_random=True,
    run_oneoverN=True,
    run_evofolio=False,
    run_grasp_pareto=False,
    run_full_oneoverN=False,
    run_buy_and_hold=True,
    return_signal="full_history",
    weight_method="sharpe_max",
    show_plots=False,
)

main(cfg)
