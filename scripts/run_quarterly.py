import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses
from config import Config
from main import main

cfg = dataclasses.replace(
    Config(),
    rebalancing="quarterly",
    run_evofolio=True,
    run_grasp_pareto=True,
    run_full_oneoverN=False,
    run_buy_and_hold=True,
    run_greedy=False,
    run_grasp=False,
    run_random=False,
    run_oneoverN=False,
    return_signal="full_history",
    show_plots=False,
)

main(cfg)
