import os
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_solutions_to_csv(solutions, assets, path: str) -> None:
    rows = []

    for i, sol in enumerate(solutions):
        row = {
            "solution_id": i,
            "expected_return": sol.expected_return,
            "risk": sol.risk,
            "sharpe": sol.sharpe,
            "feasible": sol.feasible,
            "cardinality": sol.cardinality(),
        }

        for asset, weight in zip(assets, sol.weights):
            row[asset] = weight

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)