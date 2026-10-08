from dataclasses import dataclass
import numpy as np


@dataclass
class Solution:
    weights: np.ndarray

    expected_return: float | None = None
    risk: float | None = None
    sharpe: float | None = None
    feasible: bool | None = None

    def selected_assets(self, eps: float = 1e-8) -> np.ndarray:
        return np.where(self.weights > eps)[0]

    def cardinality(self, eps: float = 1e-8) -> int:
        return len(self.selected_assets(eps))

    def __repr__(self) -> str:
        return (
            f"Solution("
            f"return={self.expected_return:.6f}, "
            f"risk={self.risk:.6f}, "
            f"sharpe={self.sharpe:.4f}, "
            f"cardinality={self.cardinality()}, "
            f"feasible={self.feasible})"
        )
