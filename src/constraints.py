import numpy as np


EPS = 1e-8


def weights_sum_to_one(weights: np.ndarray, tol: float = 1e-6) -> bool:
    return abs(weights.sum() - 1.0) <= tol


def no_short_selling(weights: np.ndarray, tol: float = 1e-8) -> bool:
    return np.all(weights >= -tol)


def max_weight_constraint(weights: np.ndarray, max_weight: float, tol: float = 1e-8) -> bool:
    selected = weights[weights > EPS]
    if len(selected) == 0:
        return False
    return np.all(selected <= max_weight + tol)


def min_weight_constraint(weights: np.ndarray, min_weight: float, tol: float = 1e-8) -> bool:
    selected = weights[weights > EPS]
    if len(selected) == 0:
        return False
    return np.all(selected >= min_weight - tol)


def cardinality_in_range(
    weights: np.ndarray,
    min_assets: int,
    max_assets: int,
    tol: float = 1e-8,
) -> bool:
    k = int(np.count_nonzero(weights > tol))
    return min_assets <= k <= max_assets


def is_feasible_weights(
    weights: np.ndarray,
    min_weight: float,
    max_weight: float,
    min_cardinality: int,
    max_cardinality: int,
) -> bool:
    """Check sum-to-one, no shorting, weight bounds and cardinality.

    Out: bool.
    """
    return (
        no_short_selling(weights)
        and weights_sum_to_one(weights)
        and min_weight_constraint(weights, min_weight)
        and max_weight_constraint(weights, max_weight)
        and cardinality_in_range(weights, min_cardinality, max_cardinality)
    )


def sample_bounded_weights(
    n_selected: int,
    min_weight: float,
    max_weight: float,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Random weights summing to 1 within [min_weight, max_weight].

    In: n_selected (int), min_weight, max_weight, rng.
    Out: np.ndarray.
    """
    if n_selected * min_weight > 1.0:
        raise ValueError("Infeasible constraint: n_selected * min_weight > 1")

    if n_selected * max_weight < 1.0:
        raise ValueError("Infeasible constraint: n_selected * max_weight < 1")

    for _ in range(max_attempts):
        weights = rng.dirichlet(np.ones(n_selected))

        if np.all(weights >= min_weight) and np.all(weights <= max_weight):
            return weights

    raise RuntimeError(
        "Could not generate feasible bounded weights. "
        "Try relaxing min_weight/max_weight or changing cardinality."
    )
