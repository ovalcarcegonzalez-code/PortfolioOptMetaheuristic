import numpy as np

from src.solution import Solution


EPS = 1e-10


def swap_local_search(
    solution: Solution,
    problem,
    max_passes: int | None = None,
) -> Solution:
    """Best-improvement 1-swap local search on the selected assets.

    In: solution (Solution), problem (PortfolioProblem), max_passes.
    Out: improved Solution.
    """
    current = Solution(weights=solution.weights.copy())
    problem.evaluate(current)

    passes = 0
    while True:
        if max_passes is not None and passes >= max_passes:
            return current
        passes += 1

        weights = current.weights
        selected_idx   = np.where(weights > EPS)[0]
        unselected_idx = np.where(weights <= EPS)[0]

        best_candidate: Solution | None = None
        best_sharpe = current.sharpe

        for out_stock in selected_idx:
            w_out = float(weights[out_stock])
            for in_stock in unselected_idx:
                new_w = weights.copy()
                new_w[in_stock]  = w_out
                new_w[out_stock] = 0.0

                candidate = Solution(weights=new_w)
                problem.evaluate(candidate)

                if candidate.sharpe > best_sharpe + EPS:
                    best_sharpe = candidate.sharpe
                    best_candidate = candidate

        if best_candidate is None:
            return _final_reoptimize(current, problem)
        current = best_candidate


def _final_reoptimize(solution: Solution, problem) -> Solution:
    if getattr(problem, "weight_method", "sharpe_max") != "sharpe_max":
        return solution

    selected = np.where(solution.weights > EPS)[0].tolist()
    if not selected:
        return solution

    eval_subset = problem._make_sharpe_evaluator()
    final_w, _ = eval_subset(selected, constrained=True)
    if not np.all(np.isfinite(final_w)):
        return solution

    weights = np.zeros(problem.n_assets)
    weights[selected] = final_w
    reoptimised = Solution(weights=weights)
    problem.evaluate(reoptimised)

    if reoptimised.sharpe >= solution.sharpe - 1e-10:
        return reoptimised
    return solution


def maybe_apply_local_search(solution: Solution, problem, enabled: bool) -> Solution:
    if not enabled:
        return solution
    return swap_local_search(solution, problem)
