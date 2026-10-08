import numpy as np

from src.solution import Solution
from src.metrics import (
    portfolio_return, portfolio_risk, portfolio_risk_evofolio, sharpe_ratio,
)
from src.constraints import is_feasible_weights, sample_bounded_weights


class PortfolioProblem:
    """Cardinality-constrained mean-variance portfolio problem over a fixed estimation window.

    Holds mean returns, covariance and weight/cardinality bounds; builds and evaluates Solutions.
    """
    def __init__(
        self,
        instance,
        t: int,
        period: int = 1,
        budget: float = 1_000_000,
        min_cardinality: int = 10,
        max_cardinality: int = 14,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        slsqp_min_weight: float | None = None,
        slsqp_max_weight: float | None = None,
        risk_free_rate: float = 0.0,
        lookback_window: int | None = None,
        mean_returns: np.ndarray | None = None,
        weight_method: str = "sharpe_max",
        risk_method: str = "markowitz",
    ):
        self.instance = instance
        self.t = t
        self.period = period
        self.budget = budget
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.slsqp_min_weight = (
            min_weight if slsqp_min_weight is None else slsqp_min_weight
        )
        self.slsqp_max_weight = (
            max_weight if slsqp_max_weight is None else slsqp_max_weight
        )
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(instance.assets)
        if weight_method not in {"sharpe_max", "equal"}:
            raise ValueError(
                f"weight_method must be 'sharpe_max' or 'equal', got {weight_method!r}."
            )
        self.weight_method = weight_method
        if risk_method not in {"markowitz", "evofolio"}:
            raise ValueError(
                f"risk_method must be 'markowitz' or 'evofolio', got {risk_method!r}."
            )
        self.risk_method = risk_method

        hist_start = 0 if lookback_window is None else max(0, t + 1 - lookback_window)
        hist = instance.returns.iloc[hist_start : t + 1]
        if len(hist) < 2:
            raise ValueError(f"Not enough history at t={t}: {len(hist)} observations.")

        if mean_returns is not None:
            self.mean_returns = np.asarray(mean_returns, dtype=float)
        else:
            self.mean_returns = hist.mean().values

        cov = hist.cov().values * instance.periods_per_year
        ridge = 1e-6 * np.trace(cov) / max(self.n_assets, 1)
        self.cov_matrix = cov + np.eye(self.n_assets) * max(ridge, 1e-10)

        self._eval_cache: dict = {}

        self._validate_problem_constraints()

    def _validate_problem_constraints(self):
        if self.max_cardinality > self.n_assets:
            raise ValueError("max_cardinality cannot exceed number of assets.")
        if self.min_cardinality > self.max_cardinality:
            raise ValueError("min_cardinality > max_cardinality.")
        if self.max_cardinality * self.min_weight > 1.0:
            raise ValueError("Infeasible: max_cardinality * min_weight > 1.")
        if self.max_cardinality * self.max_weight < 1.0:
            raise ValueError("Infeasible: max_cardinality * max_weight < 1.")
        if self.max_cardinality * self.slsqp_min_weight > 1.0:
            raise ValueError(
                "Infeasible: max_cardinality * slsqp_min_weight > 1."
            )
        if self.min_cardinality * self.slsqp_max_weight < 1.0:
            raise ValueError(
                "Infeasible: min_cardinality * slsqp_max_weight < 1."
            )

    def equal_weight_solution(self, rng: np.random.Generator) -> Solution:
        """Random 1/K portfolio with K in [min_cardinality, max_cardinality].

        In: rng (np.random.Generator).
        Out: Solution.
        """
        k = int(rng.integers(self.min_cardinality, self.max_cardinality + 1))
        selected = rng.choice(self.n_assets, size=k, replace=False)

        weights = np.zeros(self.n_assets)
        weights[selected] = 1.0 / k

        solution = Solution(weights=weights)
        self.evaluate(solution)
        return solution

    def random_feasible_solution(self, rng: np.random.Generator) -> Solution:
        k = int(rng.integers(self.min_cardinality, self.max_cardinality + 1))
        selected_indices = rng.choice(self.n_assets, size=k, replace=False)
        selected_weights = sample_bounded_weights(
            n_selected=k,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            rng=rng,
        )

        weights = np.zeros(self.n_assets)
        weights[selected_indices] = selected_weights

        solution = Solution(weights=weights)
        self.evaluate(solution)
        return solution

    def _make_sharpe_evaluator(self):
        from scipy.optimize import minimize

        cache = self._eval_cache

        def eval_subset(indices, constrained: bool = False):
            key = (frozenset(indices), constrained)
            if key in cache:
                return cache[key]

            indices = list(indices)
            m = len(indices)
            mu = self.mean_returns[indices]
            cov = self.cov_matrix[np.ix_(indices, indices)]
            rf = self.risk_free_rate
            diag = np.diag(cov)
            use_evofolio = self.risk_method == "evofolio"

            def variance(w: np.ndarray) -> float:
                if not use_evofolio:
                    return float(w @ cov @ w)
                off = float(w @ cov @ w) - float(np.sum(w * w * diag))
                return float(np.sum(w * diag)) + off

            if self.weight_method == "equal":
                w = np.full(m, 1.0 / m)
                q = variance(w)
                if not np.isfinite(q) or q <= 1e-20:
                    sharpe = -np.inf
                else:
                    sharpe = (float(w @ mu) - rf) / float(np.sqrt(q))
                val = (w, sharpe)
                cache[key] = val
                return val

            def neg_sharpe(w: np.ndarray) -> float:
                q = variance(w)
                if not np.isfinite(q) or q <= 1e-20:
                    return 0.0
                risk = float(np.sqrt(q))
                return -(float(w @ mu) - rf) / risk

            bounds = (
                [(self.slsqp_min_weight, self.slsqp_max_weight)] * m
                if constrained
                else [(0.0, 1.0)] * m
            )
            result = minimize(
                neg_sharpe,
                np.ones(m) / m,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                options={"ftol": 1e-9, "maxiter": 300},
            )
            val = (result.x, -result.fun if result.success else -np.inf)
            cache[key] = val
            return val

        return eval_subset

    def greedy_solution(self) -> Solution:
        """Forward-selection greedy: adds the asset that most improves Sharpe until max cardinality.

        Out: Solution.
        """
        eval_subset = self._make_sharpe_evaluator()

        selected: list[int] = []
        candidates = list(range(self.n_assets))

        best_sharpe = -np.inf
        best_indices: list[int] = []

        for _ in range(self.max_cardinality):
            step_best_sharpe = -np.inf
            step_best_stock = None
            for stock in candidates:
                _, sharpe = eval_subset(selected + [stock])
                if np.isfinite(sharpe) and sharpe > step_best_sharpe:
                    step_best_sharpe = sharpe
                    step_best_stock = stock

            if step_best_stock is None:
                step_best_stock = candidates[0]
            selected.append(step_best_stock)
            candidates.remove(step_best_stock)

            k = len(selected)
            if self.min_cardinality <= k <= self.max_cardinality and step_best_sharpe > best_sharpe:
                best_sharpe = step_best_sharpe
                best_indices = list(selected)

        if not best_indices:
            best_indices = selected

        final_w, _ = eval_subset(best_indices, constrained=True)
        weights = np.zeros(self.n_assets)
        weights[best_indices] = final_w

        solution = Solution(weights=weights)
        self.evaluate(solution)
        return solution

    def grasp_solution(
        self,
        n_iterations: int | None = 100,
        time_limit: float | None = None,
        alpha: float = 0.3,
        seed: int | None = None,
        return_all: bool = False,
    ) -> Solution | tuple[Solution, list[Solution]]:
        """GRASP: randomized greedy construction (RCL with alpha) + weight optimisation per iteration.

        In: n_iterations, time_limit, alpha, seed, return_all flag.
        Out: best Solution, or (best, all_solutions).
        """
        import time as _time

        if n_iterations is None and time_limit is None:
            raise ValueError("Specify n_iterations or time_limit.")

        rng = np.random.default_rng(seed)
        eval_subset = self._make_sharpe_evaluator()

        deadline = _time.time() + time_limit if time_limit is not None else None
        best_solution: Solution | None = None
        all_solutions: list[Solution] = []
        iteration = 0

        while True:
            if deadline is not None and _time.time() >= deadline:
                break
            if n_iterations is not None and iteration >= n_iterations:
                break
            iteration += 1

            k = int(rng.integers(self.min_cardinality, self.max_cardinality + 1))

            selected: list[int] = []
            candidates = list(range(self.n_assets))

            for _ in range(k):
                sharpes = {
                    stock: eval_subset(selected + [stock])[1]
                    for stock in candidates
                }
                finite = {s: sh for s, sh in sharpes.items() if np.isfinite(sh)}
                if not finite:
                    chosen = int(rng.choice(candidates))
                else:
                    sh_best  = max(finite.values())
                    sh_worst = min(finite.values())
                    threshold = sh_best - alpha * (sh_best - sh_worst)
                    rcl = [s for s, sh in finite.items() if sh >= threshold]
                    if not rcl:
                        rcl = list(finite.keys())
                    chosen = int(rng.choice(rcl))
                selected.append(chosen)
                candidates.remove(chosen)

            _, current_sharpe = eval_subset(selected)
            candidates = [i for i in range(self.n_assets) if i not in selected]

            improved = True
            while improved:
                improved = False
                for i_idx, out_stock in enumerate(selected):
                    for in_stock in candidates:
                        new_sel = selected[:i_idx] + selected[i_idx + 1:] + [in_stock]
                        _, new_sharpe = eval_subset(new_sel)
                        if new_sharpe > current_sharpe + 1e-8:
                            candidates.remove(in_stock)
                            candidates.append(out_stock)
                            selected = new_sel
                            current_sharpe = new_sharpe
                            improved = True
                            break
                    if improved:
                        break

            final_w, _ = eval_subset(selected, constrained=True)
            weights = np.zeros(self.n_assets)
            weights[selected] = final_w

            solution = Solution(weights=weights)
            self.evaluate(solution)

            if return_all:
                all_solutions.append(solution)

            if best_solution is None or solution.sharpe > best_solution.sharpe:
                best_solution = solution

        if return_all:
            return best_solution, all_solutions
        return best_solution

    def is_feasible(self, solution: Solution) -> bool:
        return is_feasible_weights(
            weights=solution.weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            min_cardinality=self.min_cardinality,
            max_cardinality=self.max_cardinality,
        )

    def evaluate(self, solution: Solution) -> None:
        """Compute expected return, risk and Sharpe of a solution in place.

        In: solution (Solution).
        Out: None.
        """
        ret = portfolio_return(solution.weights, self.mean_returns)
        if self.risk_method == "evofolio":
            risk = portfolio_risk_evofolio(solution.weights, self.cov_matrix)
        else:
            risk = portfolio_risk(solution.weights, self.cov_matrix)
        sharpe = sharpe_ratio(ret, risk, self.risk_free_rate)

        solution.expected_return = ret
        solution.risk = risk
        solution.sharpe = sharpe
        solution.feasible = self.is_feasible(solution)
