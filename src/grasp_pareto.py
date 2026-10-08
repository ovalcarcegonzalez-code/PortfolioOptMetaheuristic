import time
import numpy as np

from src.solution import Solution


EPS = 1e-10


class GRASPParetoAlgorithm:
    """GRASP that builds an approximate Pareto front by optimising f_lambda = lambda*risk - (1-lambda)*return for several lambdas."""
    def __init__(
        self,
        problem,
        n_iterations: int | None = 200,
        time_limit: float | None = None,
        alpha: float = 0.25,
        seed: int | None = None,
    ):
        if n_iterations is None and time_limit is None:
            raise ValueError("Specify n_iterations or time_limit.")
        self.problem = problem
        self.n_iterations = n_iterations
        self.time_limit = time_limit
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.mu = problem.mean_returns
        self.cov = problem.cov_matrix
        self.n_assets = problem.n_assets
        self.k_min = problem.min_cardinality
        self.k_max = problem.max_cardinality
        self.use_evofolio_risk = problem.risk_method == "evofolio"
        self.weight_method = problem.weight_method

    def _make_lambda_evaluator(self, lam: float):
        from scipy.optimize import minimize

        mu_full = self.mu
        cov_full = self.cov
        weight_method = self.weight_method
        use_evofolio_risk = self.use_evofolio_risk
        min_w = self.problem.slsqp_min_weight
        max_w = self.problem.slsqp_max_weight
        cache: dict = {}

        def variance(w: np.ndarray, cov_sub: np.ndarray,
                     diag_sub: np.ndarray) -> float:
            quad = float(w @ cov_sub @ w)
            if not use_evofolio_risk:
                return quad
            off = quad - float(np.sum(w * w * diag_sub))
            return float(np.sum(w * diag_sub)) + off

        def f_lambda(w, mu_sub, cov_sub, diag_sub):
            mu_p = float(w @ mu_sub)
            var_p = variance(w, cov_sub, diag_sub)
            sigma_p = float(np.sqrt(max(var_p, 0.0)))
            return lam * mu_p - (1.0 - lam) * sigma_p

        def eval_subset(indices, constrained: bool = False):
            key = (frozenset(indices), constrained)
            if key in cache:
                return cache[key]

            idx = list(indices)
            m = len(idx)
            mu_sub = mu_full[idx]
            cov_sub = cov_full[np.ix_(idx, idx)]
            diag_sub = np.diag(cov_sub)

            if weight_method == "equal":
                w = np.full(m, 1.0 / m)
                val = (w, f_lambda(w, mu_sub, cov_sub, diag_sub))
                cache[key] = val
                return val

            def neg_f(w):
                return -f_lambda(w, mu_sub, cov_sub, diag_sub)

            bounds = ([(min_w, max_w)] * m if constrained
                      else [(0.0, 1.0)] * m)
            result = minimize(
                neg_f,
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

    def _grasp_iteration(self, lam: float) -> Solution:
        eval_subset = self._make_lambda_evaluator(lam)
        rng = self.rng
        n = self.n_assets

        k = int(rng.integers(self.k_min, self.k_max + 1))

        selected: list[int] = []
        candidates = list(range(n))
        for _ in range(k):
            f_vals = {stock: eval_subset(selected + [stock])[1]
                      for stock in candidates}
            finite = {s: f for s, f in f_vals.items() if np.isfinite(f)}
            if not finite:
                chosen = int(rng.choice(candidates))
            else:
                f_best = max(finite.values())
                f_worst = min(finite.values())
                threshold = f_best - self.alpha * (f_best - f_worst)
                rcl = [s for s, f in finite.items() if f >= threshold]
                if not rcl:
                    rcl = list(finite.keys())
                chosen = int(rng.choice(rcl))
            selected.append(chosen)
            candidates.remove(chosen)

        _, current_f = eval_subset(selected)
        candidates = [i for i in range(n) if i not in selected]
        improved = True
        while improved:
            improved = False
            for i_idx, out_stock in enumerate(selected):
                for in_stock in candidates:
                    new_sel = selected[:i_idx] + selected[i_idx + 1:] + [in_stock]
                    _, new_f = eval_subset(new_sel)
                    if new_f > current_f + 1e-8:
                        candidates.remove(in_stock)
                        candidates.append(out_stock)
                        selected = new_sel
                        current_f = new_f
                        improved = True
                        break
                if improved:
                    break

        final_w, _ = eval_subset(selected, constrained=True)
        weights = np.zeros(n)
        weights[selected] = final_w

        sol = Solution(weights=weights)
        self.problem.evaluate(sol)
        return sol

    @staticmethod
    def _pareto_front(solutions: list[Solution]) -> list[Solution]:
        n = len(solutions)
        if n == 0:
            return []
        pts = np.array([[s.expected_return, s.risk] for s in solutions],
                       dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            mu_i, sig_i = pts[i]
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                mu_j, sig_j = pts[j]
                if (mu_j >= mu_i and sig_j <= sig_i and
                        (mu_j > mu_i or sig_j < sig_i)):
                    dominated[i] = True
                    break
        return [s for i, s in enumerate(solutions) if not dominated[i]]

    def run(self):
        """Out: (best Solution by Sharpe, all Pareto Solutions)."""
        deadline = time.time() + self.time_limit if self.time_limit else None
        all_solutions: list[Solution] = []
        iteration = 0

        while True:
            if deadline is not None and time.time() >= deadline:
                break
            if self.n_iterations is not None and iteration >= self.n_iterations:
                break
            iteration += 1

            lam = float(self.rng.random())
            sol = self._grasp_iteration(lam)
            all_solutions.append(sol)

        if not all_solutions:
            raise RuntimeError("GRASPParetoAlgorithm produced no solutions.")

        pareto = self._pareto_front(all_solutions) or all_solutions
        best = max(pareto, key=lambda s: s.sharpe)
        return best, all_solutions
