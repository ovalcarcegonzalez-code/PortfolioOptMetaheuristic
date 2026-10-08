import time
import numpy as np

from src.solution import Solution


EPS = 1e-10


def paper_risk_from_shares(P: np.ndarray, cov: np.ndarray) -> float:
    """EvoFolio paper risk metric (replicates original `calcrisk`).

    In: P (np.ndarray share vector), cov (np.ndarray).
    Out: float risk.
    """
    active = P > 0
    K = int(np.count_nonzero(active))
    if K < 2:
        return 0.0
    total = float(P.sum())
    if total <= 0.0:
        return 0.0
    w = P.astype(float) / total
    diag = np.diag(cov)
    diag_term = (K - 1) * float(np.sum(w * diag))
    quadratic = float(w @ cov @ w)
    off_diag = quadratic - float(np.sum(w * w * diag))
    return max(diag_term + off_diag, 0.0)


class EvoFolioAlgorithm:
    """Genetic algorithm replicating EvoFolio: integer share encoding, crossover, mutation and Sharpe-based selection."""
    def __init__(
        self,
        problem,
        population_size: int = 250,
        n_generations: int | None = 250,
        crossover_rate: float = 0.8,
        indpb: float = 0.02,
        tournparam: float = 0.9,
        selparam: float = 0.8,
        elite_cxpb: float = 0.3,
        time_limit: float | None = None,
        seed: int | None = None,
    ):
        if n_generations is None and time_limit is None:
            raise ValueError("Specify n_generations or time_limit.")
        if not (0.0 < selparam <= 1.0):
            raise ValueError("selparam must be in (0, 1].")
        if not (0.0 <= elite_cxpb <= 1.0):
            raise ValueError("elite_cxpb must be in [0, 1].")
        if not (0.0 <= indpb <= 1.0):
            raise ValueError("indpb must be in [0, 1].")
        if not (0.0 <= tournparam <= 1.0):
            raise ValueError("tournparam must be in [0, 1].")

        self.problem = problem
        self.pop_size = population_size
        self.n_gen = n_generations
        self.pc = crossover_rate
        self.indpb = indpb
        self.tournparam = tournparam
        self.selparam = selparam
        self.elite_cxpb = elite_cxpb
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)

        self.mu = problem.mean_returns
        self.cov = problem.cov_matrix
        self.rf = problem.risk_free_rate
        self.n = problem.n_assets
        self.k_min = problem.min_cardinality
        self.k_max = problem.max_cardinality
        self.budget = float(problem.budget)

        price_idx = problem.t + 1
        self.current_prices = np.asarray(
            problem.instance.prices.values[price_idx], dtype=float,
        )
        self.max_shares_per_asset = np.floor(
            self.budget / np.maximum(self.current_prices, EPS)
        ).astype(np.int64)

        self.gen_history: list[tuple[float, float]] = []

    def _random_individual(self) -> np.ndarray:
        for _ in range(20):
            ind = np.zeros(self.n, dtype=np.int64)
            K = int(self.rng.integers(self.k_min, self.k_max + 1))
            chosen = self.rng.choice(self.n, size=K, replace=False)
            remaining = self.budget
            for stock in chosen:
                price = self.current_prices[stock]
                if price <= 0 or price > remaining:
                    continue
                upper = int(remaining // price)
                if upper >= 2:
                    shares = int(self.rng.integers(1, upper + 1))
                    ind[stock] = shares
                    remaining = self.budget - float(np.sum(ind * self.current_prices))
            if (self.k_min <= int(np.count_nonzero(ind)) <= self.k_max
                    and float(np.sum(ind * self.current_prices)) <= self.budget):
                return ind
        cheap = np.argsort(self.current_prices)[: self.k_min]
        ind = np.zeros(self.n, dtype=np.int64)
        ind[cheap] = 1
        return ind

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.n <= 1:
            return p1.copy()
        point = int(self.rng.integers(1, self.n))
        child = np.concatenate([p1[:point], p2[point:]]).astype(np.int64)
        return self._repair(child)

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        new_ind = ind.copy()
        for i in range(self.n):
            if self.rng.random() >= self.indpb:
                continue
            cost_others = float(
                np.sum(new_ind * self.current_prices)
                - new_ind[i] * self.current_prices[i]
            )
            slack = max(self.budget - cost_others, 0.0)
            upper = int(slack // max(self.current_prices[i], EPS))
            if upper <= 0:
                new_ind[i] = 0
                continue
            new_ind[i] = int(self.rng.integers(0, upper + 1))
        return self._repair(new_ind)

    def _repair(self, ind: np.ndarray) -> np.ndarray:
        ind = ind.astype(np.int64, copy=False)
        ind[ind < 0] = 0

        cost = float(np.sum(ind * self.current_prices))
        guard = 0
        while cost > self.budget and np.any(ind > 0) and guard < 5 * self.n:
            active = np.where(ind > 0)[0]
            costs = ind[active] * self.current_prices[active]
            worst = active[int(np.argmax(costs))]
            drop = max(1, int(np.ceil(
                (cost - self.budget) / max(self.current_prices[worst], EPS))))
            ind[worst] = max(0, int(ind[worst]) - drop)
            cost = float(np.sum(ind * self.current_prices))
            guard += 1

        K = int(np.count_nonzero(ind))
        if K > self.k_max:
            active = np.where(ind > 0)[0]
            costs = ind[active] * self.current_prices[active]
            drop_order = active[np.argsort(costs)]
            n_drop = K - self.k_max
            for idx in drop_order[:n_drop]:
                ind[idx] = 0
            K = int(np.count_nonzero(ind))

        if K < self.k_min:
            inactive = np.where(ind == 0)[0]
            order = inactive[np.argsort(self.current_prices[inactive])]
            for idx in order:
                if self.current_prices[idx] <= self.budget - float(
                        np.sum(ind * self.current_prices)) + EPS:
                    ind[idx] = 1
                    K += 1
                    if K >= self.k_min:
                        break
        return ind

    def _objectives(self, ind: np.ndarray) -> tuple[float, float]:
        yld = float(np.sum(ind * self.mu))
        risk = paper_risk_from_shares(ind, self.cov)
        return -yld, risk

    @staticmethod
    def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _non_dominated_sort(self, objs: list[tuple[float, float]]):
        n = len(objs)
        dominated_by = [[] for _ in range(n)]
        dom_count = [0] * n
        rank = [0] * n
        fronts: list[list[int]] = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(objs[p], objs[q]):
                    dominated_by[p].append(q)
                elif self._dominates(objs[q], objs[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt: list[int] = []
            for p in fronts[i]:
                for q in dominated_by[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        rank[q] = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1], rank

    @staticmethod
    def _crowding_distance(front: list[int],
                           objs: list[tuple[float, float]]) -> dict[int, float]:
        n = len(front)
        dist = {i: 0.0 for i in front}
        if n <= 2:
            for i in front:
                dist[i] = float("inf")
            return dist
        for m in (0, 1):
            srt = sorted(front, key=lambda i: objs[i][m])
            dist[srt[0]] = float("inf")
            dist[srt[-1]] = float("inf")
            f_min = objs[srt[0]][m]
            f_max = objs[srt[-1]][m]
            spread = f_max - f_min
            if spread < EPS:
                continue
            for k in range(1, n - 1):
                prev_v = objs[srt[k - 1]][m]
                next_v = objs[srt[k + 1]][m]
                dist[srt[k]] += (next_v - prev_v) / spread
        return dist

    def _tournament(self, rank: list[int], crowding: dict[int, float]) -> int:
        i, j = self.rng.integers(0, len(rank), size=2)
        i, j = int(i), int(j)
        if rank[i] < rank[j]:
            better, worse = i, j
        elif rank[j] < rank[i]:
            better, worse = j, i
        elif crowding.get(i, 0.0) >= crowding.get(j, 0.0):
            better, worse = i, j
        else:
            better, worse = j, i
        return better if self.rng.random() < self.tournparam else worse

    def _shares_to_monetary_weights(self, P: np.ndarray) -> np.ndarray:
        costs = P * self.current_prices
        total = float(costs.sum())
        if total <= 0.0:
            return np.zeros(self.n)
        return costs / total

    def _make_solution(self, P: np.ndarray) -> Solution:
        w = self._shares_to_monetary_weights(P)
        s = Solution(weights=w)
        self.problem.evaluate(s)
        s.shares = P.astype(int).copy()
        return s

    def _record_generation_avg(self, population: list[np.ndarray]) -> None:
        if not population:
            return
        risks: list[float] = []
        rets: list[float] = []
        for P in population:
            w = self._shares_to_monetary_weights(P)
            s = Solution(weights=w)
            self.problem.evaluate(s)
            if np.isfinite(s.risk):
                risks.append(float(s.risk))
            if np.isfinite(s.expected_return):
                rets.append(float(s.expected_return))
        avg_risk = float(np.mean(risks)) if risks else 0.0
        avg_ret = float(np.mean(rets)) if rets else 0.0
        self.gen_history.append((avg_risk, avg_ret))

    def run(self):
        """Evolve the population for the configured generations.

        Out: (best Solution, final population list[Solution]).
        """
        self.gen_history = []

        population = [self._random_individual() for _ in range(self.pop_size)]
        objs = [self._objectives(p) for p in population]
        self._record_generation_avg(population)

        deadline = time.time() + self.time_limit if self.time_limit else None
        n_survivors = max(1, int(round(self.pop_size * self.selparam)))
        n_refill = self.pop_size - n_survivors

        gen = 0
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            if self.n_gen is not None and gen >= self.n_gen:
                break
            gen += 1

            fronts, rank = self._non_dominated_sort(objs)
            crowding: dict[int, float] = {}
            for front in fronts:
                crowding.update(self._crowding_distance(front, objs))
            elite_pool = fronts[0] if fronts and fronts[0] else None

            offspring: list[np.ndarray] = []
            while len(offspring) < self.pop_size:
                p1 = population[self._tournament(rank, crowding)]
                p2 = population[self._tournament(rank, crowding)]
                if elite_pool and self.rng.random() < self.elite_cxpb:
                    elite_ind = population[int(self.rng.choice(elite_pool))]
                    if self.rng.random() < 0.5:
                        p1 = elite_ind
                    else:
                        p2 = elite_ind
                if self.rng.random() < self.pc:
                    c = self._crossover(p1, p2)
                else:
                    c = p1.copy()
                c = self._mutate(c)
                offspring.append(c)

            combined_pop = population + offspring
            combined_objs = objs + [self._objectives(p) for p in offspring]
            fronts_c, _ = self._non_dominated_sort(combined_objs)

            new_pop: list[np.ndarray] = []
            new_objs: list[tuple[float, float]] = []
            for front in fronts_c:
                if len(new_pop) + len(front) <= n_survivors:
                    for i in front:
                        new_pop.append(combined_pop[i])
                        new_objs.append(combined_objs[i])
                    if len(new_pop) == n_survivors:
                        break
                else:
                    cd = self._crowding_distance(front, combined_objs)
                    sorted_by_cd = sorted(front, key=lambda i: -cd[i])
                    remaining = n_survivors - len(new_pop)
                    for i in sorted_by_cd[:remaining]:
                        new_pop.append(combined_pop[i])
                        new_objs.append(combined_objs[i])
                    break

            for _ in range(n_refill):
                p = self._random_individual()
                new_pop.append(p)
                new_objs.append(self._objectives(p))

            population, objs = new_pop, new_objs
            self._record_generation_avg(population)

        all_solutions = [self._make_solution(p) for p in population]
        fronts_final, _ = self._non_dominated_sort(objs)
        pareto_idx = (fronts_final[0] if fronts_final
                       else list(range(len(population))))
        pareto_solutions = [all_solutions[i] for i in pareto_idx]
        best = max(pareto_solutions, key=lambda s: s.sharpe)

        return best, all_solutions
