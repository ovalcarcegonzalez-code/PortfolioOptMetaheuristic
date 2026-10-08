import time as _time
import numpy as np


class GreedyAlgorithm:
    def __init__(self, problem):
        self.problem = problem

    def run(self):
        """Out: (best Solution, [best Solution])."""
        sol = self.problem.greedy_solution()
        return sol, [sol]


class GRASPAlgorithm:
    def __init__(
        self,
        problem,
        n_iterations: int | None = None,
        time_limit: float | None = None,
        alpha: float = 0.3,
        seed: int | None = None,
    ):
        if n_iterations is None and time_limit is None:
            raise ValueError("Specify n_iterations or time_limit.")
        self.problem = problem
        self.n_iterations = n_iterations
        self.time_limit = time_limit
        self.alpha = alpha
        self.seed = seed

    def run(self):
        """Out: (best Solution, list of per-iteration Solutions)."""
        best, all_sols = self.problem.grasp_solution(
            n_iterations=self.n_iterations,
            time_limit=self.time_limit,
            alpha=self.alpha,
            seed=self.seed,
            return_all=True,
        )
        return best, all_sols


class OneOverNAlgorithm:

    def __init__(
        self,
        problem,
        population_size: int | None = None,
        time_limit: float | None = None,
        seed: int | None = None,
    ):
        if population_size is None and time_limit is None:
            raise ValueError("Specify population_size or time_limit.")
        self.problem = problem
        self.population_size = population_size
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)

    def run(self):
        """Out: (best Solution, list of sampled 1/N Solutions)."""
        deadline = _time.time() + self.time_limit if self.time_limit is not None else None
        population: list = []
        n = 0

        while True:
            if deadline is not None and _time.time() >= deadline:
                break
            if self.population_size is not None and n >= self.population_size:
                break

            solution = self.problem.equal_weight_solution(self.rng)
            population.append(solution)
            n += 1

        population.sort(key=lambda s: s.sharpe, reverse=True)
        return population[0], population


class RandomFeasibleAlgorithm:
    def __init__(
        self,
        problem,
        population_size: int | None = None,
        time_limit: float | None = None,
        seed: int | None = None,
    ):
        if population_size is None and time_limit is None:
            raise ValueError("Specify population_size or time_limit.")
        self.problem = problem
        self.population_size = population_size
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)

    def run(self):
        """Out: (best Solution, population sorted by Sharpe desc)."""
        deadline = _time.time() + self.time_limit if self.time_limit is not None else None
        population: list = []
        n = 0

        while True:
            if deadline is not None and _time.time() >= deadline:
                break
            if self.population_size is not None and n >= self.population_size:
                break

            solution = self.problem.random_feasible_solution(self.rng)
            if not solution.feasible:
                raise RuntimeError("Generated solution is not feasible.")
            population.append(solution)
            n += 1

        population.sort(key=lambda s: s.sharpe, reverse=True)
        return population[0], population
