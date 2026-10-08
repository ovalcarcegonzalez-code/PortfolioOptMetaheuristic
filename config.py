from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    data_root: str = "data/stock"
    frequency: str = "WEEK"
    price_col: str = "Close"

    budget: float = 1_000_000
    min_cardinality: int = 10
    max_cardinality: int = 14
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_rate: float = 0.0

    slsqp_min_weight: float = 0.035
    slsqp_max_weight: float = 1.0

    mode: str = "multi_period"

    single_period_t: int = 10

    use_time_limit: bool = False
    time_limit: float = 50.0
    population_size: int = 1_000
    grasp_iterations: int = 30

    rebalancing: str = "monthly"
    lookback_window: int | None = None
    max_periods: int = 127
    warmup_periods: int | None = None

    backtest_grasp_iterations: int = 250
    backtest_random_solutions: int = 10_000

    use_time_budget: bool = False
    time_budget_seconds: float = 3600.0

    run_greedy:       bool = False
    run_grasp:        bool = False
    run_random:       bool = False
    run_oneoverN:     bool = False
    run_evofolio:     bool = True
    run_grasp_pareto: bool = True

    run_full_oneoverN: bool = False
    run_buy_and_hold:  bool = False

    pareto_grasp_alpha:                float = 0.26
    backtest_pareto_grasp_iterations:  int   = 250

    evofolio_population_size: int = 250
    evofolio_generations:     int = 250
    evofolio_crossover_rate:  float = 0.8
    evofolio_indpb:           float = 0.02
    evofolio_tournparam:      float = 0.9
    evofolio_selparam:        float = 0.8
    evofolio_elite_cxpb:      float = 0.3

    return_signal: str = "oracle"

    ewma_half_life_weeks: int = 12

    bl_delta: float = 2.5
    bl_tau:   float = 0.05

    grasp_alpha: float = 0.4
    seed: int = 41

    use_local_search: bool = False

    weight_method: str = "sharpe_max"

    risk_method: str = "markowitz"

    target_period_index: int | None = 14

    results_dir: str = "results"
    plots_dir:   str = "results/plots"
    plot_format: str = "png"
    show_plots:  bool = True
    solutions_output_path: str = "results/solutions.csv"
