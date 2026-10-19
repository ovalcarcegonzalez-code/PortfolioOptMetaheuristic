# Cardinality-Constrained Portfolio Optimisation

Research code. It compares metaheuristics for **cardinality-constrained mean-variance portfolio selection** (10–14 assets out of 47 stocks), evaluated with a rolling multi-period backtest on weekly data.

## Algorithms

| Algorithm | File | Idea |
|---|---|---|
| EvoFolio | `src/evofolio.py` | Genetic algorithm replicating the EvoFolio paper (integer shares, crossover, mutation, tournament selection) |
| Pareto-GRASP | `src/grasp_pareto.py` | GRASP that builds an approximate risk/return Pareto front, picks the max-Sharpe portfolio |
| Greedy / GRASP | `src/problem.py`, `src/algorithm.py` | Forward-selection greedy and randomized GRASP, optional 1-swap local search |
| Random, 1/N | `src/algorithm.py`, `src/baselines.py` | Random feasible portfolios, equal weight, buy & hold baselines |

## How it works

```
data/stock/WEEK/*.csv ─► Instance (price matrix) ─► backtest loop:
    for each rebalancing date t:
        estimate μ, Σ on past window (return_signal)
        PortfolioProblem ─► each algorithm ─► best portfolio
        hold until t+period ─► realised return
─► metrics (return, vol, Sharpe, drawdown) + plots
```

Expected returns (`return_signal` in `config.py`) can be: `oracle`, `full_history`, `ewma`, `bl_views_ewma` (Black-Litterman with EWMA views), among others.

## Layout

```
config.py        all parameters (Config dataclass)
main.py          single run: backtest + plots
src/             core: instance, problem, algorithms, backtest, metrics, plots
scripts/         experiments
  run_multiseed.py               parallel multi-seed backtests (presets: full_history | ewma | bl_quarterly)
  run_experiments.py             variant × rebalancing grid
  plot_multiseed.py, plot_conclusiones.py   result figures
  describe_dataset.py            dataset EDA
  expected_return_experiment/    study of return estimators (EWMA, momentum, BL, Fama-French)
data/stock/      DAY and WEEK CSVs, one per asset
```

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py                               # single run with config.py defaults
python scripts/run_multiseed.py ewma         # 5 seeds in parallel
python -m scripts.expected_return_experiment.run
```

Results go to `results/` (not tracked).
