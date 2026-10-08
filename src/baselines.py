import numpy as np

from src.backtest import (
    BacktestResult, PeriodResult, period_length, _rebalancing_indices,
)
from src.metrics import (
    portfolio_return, portfolio_risk, portfolio_risk_evofolio, sharpe_ratio,
)


def _ex_ante_metrics(weights: np.ndarray, instance, cfg, t: int):
    lookback = cfg.lookback_window
    hist_start = 0 if lookback is None else max(0, t + 1 - lookback)
    hist = instance.returns.iloc[hist_start : t + 1]
    if len(hist) < 2:
        return 0.0, 0.0, 0.0
    mean_returns = hist.mean().values
    cov = hist.cov().values * instance.periods_per_year
    risk_fn = (portfolio_risk_evofolio
               if cfg.risk_method == "evofolio" else portfolio_risk)
    mu = portfolio_return(weights, mean_returns)
    sigma = risk_fn(weights, cov)
    sr = sharpe_ratio(mu, sigma, cfg.risk_free_rate)
    return mu, sigma, sr


def _allocate_shares(weights: np.ndarray, prices: np.ndarray, cash: float):
    shares    = np.floor(weights * cash / prices)
    spent     = float(np.sum(shares * prices))
    cash_left = cash - spent
    return shares, cash_left


def run_full_one_over_n(instance, cfg, period: int | None = None) -> BacktestResult:
    """Equal-weight portfolio over all assets, rebalanced every period.

    In: instance, cfg, period.
    Out: BacktestResult.
    """
    if period is None:
        period = period_length(cfg, instance)
    rebalancing_ts = _rebalancing_indices(cfg, instance, period)
    if not rebalancing_ts:
        raise ValueError("No rebalancing dates for the configured horizon.")

    n_assets = len(instance.assets)
    weights = np.full(n_assets, 1.0 / n_assets)
    close = instance.prices.values

    cash = float(cfg.budget)
    result = BacktestResult(name="Full 1/N")

    for t in rebalancing_ts:
        price_idx      = t + 1
        next_price_idx = min(price_idx + period, len(instance.prices) - 1)
        date           = instance.prices.index[price_idx]

        prices_now  = close[price_idx]
        prices_next = close[next_price_idx]

        cash_before = cash
        shares, cash_left = _allocate_shares(weights, prices_now, cash_before)
        value_next  = float(np.sum(shares * prices_next) + cash_left)
        realized    = np.log(value_next / cash_before) if cash_before > 0 else 0.0
        cash        = value_next

        mu, sigma, sr = _ex_ante_metrics(weights, instance, cfg, t)

        result.periods.append(PeriodResult(
            date=date, t=t,
            weights=weights.copy(), shares=shares, cash_left=cash_left,
            expected_sharpe=sr,
            realized_log_return=float(realized),
            portfolio_value=value_next,
            avg_risk=sigma, avg_return=mu, n_candidates=1,
            trace=np.zeros((0, 2)),
        ))

    return result


def run_buy_and_hold_one_over_n(instance, cfg, period: int | None = None) -> BacktestResult:
    """Buy 1/N of every asset at the first date and hold.

    In: instance, cfg, period.
    Out: BacktestResult.
    """
    if period is None:
        period = period_length(cfg, instance)
    rebalancing_ts = _rebalancing_indices(cfg, instance, period)
    if not rebalancing_ts:
        raise ValueError("No rebalancing dates for the configured horizon.")

    n_assets = len(instance.assets)
    init_weights = np.full(n_assets, 1.0 / n_assets)
    close = instance.prices.values

    t0 = rebalancing_ts[0]
    price0_idx = t0 + 1
    prices0    = close[price0_idx]
    shares, cash_left = _allocate_shares(init_weights, prices0, float(cfg.budget))

    result = BacktestResult(name="Buy&Hold 1/N")

    for t in rebalancing_ts:
        price_idx      = t + 1
        next_price_idx = min(price_idx + period, len(instance.prices) - 1)
        date           = instance.prices.index[price_idx]

        prices_now  = close[price_idx]
        prices_next = close[next_price_idx]

        value_now  = float(np.sum(shares * prices_now) + cash_left)
        value_next = float(np.sum(shares * prices_next) + cash_left)
        realized   = np.log(value_next / value_now) if value_now > 0 else 0.0

        holdings = shares * prices_now
        realised_weights = (holdings / value_now) if value_now > 0 else init_weights

        mu, sigma, sr = _ex_ante_metrics(realised_weights, instance, cfg, t)

        result.periods.append(PeriodResult(
            date=date, t=t,
            weights=realised_weights, shares=shares.copy(), cash_left=cash_left,
            expected_sharpe=sr,
            realized_log_return=float(realized),
            portfolio_value=value_next,
            avg_risk=sigma, avg_return=mu, n_candidates=1,
            trace=np.zeros((0, 2)),
        ))

    return result
