from __future__ import annotations

import numpy as np


def historical_mean(returns_history: np.ndarray, period: int) -> np.ndarray:
    weekly_mean = returns_history.mean(axis=0)
    return weekly_mean * period


def ewma(
    returns_history: np.ndarray,
    period: int,
    half_life_weeks: int = 12,
) -> np.ndarray:
    T = len(returns_history)
    lam = 0.5 ** (1.0 / half_life_weeks)
    weights = lam ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()
    weekly_mean = (returns_history * weights[:, None]).sum(axis=0)
    return weekly_mean * period


def momentum_12_1(
    returns_history: np.ndarray,
    period: int,
    lookback_weeks: int = 52,
    skip_weeks: int = 4,
) -> np.ndarray:
    T = len(returns_history)
    end = T - skip_weeks
    start = max(0, end - lookback_weeks)
    if end <= start:
        return historical_mean(returns_history, period)
    window = returns_history[start:end]
    weekly_mean = window.mean(axis=0)
    return weekly_mean * period


def shrinkage_toward_zero(
    returns_history: np.ndarray,
    period: int,
    alpha: float = 0.5,
) -> np.ndarray:
    hist = historical_mean(returns_history, period)
    return (1.0 - alpha) * hist


def black_litterman_equilibrium(
    returns_history: np.ndarray,
    period: int,
    delta: float = 2.5,
    market_weights: np.ndarray | None = None,
) -> np.ndarray:
    T, N = returns_history.shape
    if market_weights is None:
        market_weights = np.ones(N) / N
    Sigma = np.cov(returns_history, rowvar=False)
    Pi_weekly = delta * (Sigma @ market_weights)
    return Pi_weekly * period


def black_litterman_with_views(
    returns_history: np.ndarray,
    period: int,
    delta: float = 2.5,
    tau: float = 0.05,
    half_life_weeks: int = 12,
    market_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Black-Litterman posterior with EWMA views.

    In: returns_history (np.ndarray T x N), period (int), delta, tau, half_life_weeks, market_weights.
    Out: np.ndarray (N,) expected returns.
    """
    T, N = returns_history.shape
    if market_weights is None:
        market_weights = np.ones(N) / N

    Sigma = np.cov(returns_history, rowvar=False)
    Pi = delta * (Sigma @ market_weights)

    lam = 0.5 ** (1.0 / half_life_weeks)
    weights = lam ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()
    Q = (returns_history * weights[:, None]).sum(axis=0)

    Omega = tau * np.diag(np.diag(Sigma))
    tau_Sigma = tau * Sigma

    mid = np.linalg.solve(tau_Sigma + Omega, Q - Pi)
    mu_BL_weekly = Pi + tau_Sigma @ mid
    return mu_BL_weekly * period


def black_litterman_with_momentum_views(
    returns_history: np.ndarray,
    period: int,
    delta: float = 2.5,
    tau: float = 0.05,
    lookback_weeks: int = 52,
    skip_weeks: int = 4,
    market_weights: np.ndarray | None = None,
) -> np.ndarray:
    T, N = returns_history.shape
    if market_weights is None:
        market_weights = np.ones(N) / N

    Sigma = np.cov(returns_history, rowvar=False)
    Pi = delta * (Sigma @ market_weights)

    end = T - skip_weeks
    start = max(0, end - lookback_weeks)
    if end <= start:
        Q = returns_history.mean(axis=0)
    else:
        Q = returns_history[start:end].mean(axis=0)

    Omega = tau * np.diag(np.diag(Sigma))
    tau_Sigma = tau * Sigma
    mid = np.linalg.solve(tau_Sigma + Omega, Q - Pi)
    mu_BL_weekly = Pi + tau_Sigma @ mid
    return mu_BL_weekly * period


def _self_constructed_factors(
    returns_history: np.ndarray,
    mom_lookback: int = 52,
    mom_skip: int = 4,
    vol_window: int = 26,
    quantile_size: int = 10,
) -> tuple[np.ndarray, int]:
    T, N = returns_history.shape
    lookback_needed = max(mom_lookback, vol_window) + 1
    if T < lookback_needed + 1:
        return np.zeros((0, 3)), T

    valid_start = lookback_needed
    F = T - valid_start
    factors = np.zeros((F, 3))

    for k, s in enumerate(range(valid_start, T)):
        mkt = float(returns_history[s].mean())

        mom_window = returns_history[s - mom_lookback : s - mom_skip]
        mom_scores = mom_window.mean(axis=0)
        order_mom = np.argsort(mom_scores)
        losers  = order_mom[:quantile_size]
        winners = order_mom[-quantile_size:]
        mom_factor = float(returns_history[s, winners].mean()
                           - returns_history[s, losers].mean())

        vol_win = returns_history[s - vol_window : s]
        vol_scores = vol_win.std(axis=0)
        order_vol = np.argsort(vol_scores)
        low_vol  = order_vol[:quantile_size]
        high_vol = order_vol[-quantile_size:]
        vol_factor = float(returns_history[s, low_vol].mean()
                           - returns_history[s, high_vol].mean())

        factors[k] = [mkt, mom_factor, vol_factor]

    return factors, valid_start


def fama_french_3f(
    returns_history: np.ndarray,
    period: int,
    mom_lookback: int = 52,
    mom_skip: int = 4,
    vol_window: int = 26,
    quantile_size: int = 10,
    min_regression_obs: int = 20,
) -> np.ndarray:
    """Fama-French/Carhart-style 3-factor expected returns.

    In: returns_history (np.ndarray T x N), period (int), factor params.
    Out: np.ndarray (N,).
    """
    T, N = returns_history.shape

    factors, valid_start = _self_constructed_factors(
        returns_history,
        mom_lookback=mom_lookback,
        mom_skip=mom_skip,
        vol_window=vol_window,
        quantile_size=quantile_size,
    )

    F = len(factors)
    if F < min_regression_obs:
        return historical_mean(returns_history, period)

    asset_returns_aligned = returns_history[valid_start:T]
    X = np.column_stack([np.ones(F), factors])
    factor_means = factors.mean(axis=0)

    predicted_weekly = np.zeros(N)
    for i in range(N):
        y = asset_returns_aligned[:, i]
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            predicted_weekly[i] = float(returns_history[:, i].mean())
            continue
        alpha = coeffs[0]
        betas = coeffs[1:]
        predicted_weekly[i] = float(alpha + betas @ factor_means)

    return predicted_weekly * period


ESTIMATORS: dict[str, tuple] = {
    "historical_mean":   (historical_mean,             {},                          "Historical mean (baseline)"),
    "ewma_hl12":         (ewma,                        {"half_life_weeks": 12},     "EWMA (half-life=12w)"),
    "momentum_12_1":     (momentum_12_1,               {},                          "Momentum 12-1"),
    "shrinkage_a0.5":    (shrinkage_toward_zero,       {"alpha": 0.5},              "Shrinkage → 0 (α=0.5)"),
    "BL_equilibrium":    (black_litterman_equilibrium, {"delta": 2.5},              "Black-Litterman (equilibrium)"),
    "BL_views_EWMA":     (black_litterman_with_views,  {"delta": 2.5, "tau": 0.05,
                                                       "half_life_weeks": 12},      "Black-Litterman (EWMA views)"),
    "BL_views_MOM":      (black_litterman_with_momentum_views,
                                                       {"delta": 2.5, "tau": 0.05}, "Black-Litterman (Momentum views)"),
    "fama_french_3f":    (fama_french_3f,              {},                          "3-factor model (self-built)"),
}
