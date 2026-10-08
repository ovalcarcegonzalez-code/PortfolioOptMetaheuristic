import numpy as np


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(weights @ mean_returns)


def portfolio_risk(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def portfolio_risk_evofolio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    diag = np.diag(cov_matrix)
    off_diag = float(weights @ cov_matrix @ weights) - float(np.sum(weights ** 2 * diag))
    var_star = float(np.sum(weights * diag)) + off_diag
    return float(np.sqrt(max(var_star, 0.0)))


def sharpe_ratio(expected_return: float, risk: float, risk_free_rate: float = 0.0) -> float:
    if risk <= 0:
        return 0.0
    return float((expected_return - risk_free_rate) / risk)


def compute_metrics(values, periods_per_year: float, rf: float = 0.0) -> dict:
    """Performance metrics from a portfolio value series.

    In: values (array-like), periods_per_year (float), rf (float).
    Out: dict (final_value, total_return, annualised_return, annualised_vol, sharpe, max_drawdown, n_periods).
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("Need at least 2 values to compute returns.")

    log_rets = np.diff(np.log(v))
    ann_ret = float(log_rets.mean()) * periods_per_year
    ann_vol = float(log_rets.std(ddof=1)) * float(np.sqrt(periods_per_year)) \
              if log_rets.size > 1 else 0.0
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-12 else float("nan")

    cummax = np.maximum.accumulate(v)
    drawdown = (v - cummax) / cummax
    max_dd = float(drawdown.min())

    return {
        "final_value":       float(v[-1]),
        "total_return":      float(v[-1] / v[0] - 1.0),
        "annualised_return": ann_ret,
        "annualised_vol":    ann_vol,
        "sharpe":            float(sharpe),
        "max_drawdown":      max_dd,
        "n_periods":         int(log_rets.size),
    }