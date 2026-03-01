"""
optimizer.py
============
Mean-variance portfolio optimization utilities.
Implements Markowitz efficient frontier and Max-Sharpe weight finding
using scipy.optimize.

References
----------
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float]:
    """
    Compute annualised portfolio return and volatility.

    Parameters
    ----------
    weights : np.ndarray, shape (n,)
        Portfolio weights (must sum to 1).
    mean_returns : np.ndarray, shape (n,)
        Daily mean returns per asset.
    cov_matrix : np.ndarray, shape (n, n)
        Daily covariance matrix.
    trading_days : int
        Number of trading days to annualise (default 252).

    Returns
    -------
    (portfolio_return, portfolio_volatility) both annualised.
    """
    port_return = np.dot(weights, mean_returns) * trading_days
    port_variance = np.dot(weights, np.dot(cov_matrix, weights))
    port_vol = np.sqrt(port_variance * trading_days)
    return port_return, port_vol


def _neg_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Objective: negative Sharpe ratio (for minimisation)."""
    ret, vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
    if vol < 1e-10:
        return np.inf
    return -(ret - risk_free_rate) / vol


def _portfolio_vol(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Objective: portfolio volatility (for min-variance optimisation)."""
    _, vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation functions
# ─────────────────────────────────────────────────────────────────────────────

def max_sharpe_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
) -> tuple[np.ndarray, float, float, float]:
    """
    Find the portfolio weights that maximise the Sharpe ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns DataFrame, one column per asset.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.045 for 4.5%).
    allow_short : bool
        If True, weights can be negative (short selling). Default False.

    Returns
    -------
    (weights, annual_return, annual_volatility, sharpe_ratio)
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Bounds
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n

    # Multiple random starts to avoid local optima
    best_result = None
    best_sharpe = -np.inf

    for _ in range(50):
        w0 = np.random.dirichlet(np.ones(n))
        result = minimize(
            _neg_sharpe,
            w0,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success and (-result.fun) > best_sharpe:
            best_sharpe = -result.fun
            best_result = result

    if best_result is None:
        # Fall back to equal weights
        weights = np.ones(n) / n
    else:
        weights = best_result.x

    ann_ret, ann_vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    return weights, ann_ret, ann_vol, sharpe


def efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 500,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
) -> dict:
    """
    Generate random portfolio points to approximate the efficient frontier.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.
    n_portfolios : int
        Number of random portfolios to simulate.
    risk_free_rate : float
        Annual risk-free rate.
    allow_short : bool
        Allow negative weights.

    Returns
    -------
    dict with keys: 'rets', 'vols', 'sharpes', 'weights'
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    results_ret = np.zeros(n_portfolios)
    results_vol = np.zeros(n_portfolios)
    results_sharpe = np.zeros(n_portfolios)
    results_weights = np.zeros((n_portfolios, n))

    for i in range(n_portfolios):
        if allow_short:
            w = np.random.randn(n)
            w /= np.sum(np.abs(w))  # normalise to sum of abs = 1
            w /= np.sum(w)          # then to sum = 1
        else:
            w = np.random.dirichlet(np.ones(n))

        ret, vol = compute_portfolio_metrics(w, mean_returns, cov_matrix)
        results_ret[i] = ret
        results_vol[i] = vol
        results_sharpe[i] = (ret - risk_free_rate) / vol if vol > 0 else 0
        results_weights[i] = w

    return {
        "rets": results_ret,
        "vols": results_vol,
        "sharpes": results_sharpe,
        "weights": results_weights,
    }


def min_variance_weights(
    returns: pd.DataFrame,
    allow_short: bool = False,
) -> tuple[np.ndarray, float, float]:
    """
    Find the global minimum-variance portfolio weights.

    Returns
    -------
    (weights, annual_return, annual_volatility)
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    w0 = np.ones(n) / n

    result = minimize(
        _portfolio_vol,
        w0,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = result.x if result.success else w0
    ann_ret, ann_vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
    return weights, ann_ret, ann_vol
