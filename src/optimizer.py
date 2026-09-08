"""
optimizer.py
============
Mean-variance portfolio optimization utilities.

Implements Markowitz efficient-frontier simulation, maximum-Sharpe allocation,
and minimum-variance allocation with bounded short-selling support.

References
----------
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float]:
    """Compute annualised portfolio return and volatility."""
    port_return = np.dot(weights, mean_returns) * trading_days
    port_variance = np.dot(weights, np.dot(cov_matrix, weights))
    # Ill-conditioned covariance matrices can produce a tiny negative value
    # from floating-point noise even though variance is non-negative.
    port_vol = np.sqrt(max(float(port_variance), 0.0) * trading_days)
    return float(port_return), float(port_vol)


def _sample_bounded_short_weights(
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample weights that sum to one while respecting per-asset [-1, 1] bounds.

    The previous implementation normalized a random vector by gross exposure
    and then normalized it again by net exposure. When net exposure was close
    to zero, the second division could create extremely large weights and
    unrealistic leverage. This sampler moves from equal weight along a zero-sum
    random direction and caps the step before any asset crosses its bound.
    """
    base = np.ones(n_assets) / n_assets
    if n_assets == 1:
        return base

    direction = rng.normal(size=n_assets)
    direction -= direction.mean()
    if np.linalg.norm(direction) < 1e-12:
        return base

    scale_limits: list[float] = []
    for base_weight, delta in zip(base, direction):
        if delta > 0:
            scale_limits.append((1.0 - base_weight) / delta)
        elif delta < 0:
            scale_limits.append((-1.0 - base_weight) / delta)

    max_scale = min(scale_limits) if scale_limits else 0.0
    return base + rng.uniform(0.0, max_scale) * direction


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


def _validate_returns(returns: pd.DataFrame) -> None:
    """Validate the minimum shape and numeric quality required by optimizers."""
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column")
    if returns.shape[0] < 2:
        raise ValueError("returns must contain at least two observations")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must contain only finite numeric values")


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation functions
# ─────────────────────────────────────────────────────────────────────────────

def max_sharpe_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Find portfolio weights that maximise the Sharpe ratio.

    ``random_state`` makes the multi-start search reproducible for tests,
    notebooks, and demonstrations while preserving stochastic defaults.
    """
    _validate_returns(returns)

    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    rng = np.random.default_rng(random_state)

    best_result = None
    best_sharpe = -np.inf

    for _ in range(50):
        if allow_short:
            w0 = _sample_bounded_short_weights(n, rng)
        else:
            w0 = rng.dirichlet(np.ones(n))

        result = minimize(
            _neg_sharpe,
            w0,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if (
            result.success
            and np.isfinite(result.fun)
            and (-result.fun) > best_sharpe
        ):
            best_sharpe = -result.fun
            best_result = result

    weights = np.ones(n) / n if best_result is None else best_result.x
    ann_ret, ann_vol = compute_portfolio_metrics(
        weights,
        mean_returns,
        cov_matrix,
    )
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    return weights, ann_ret, ann_vol, float(sharpe)


def efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 500,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
    random_state: Optional[int] = None,
) -> dict:
    """
    Generate random portfolio points to approximate the efficient frontier.

    Short-enabled simulations now use the same [-1, 1] per-asset bounds as the
    optimizer instead of producing accidental high-leverage outliers.
    ``random_state`` allows deterministic simulations when desired.
    """
    _validate_returns(returns)
    if n_portfolios < 1:
        raise ValueError("n_portfolios must be at least 1")

    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    rng = np.random.default_rng(random_state)

    results_ret = np.zeros(n_portfolios)
    results_vol = np.zeros(n_portfolios)
    results_sharpe = np.zeros(n_portfolios)
    results_weights = np.zeros((n_portfolios, n))

    for i in range(n_portfolios):
        if allow_short:
            weights = _sample_bounded_short_weights(n, rng)
        else:
            weights = rng.dirichlet(np.ones(n))

        ret, vol = compute_portfolio_metrics(
            weights,
            mean_returns,
            cov_matrix,
        )
        results_ret[i] = ret
        results_vol[i] = vol
        results_sharpe[i] = (
            (ret - risk_free_rate) / vol if vol > 0 else 0.0
        )
        results_weights[i] = weights

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
    """Find the global minimum-variance portfolio weights."""
    _validate_returns(returns)

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
    ann_ret, ann_vol = compute_portfolio_metrics(
        weights,
        mean_returns,
        cov_matrix,
    )
    return weights, ann_ret, ann_vol
