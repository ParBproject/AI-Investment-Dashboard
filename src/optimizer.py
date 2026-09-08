"""
optimizer.py
============
Portfolio optimization utilities for the investment dashboard.

Implements Markowitz mean-variance optimization, Monte Carlo efficient-frontier
sampling, maximum-Sharpe portfolios, minimum-variance portfolios, and an
equal-risk-contribution (risk parity) allocation.

References
----------
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of
  Equally Weighted Risk Contribution Portfolios.
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
    # Numerical noise can make a theoretically non-negative variance slightly
    # negative for ill-conditioned covariance matrices.
    port_vol = np.sqrt(max(float(port_variance), 0.0) * trading_days)
    return float(port_return), float(port_vol)


def portfolio_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Return each asset's contribution to total portfolio volatility.

    Risk contributions sum to the portfolio's annualised volatility, making the
    result directly useful for diagnosing concentration that is not obvious from
    capital weights alone.
    """
    weights = np.asarray(weights, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    variance = float(weights @ cov_matrix @ weights)
    if variance <= 1e-20:
        return np.zeros_like(weights)

    portfolio_vol = np.sqrt(variance * trading_days)
    marginal_risk = (cov_matrix @ weights) * trading_days / portfolio_vol
    return weights * marginal_risk


def _sample_bounded_short_weights(
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample weights that sum to one while respecting the [-1, 1] short bounds.

    The previous implementation normalised a random vector twice. When the
    vector's net exposure was close to zero, the second normalisation could
    create extremely large weights and unrealistic leverage. This sampler moves
    from equal weight along a zero-sum random direction and caps the step before
    any asset can cross the configured optimizer bounds.
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


def _risk_parity_objective(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Minimise dispersion between asset-level volatility contributions."""
    contributions = portfolio_risk_contributions(
        weights,
        cov_matrix,
        trading_days=1,
    )
    total_risk = contributions.sum()
    if total_risk <= 1e-12:
        return 1e6

    target = total_risk / len(weights)
    return float(np.sum((contributions - target) ** 2))


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

    ``random_state`` makes the multi-start optimisation reproducible for tests,
    demos, and research notebooks without changing the default behaviour.
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

    When short selling is enabled, sampled portfolios now respect the same
    [-1, 1] per-asset bounds used by the optimizer instead of creating accidental
    high-leverage outliers. ``random_state`` enables reproducible simulations.
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


def risk_parity_weights(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Find a long-only equal-risk-contribution (risk parity) portfolio.

    Unlike equal capital weights, risk parity targets the same contribution to
    total portfolio volatility from each asset. The returned risk-contribution
    vector makes the allocation auditable in the dashboard or notebooks.

    Returns
    -------
    (weights, annual_return, annual_volatility, risk_contributions)
    """
    _validate_returns(returns)

    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((0.0, 1.0),) * n

    result = minimize(
        _risk_parity_objective,
        w0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    weights = result.x if result.success else w0
    ann_ret, ann_vol = compute_portfolio_metrics(
        weights,
        mean_returns,
        cov_matrix,
    )
    contributions = portfolio_risk_contributions(weights, cov_matrix)
    return weights, ann_ret, ann_vol, contributions
