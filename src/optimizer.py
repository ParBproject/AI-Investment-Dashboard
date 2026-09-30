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

# Daily series are annualised with 252 trading days. The return-target
# constraint below must use this same factor as ``compute_portfolio_metrics``.
TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = TRADING_DAYS,
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
    n_starts: int = 50,
    initial_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Find portfolio weights that maximise the Sharpe ratio.

    ``random_state`` makes the multi-start search reproducible for tests,
    notebooks, and demonstrations while preserving stochastic defaults.
    ``initial_weights`` is tried first when provided (a warm start). It does
    not replace the random restarts. A zero-volatility result has an undefined
    Sharpe ratio and is reported as NaN rather than 0.
    """
    _validate_returns(returns)
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    if initial_weights is not None:
        start = np.asarray(initial_weights, dtype=float)
        if start.shape != (n,) or not np.isfinite(start).all():
            raise ValueError("initial_weights must be a finite vector, one weight per asset")
    else:
        start = None

    # A zero-volatility sample makes the Sharpe ratio undefined. Skip the
    # solver so SLSQP does not divide by a zero scale inside its gradient.
    if float(np.max(np.abs(cov_matrix))) <= 1e-18:
        weights = np.ones(n) / n
        ann_ret, ann_vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
        return weights, ann_ret, ann_vol, float("nan")

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    rng = np.random.default_rng(random_state)

    starts: list[np.ndarray] = []
    if start is not None:
        starts.append(start)
    for _ in range(n_starts):
        if allow_short:
            starts.append(_sample_bounded_short_weights(n, rng))
        else:
            starts.append(rng.dirichlet(np.ones(n)))

    best_result = None
    best_sharpe = -np.inf

    for w0 in starts:
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
    if ann_vol > 1e-12:
        sharpe = (ann_ret - risk_free_rate) / ann_vol
    else:
        sharpe = float("nan")

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
            (ret - risk_free_rate) / vol if vol > 1e-12 else float("nan")
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


def _max_return_weights(
    returns: pd.DataFrame,
    allow_short: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Highest-return portfolio on the same bounds as the other optimizers."""
    _validate_returns(returns)

    n = returns.shape[1]
    mean_returns = returns.mean().to_numpy()
    cov_matrix = returns.cov().to_numpy()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    w0 = np.ones(n) / n

    result = minimize(
        lambda weights, means: -float(np.dot(weights, means)),
        w0,
        args=(mean_returns,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if result.success:
        weights = result.x
    elif allow_short:
        weights = w0
    else:
        weights = np.zeros(n)
        weights[int(np.argmax(mean_returns))] = 1.0

    ann_ret, ann_vol = compute_portfolio_metrics(weights, mean_returns, cov_matrix)
    return weights, ann_ret, ann_vol


def minimum_variance_frontier(
    returns: pd.DataFrame,
    n_points: int = 25,
    allow_short: bool = False,
) -> dict:
    """
    Trace the mean-variance frontier from the global minimum-variance portfolio
    up to the highest feasible return.

    ``efficient_frontier`` still draws random feasible portfolios. This curve
    is the set of minimum-variance portfolios for a grid of target returns,
    which is the efficient frontier on these bounds.
    """
    _validate_returns(returns)
    if n_points < 1:
        raise ValueError("n_points must be at least 1")

    mean_returns = returns.mean().to_numpy()
    cov_matrix = returns.cov().to_numpy()
    n = returns.shape[1]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n

    w_min, ret_min, _ = min_variance_weights(returns, allow_short=allow_short)
    _, ret_max, _ = _max_return_weights(returns, allow_short=allow_short)

    if n_points == 1 or abs(ret_max - ret_min) < 1e-10:
        weights = w_min.reshape(1, -1)
        _, vol_min = compute_portfolio_metrics(w_min, mean_returns, cov_matrix)
        return {
            "rets": np.array([ret_min]),
            "vols": np.array([vol_min]),
            "weights": weights,
        }

    targets = np.linspace(ret_min, ret_max, n_points)
    curve_weights: list[np.ndarray] = []
    curve_rets: list[float] = []
    curve_vols: list[float] = []

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, level=float(target): (
                    float(np.dot(w, mean_returns)) * TRADING_DAYS - level
                ),
            },
        ]
        result = minimize(
            _portfolio_vol,
            w_min,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success or not np.isfinite(result.x).all():
            continue
        achieved_ret, achieved_vol = compute_portfolio_metrics(
            result.x,
            mean_returns,
            cov_matrix,
        )
        curve_weights.append(np.asarray(result.x, dtype=float))
        curve_rets.append(achieved_ret)
        curve_vols.append(achieved_vol)

    if not curve_rets:
        raise ValueError("Could not trace a minimum-variance frontier for these returns.")

    order = np.argsort(curve_rets)
    return {
        "rets": np.asarray(curve_rets, dtype=float)[order],
        "vols": np.asarray(curve_vols, dtype=float)[order],
        "weights": np.vstack(curve_weights)[order],
    }
