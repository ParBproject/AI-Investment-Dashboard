"""
models.py
=========
Financial models:
  - Black-Scholes option pricing + Greeks
  - Monte Carlo path simulation
  - VaR / CVaR
  - GMM-based synthetic scenario generation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float, float, dict]:
    """
    Compute Black-Scholes call/put prices and option Greeks.

    Parameters
    ----------
    S : float    Spot price
    K : float    Strike price
    T : float    Time to maturity (years)
    r : float    Risk-free rate (annualised, e.g. 0.045)
    sigma : float  Volatility (annualised, e.g. 0.25)

    Returns
    -------
    (call_price, put_price, greeks_dict)
    """
    if T <= 0 or sigma <= 0:
        call = max(S - K, 0.0)
        put = max(K - S, 0.0)
        return call, put, {}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Greeks
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100          # per 1% vol change
    theta_call = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365                                               # per calendar day
    theta_put = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    ) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    greeks = {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
        "d1": d1,
        "d2": d2,
    }

    return call_price, put_price, greeks


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    horizon: int = 252,
    initial_value: float = 1.0,
) -> np.ndarray:
    """
    Generate Monte Carlo simulation paths using historical return statistics.

    Assumes returns are normally distributed (parametric MC).

    Parameters
    ----------
    returns : pd.Series
        Historical daily portfolio returns.
    n_paths : int
        Number of simulation paths.
    horizon : int
        Forecast horizon in trading days.
    initial_value : float
        Starting portfolio value.

    Returns
    -------
    np.ndarray of shape (horizon + 1, n_paths)
        Simulated portfolio value paths (first row = initial_value).
    """
    mu = returns.mean()
    sigma = returns.std()

    # Draw random shocks: shape (horizon, n_paths)
    shocks = np.random.normal(mu, sigma, (horizon, n_paths))

    # Compute cumulative returns
    paths = np.zeros((horizon + 1, n_paths))
    paths[0] = initial_value
    for t in range(1, horizon + 1):
        paths[t] = paths[t - 1] * (1 + shocks[t - 1])

    return paths


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate asset price paths using Geometric Brownian Motion (GBM).

    Parameters
    ----------
    S0 : float     Initial asset price.
    mu : float     Annual drift (e.g. 0.08 for 8%).
    sigma : float  Annual volatility (e.g. 0.20 for 20%).
    T : float      Time horizon in years.
    n_steps : int  Number of time steps.
    n_paths : int  Number of simulation paths.
    seed : int, optional  Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    Z = np.random.standard_normal((n_steps, n_paths))
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
        )

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

def var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Compute historical Value-at-Risk (VaR) and Conditional VaR (CVaR / ES).

    Parameters
    ----------
    returns : pd.Series  Daily portfolio returns.
    confidence_level : float  E.g. 0.95 for 95% VaR.

    Returns
    -------
    (var, cvar) — both expressed as negative numbers indicating loss.
    """
    sorted_returns = np.sort(returns.values)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]                         # already negative
    cvar = sorted_returns[:index].mean() if index > 0 else sorted_returns[0]
    return var, cvar


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a cumulative returns series.

    Parameters
    ----------
    cum_returns : pd.Series  Cumulative portfolio values.

    Returns
    -------
    float  Maximum drawdown as a negative fraction (e.g. -0.35 = -35%).
    """
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()


def calmar_ratio(
    returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calmar ratio = annual return / |max drawdown|."""
    ann_return = returns.mean() * trading_days
    cum = (1 + returns).cumprod()
    mdd = max_drawdown(cum)
    return ann_return / abs(mdd) if mdd != 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# GMM Scenario Generation (AI Component)
# ─────────────────────────────────────────────────────────────────────────────

def gmm_scenario_returns(
    historical_returns: pd.Series,
    n_paths: int = 1000,
    horizon: int = 252,
    shock_pct: float = -0.20,
    rate_shock: float = 0.01,
    risk_free_rate: float = 0.04,
    n_components: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use a Gaussian Mixture Model (GMM) to learn the return distribution
    and generate synthetic price paths — both baseline (normal) and shocked.

    The GMM captures regime-switching behaviour (e.g. calm vs crisis periods)
    that simple Gaussian models miss.

    Parameters
    ----------
    historical_returns : pd.Series
        Daily portfolio returns for model training.
    n_paths : int
        Number of simulation paths.
    horizon : int
        Forecast horizon in trading days.
    shock_pct : float
        Market shock fraction (e.g. -0.20 for -20% market crash). Applied
        as a one-time adjustment on day 1 of shocked paths.
    rate_shock : float
        Rate hike in decimal (e.g. 0.01 for 100bps). Reduces drift.
    risk_free_rate : float
        Annual risk-free rate (used for drift adjustment).
    n_components : int
        Number of GMM mixture components.
    seed : int
        Random seed.

    Returns
    -------
    (shocked_paths, normal_paths) — both np.ndarray of shape (horizon+1, n_paths)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    np.random.seed(seed)
    ret_vals = historical_returns.values.reshape(-1, 1)

    # Fit GMM to historical return distribution
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
        max_iter=500,
    )
    gmm.fit(ret_vals)

    # Sample from GMM for both scenarios
    def _simulate(shock_day_0: float = 0.0, drift_adj: float = 0.0):
        """Simulate n_paths of horizon steps using GMM-drawn returns."""
        paths = np.zeros((horizon + 1, n_paths))
        paths[0] = 1.0
        for t in range(1, horizon + 1):
            sampled, _ = gmm.sample(n_paths)
            daily_returns = sampled.flatten() - drift_adj / 252
            if t == 1 and shock_day_0 != 0.0:
                daily_returns += shock_day_0        # apply one-time shock
            paths[t] = paths[t - 1] * (1 + daily_returns)
        return paths

    normal_paths = _simulate(shock_day_0=0.0, drift_adj=0.0)
    shocked_paths = _simulate(shock_day_0=shock_pct, drift_adj=rate_shock)

    return shocked_paths, normal_paths
