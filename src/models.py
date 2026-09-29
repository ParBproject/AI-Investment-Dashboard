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

def _validate_black_scholes_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> None:
    """Reject inputs that make the closed form undefined."""
    if not np.isfinite([S, K, T, r, sigma]).all():
        raise ValueError("Spot, strike, maturity, rate, and volatility must be finite.")
    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive.")
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative.")


def _greeks(
    delta_call: float,
    gamma: float,
    vega: float,
    theta_call: float,
    theta_put: float,
    rho_call: float,
    rho_put: float,
    d1: float,
    d2: float,
) -> dict:
    """Pack Greeks. Vega and rho are per 1 percentage point; theta is per day."""
    return {
        "delta_call": float(delta_call),
        "delta_put": float(delta_call - 1.0),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_call": float(theta_call),
        "theta_put": float(theta_put),
        "rho_call": float(rho_call),
        "rho_put": float(rho_put),
        "d1": float(d1),
        "d2": float(d2),
    }


def _discounted_intrinsic(
    S: float,
    K: float,
    T: float,
    r: float,
) -> tuple[float, float, dict]:
    """
    Black-Scholes limit for zero volatility or zero time.

    With no volatility the spot grows deterministically at the risk-free rate,
    so the call is the discounted forward intrinsic ``max(S - K e^{-rT}, 0)``,
    not the undiscounted ``max(S - K, 0)``. At expiry (T = 0) that reduces to
    ordinary intrinsic value. Gamma and vega are reported as 0, including at
    the kink, so callers always receive finite Greeks.
    """
    discount = float(np.exp(-r * T))
    present_strike = K * discount
    call = max(S - present_strike, 0.0)
    put = max(present_strike - S, 0.0)

    if np.isclose(S, present_strike):
        delta_call = 0.5
    elif S > present_strike:
        delta_call = 1.0
    else:
        delta_call = 0.0

    # Theta is dV/dt with t calendar time, so the sign flips relative to dV/dT.
    # An in-the-money call V = S - K e^{-rT} has theta/year = -r K e^{-rT}.
    carry = r * present_strike
    if T == 0 or np.isclose(S, present_strike):
        theta_call = 0.0
        theta_put = 0.0
        rho_call = 0.0
        rho_put = 0.0
    elif call > 0.0:
        theta_call = -carry / 365.0
        theta_put = 0.0
        rho_call = (T * present_strike) / 100.0
        rho_put = 0.0
    else:
        theta_call = 0.0
        theta_put = carry / 365.0
        rho_call = 0.0
        rho_put = -(T * present_strike) / 100.0

    greeks = _greeks(
        delta_call, 0.0, 0.0, theta_call, theta_put, rho_call, rho_put, np.nan, np.nan
    )
    return float(call), float(put), greeks


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float, float, dict]:
    """
    Compute Black-Scholes call/put prices and option Greeks.

    Prices are the standard no-dividend formulas. Greeks use the quoting
    convention already baked into the dashboard:

    - delta, gamma: raw partial derivatives
    - vega: per 1 percentage point of volatility (textbook dV/dσ divided by 100)
    - theta: per calendar day (textbook dV/dt divided by 365)
    - rho: per 1 percentage point of the rate (textbook dV/dr divided by 100)

    Parameters
    ----------
    S : float    Spot price
    K : float    Strike price
    T : float    Time to maturity (years)
    r : float    Risk-free rate (annualised, continuous, e.g. 0.045)
    sigma : float  Volatility (annualised, e.g. 0.25)

    Returns
    -------
    (call_price, put_price, greeks_dict)

    Raises
    ------
    ValueError
        If spot or strike is non-positive, maturity or volatility is negative,
        or any input is non-finite.
    """
    _validate_black_scholes_inputs(S, K, T, r, sigma)
    if T == 0 or sigma == 0:
        return _discounted_intrinsic(S, K, T, r)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    call_price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    put_price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100          # per 1 percentage point of vol
    theta_call = (
        -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
        - r * K * discount * norm.cdf(d2)
    ) / 365                                       # per calendar day
    theta_put = (
        -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
        + r * K * discount * norm.cdf(-d2)
    ) / 365
    rho_call = K * T * discount * norm.cdf(d2) / 100
    rho_put = -K * T * discount * norm.cdf(-d2) / 100

    greeks = _greeks(
        norm.cdf(d1),
        gamma,
        vega,
        theta_call,
        theta_put,
        rho_call,
        rho_put,
        d1,
        d2,
    )
    return float(call_price), float(put_price), greeks


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
