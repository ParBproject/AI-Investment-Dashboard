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
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Monte Carlo simulation paths using historical return statistics.

    Assumes simple returns are normally distributed (parametric MC). ``returns``
    are per-step, not annualised: a daily series uses the daily mean as drift
    and the daily sample standard deviation (ddof=1) as volatility. Each step
    multiplies wealth by ``1 + shock``. The random draws come from a local
    Generator so a seed is reproducible and does not change NumPy's global RNG.

    Parameters
    ----------
    returns : pd.Series
        Historical per-step portfolio returns (daily when horizon is in days).
    n_paths : int
        Number of simulation paths.
    horizon : int
        Forecast horizon in steps (trading days for a daily series).
    initial_value : float
        Starting portfolio value.
    seed : int, optional
        Seed for the local random generator.

    Returns
    -------
    np.ndarray of shape (horizon + 1, n_paths)
        Simulated portfolio value paths (first row = initial_value).
    """
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        raise ValueError("returns must contain at least two finite observations")

    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, (horizon, n_paths))

    paths = np.empty((horizon + 1, n_paths))
    paths[0] = initial_value
    paths[1:] = initial_value * np.cumprod(1.0 + shocks, axis=0)
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
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if T < 0:
        raise ValueError("T cannot be negative")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")
    if not np.isfinite([S0, mu, sigma, T]).all():
        raise ValueError("GBM inputs must be finite")

    # Local generator: np.random.seed would leak into every later simulation.
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shocks = rng.standard_normal((n_steps, n_paths))
    increments = np.exp(
        (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shocks
    )

    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = S0
    paths[1:] = S0 * np.cumprod(increments, axis=0)
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

    The tail holds ``k = round-half-up((1 - confidence_level) * n)``
    observations (at least one). VaR is the least extreme of those returns and
    CVaR is their mean. ``int((1 - confidence) * n)`` is not used: it is off by
    one and ``1 - confidence`` is not exact in floating point.

    Both values are return quantiles. They are negative when the tail is a loss.

    Parameters
    ----------
    returns : pd.Series  Portfolio returns, one observation per row.
    confidence_level : float  E.g. 0.95 for 95% VaR. Must lie in (0, 1).

    Returns
    -------
    (var, cvar)
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    values = np.asarray(returns, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("returns must contain at least one finite observation")

    ordered = np.sort(values)
    # floor(x + 0.5) is round-half-up and absorbs a 1-ulp error on exact integers.
    tail_count = int(np.floor((1.0 - confidence_level) * ordered.size + 0.5))
    tail_count = min(max(tail_count, 1), ordered.size)
    tail = ordered[:tail_count]
    return float(tail[-1]), float(tail.mean())


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
    if len(cum_returns) == 0:
        return float("nan")
    rolling_max = cum_returns.cummax()
    safe_max = rolling_max.replace(0, np.nan)
    drawdown = (cum_returns - rolling_max) / safe_max
    if drawdown.isna().all():
        return float("nan")
    return float(drawdown.min())


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
