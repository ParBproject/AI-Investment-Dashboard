"""Derivatives pricing and validation utilities.

This module complements the portfolio and scenario models with multiple option
pricing methods so theoretical prices can be cross-checked instead of relying
on one formula in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .models import black_scholes


@dataclass(frozen=True)
class MonteCarloOptionResult:
    """Risk-neutral Monte Carlo option estimate."""

    price: float
    standard_error: float
    n_paths: int


@dataclass(frozen=True)
class OptionPricingComparison:
    """Cross-method option-pricing comparison."""

    black_scholes: float
    binomial: float
    monte_carlo: float
    monte_carlo_standard_error: float
    binomial_difference: float
    monte_carlo_difference: float


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    values = np.asarray([spot, strike, maturity, volatility], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("option inputs must be finite")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")


def _validate_option_type(option_type: str) -> str:
    normalized = option_type.lower().strip()
    if normalized not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return normalized


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    """Return one Black-Scholes option price using the shared model."""
    _validate_inputs(spot, strike, maturity, volatility)
    kind = _validate_option_type(option_type)
    call_price, put_price, _ = black_scholes(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
    )
    return float(call_price if kind == "call" else put_price)


def put_call_parity_gap(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    call_price: float,
    put_price: float,
) -> float:
    """Return call-put parity residual for non-dividend European options."""
    values = np.asarray(
        [spot, strike, maturity, risk_free_rate, call_price, put_price],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("put-call parity inputs must be finite")
    if spot <= 0 or strike <= 0 or maturity < 0:
        raise ValueError("spot/strike must be positive and maturity non-negative")
    theoretical_difference = spot - strike * np.exp(-risk_free_rate * maturity)
    return float((call_price - put_price) - theoretical_difference)


def binomial_option_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    *,
    steps: int = 300,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """Price an option with a Cox-Ross-Rubinstein recombining binomial tree."""
    _validate_inputs(spot, strike, maturity, volatility)
    kind = _validate_option_type(option_type)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be a positive integer")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    if maturity == 0 or volatility == 0:
        intrinsic = max(
            spot - strike if kind == "call" else strike - spot,
            0.0,
        )
        return float(intrinsic)

    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1.0 / up
    growth = np.exp(risk_free_rate * dt)
    probability = (growth - down) / (up - down)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "binomial risk-neutral probability is outside [0, 1]; "
            "increase steps or review inputs"
        )

    levels = np.arange(steps + 1)
    terminal_spot = spot * (up ** levels) * (down ** (steps - levels))
    if kind == "call":
        values = np.maximum(terminal_spot - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spot, 0.0)

    discount = np.exp(-risk_free_rate * dt)
    for step in range(steps - 1, -1, -1):
        values = discount * (
            probability * values[1 : step + 2]
            + (1.0 - probability) * values[: step + 1]
        )

        if american:
            nodes = np.arange(step + 1)
            node_spot = spot * (up ** nodes) * (down ** (step - nodes))
            if kind == "call":
                intrinsic = np.maximum(node_spot - strike, 0.0)
            else:
                intrinsic = np.maximum(strike - node_spot, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def monte_carlo_option_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    *,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = 42,
    antithetic: bool = True,
) -> MonteCarloOptionResult:
    """Price a European option under risk-neutral GBM with standard error."""
    _validate_inputs(spot, strike, maturity, volatility)
    kind = _validate_option_type(option_type)
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
        raise TypeError("n_paths must be a positive integer")
    if n_paths < 100:
        raise ValueError("n_paths must be at least 100")

    if maturity == 0 or volatility == 0:
        intrinsic = max(
            spot - strike if kind == "call" else strike - spot,
            0.0,
        )
        return MonteCarloOptionResult(
            price=float(intrinsic),
            standard_error=0.0,
            n_paths=int(n_paths),
        )

    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        shocks = rng.standard_normal(half)
        shocks = np.concatenate([shocks, -shocks])[:n_paths]
    else:
        shocks = rng.standard_normal(n_paths)

    terminal = spot * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * maturity
        + volatility * np.sqrt(maturity) * shocks
    )
    if kind == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)

    discounted = np.exp(-risk_free_rate * maturity) * payoff
    price = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return MonteCarloOptionResult(
        price=price,
        standard_error=standard_error,
        n_paths=int(n_paths),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    *,
    option_type: str = "call",
    lower: float = 1e-4,
    upper: float = 5.0,
    tolerance: float = 1e-8,
) -> float:
    """Recover Black-Scholes implied volatility with a bracketed root solver."""
    kind = _validate_option_type(option_type)
    values = np.asarray(
        [market_price, spot, strike, maturity, risk_free_rate],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("implied-volatility inputs must be finite")
    if market_price < 0:
        raise ValueError("market_price must be non-negative")
    if maturity <= 0:
        raise ValueError("maturity must be positive for implied volatility")
    if lower <= 0 or upper <= lower:
        raise ValueError("volatility bounds must satisfy 0 < lower < upper")

    discounted_strike = strike * np.exp(-risk_free_rate * maturity)
    if kind == "call":
        lower_bound = max(spot - discounted_strike, 0.0)
        upper_bound = spot
    else:
        lower_bound = max(discounted_strike - spot, 0.0)
        upper_bound = discounted_strike

    if not lower_bound - 1e-12 <= market_price <= upper_bound + 1e-12:
        raise ValueError(
            "market_price violates no-arbitrage bounds for the selected option"
        )

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                spot,
                strike,
                maturity,
                risk_free_rate,
                volatility,
                option_type=kind,
            )
            - market_price
        )

    f_lower = objective(lower)
    f_upper = objective(upper)
    if abs(f_lower) <= tolerance:
        return float(lower)
    if abs(f_upper) <= tolerance:
        return float(upper)
    if f_lower * f_upper > 0:
        raise ValueError("implied volatility is not bracketed by the supplied bounds")

    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=tolerance,
            rtol=tolerance,
            maxiter=200,
        )
    )


def compare_option_pricers(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    binomial_steps: int = 400,
    monte_carlo_paths: int = 100_000,
    seed: int | None = 42,
) -> OptionPricingComparison:
    """Compare Black-Scholes, binomial, and Monte Carlo prices."""
    bs = black_scholes_price(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
        option_type=option_type,
    )
    tree = binomial_option_price(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
        steps=binomial_steps,
        option_type=option_type,
    )
    mc = monte_carlo_option_price(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
        n_paths=monte_carlo_paths,
        option_type=option_type,
        seed=seed,
    )
    return OptionPricingComparison(
        black_scholes=bs,
        binomial=tree,
        monte_carlo=mc.price,
        monte_carlo_standard_error=mc.standard_error,
        binomial_difference=tree - bs,
        monte_carlo_difference=mc.price - bs,
    )
