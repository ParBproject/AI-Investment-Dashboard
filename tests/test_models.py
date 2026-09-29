"""Reference tests for option pricing, simulation, and risk metrics."""

import numpy as np
import pytest

from src.models import black_scholes


# Hull, Options, Futures, and Other Derivatives. European call on a
# non-dividend-paying stock. Hull rounds these to 4.76 and 2.40.
# The literals below are the unrounded closed form (independent of this repo).
HULL_CALL_42 = {
    "S": 42.0,
    "K": 40.0,
    "T": 0.5,
    "r": 0.10,
    "sigma": 0.20,
    "call": 4.75942239,
    "put": 0.80859937,
    "delta": 0.77913129,
    "gamma": 0.04996267,
    "vega": 8.81341506,       # textbook dV/dσ, not per 1%
    "theta_year": -4.55909219,
    "rho": 13.98204591,       # textbook dV/dr, not per 1%
}

HULL_CALL_49 = {
    "S": 49.0,
    "K": 50.0,
    "T": 20.0 / 52.0,
    "r": 0.05,
    "sigma": 0.20,
    "call": 2.40052732,
    "put": 2.44817544,
    "delta": 0.52160466,
    "gamma": 0.06554404,
    "vega": 12.10547988,
    "theta_year": -4.30532982,
    "rho": 8.90696195,
}


def test_hull_prices_and_scaled_greeks() -> None:
    for case in (HULL_CALL_42, HULL_CALL_49):
        call, put, greeks = black_scholes(
            case["S"], case["K"], case["T"], case["r"], case["sigma"]
        )
        assert call == pytest.approx(case["call"], abs=1e-6)
        assert put == pytest.approx(case["put"], abs=1e-6)
        assert greeks["delta_call"] == pytest.approx(case["delta"], abs=1e-6)
        assert greeks["delta_put"] == pytest.approx(case["delta"] - 1.0, abs=1e-6)
        assert greeks["gamma"] == pytest.approx(case["gamma"], abs=1e-6)
        # Dashboard convention: vega and rho per 1 percentage point, theta per day.
        assert greeks["vega"] == pytest.approx(case["vega"] / 100.0, abs=1e-8)
        assert greeks["theta_call"] == pytest.approx(case["theta_year"] / 365.0, abs=1e-8)
        assert greeks["rho_call"] == pytest.approx(case["rho"] / 100.0, abs=1e-8)


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "sigma"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (100.0, 120.0, 0.25, 0.01, 0.35),
        (50.0, 40.0, 2.0, 0.0, 0.15),
        (80.0, 100.0, 0.75, 0.08, 0.50),
    ],
)
def test_put_call_parity(S: float, K: float, T: float, r: float, sigma: float) -> None:
    call, put, _ = black_scholes(S, K, T, r, sigma)
    assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


def test_greeks_match_finite_differences() -> None:
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.03, 0.22
    call, _, greeks = black_scholes(S, K, T, r, sigma)

    bump = 1e-4
    call_up, _, _ = black_scholes(S + bump, K, T, r, sigma)
    call_dn, _, _ = black_scholes(S - bump, K, T, r, sigma)
    assert greeks["delta_call"] == pytest.approx((call_up - call_dn) / (2 * bump), abs=1e-4)
    assert greeks["gamma"] == pytest.approx(
        (call_up - 2 * call + call_dn) / bump**2, abs=1e-3
    )

    # One volatility point is 0.01. The centered difference estimates dV/dσ,
    # and the quoted vega is that derivative times 0.01.
    vol_bump = 0.01
    call_vol_up, _, _ = black_scholes(S, K, T, r, sigma + vol_bump)
    call_vol_dn, _, _ = black_scholes(S, K, T, r, sigma - vol_bump)
    vega_per_point = (call_vol_up - call_vol_dn) / (2 * vol_bump) * vol_bump
    assert greeks["vega"] == pytest.approx(vega_per_point, rel=1e-3)

    day = 1.0 / 365.0
    call_tomorrow, _, _ = black_scholes(S, K, T - day, r, sigma)
    assert greeks["theta_call"] == pytest.approx(call_tomorrow - call, rel=1e-2, abs=1e-4)

    call_r_up, _, _ = black_scholes(S, K, T, r + vol_bump, sigma)
    call_r_dn, _, _ = black_scholes(S, K, T, r - vol_bump, sigma)
    rho_per_point = (call_r_up - call_r_dn) / (2 * vol_bump) * vol_bump
    assert greeks["rho_call"] == pytest.approx(rho_per_point, rel=1e-3)


def test_zero_volatility_uses_discounted_forward_intrinsic() -> None:
    S, K, T, r = 100.0, 90.0, 1.0, 0.05
    call, put, greeks = black_scholes(S, K, T, r, 0.0)

    present_strike = K * np.exp(-r * T)
    assert call == pytest.approx(S - present_strike, abs=1e-10)
    assert put == pytest.approx(0.0, abs=1e-10)
    # The old implementation returned the undiscounted intrinsic, 10, and no Greeks.
    assert call != pytest.approx(10.0, abs=0.1)
    assert greeks["delta_call"] == pytest.approx(1.0)
    assert greeks["gamma"] == 0.0
    assert greeks["vega"] == 0.0
    assert np.isfinite(greeks["theta_call"])
    assert call - put == pytest.approx(S - present_strike, abs=1e-10)


def test_zero_volatility_out_of_the_money_put() -> None:
    S, K, T, r = 90.0, 100.0, 1.0, 0.02
    call, put, greeks = black_scholes(S, K, T, r, 0.0)
    present_strike = K * np.exp(-r * T)
    assert call == pytest.approx(0.0, abs=1e-10)
    assert put == pytest.approx(present_strike - S, abs=1e-10)
    assert greeks["delta_put"] == pytest.approx(-1.0)
    assert greeks["rho_put"] == pytest.approx(-(T * present_strike) / 100.0, abs=1e-10)


def test_expiry_returns_intrinsic_and_finite_greeks() -> None:
    call, put, greeks = black_scholes(100.0, 95.0, 0.0, 0.05, 0.2)
    assert call == pytest.approx(5.0)
    assert put == pytest.approx(0.0)
    assert greeks["delta_call"] == pytest.approx(1.0)
    assert np.isnan(greeks["d1"])
    for key in ("delta_call", "delta_put", "gamma", "vega", "theta_call", "rho_call"):
        assert np.isfinite(greeks[key])


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-1.0, 100.0, 1.0, 0.01, 0.2), "positive"),
        ((100.0, 0.0, 1.0, 0.01, 0.2), "positive"),
        ((100.0, 100.0, -0.1, 0.01, 0.2), "negative"),
        ((100.0, 100.0, 1.0, 0.01, -0.2), "negative"),
        ((np.nan, 100.0, 1.0, 0.01, 0.2), "finite"),
    ],
)
def test_black_scholes_rejects_invalid_inputs(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        black_scholes(*args)
