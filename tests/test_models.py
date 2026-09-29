"""Reference tests for option pricing, simulation, and risk metrics."""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    black_scholes,
    geometric_brownian_motion,
    max_drawdown,
    monte_carlo_paths,
    var_cvar,
)


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


def test_monte_carlo_compounds_daily_drift_not_annualised_drift() -> None:
    daily = 0.001
    history = pd.Series(np.full(30, daily))
    paths = monte_carlo_paths(history, n_paths=4, horizon=20, initial_value=1.0, seed=1)
    expected = (1.0 + daily) ** np.arange(21)
    assert paths.shape == (21, 4)
    assert np.allclose(paths[:, 0], expected)
    # An accidental * 252 on the drift would make the first step 1.252, not 1.001.
    assert paths[1, 0] == pytest.approx(1.001)


def test_monte_carlo_volatility_is_per_step() -> None:
    rng = np.random.default_rng(0)
    daily_vol = 0.02
    history = pd.Series(rng.normal(0.0005, daily_vol, 200_000))
    paths = monte_carlo_paths(history, n_paths=8_000, horizon=1, seed=123)
    step = paths[1] - 1.0
    assert step.mean() == pytest.approx(history.mean(), abs=1e-3)
    assert step.std(ddof=1) == pytest.approx(history.std(ddof=1), abs=1e-3)
    # Annualising by sqrt(252) inside the daily step would land near 0.32, not 0.02.
    assert step.std(ddof=1) < 0.05


def test_monte_carlo_seed_is_reproducible_and_local() -> None:
    history = pd.Series(np.linspace(-0.02, 0.03, 40))
    first = monte_carlo_paths(history, n_paths=30, horizon=10, seed=7)
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    second = monte_carlo_paths(history, n_paths=30, horizon=10, seed=7)
    after = np.random.random()
    assert np.allclose(first, second)
    assert before == after


def test_monte_carlo_rejects_short_history() -> None:
    with pytest.raises(ValueError, match="two finite"):
        monte_carlo_paths(pd.Series([0.01]), n_paths=5, horizon=3, seed=1)


def test_gbm_log_return_mean_and_volatility() -> None:
    mu, sigma, years = 0.08, 0.20, 1.0
    paths = geometric_brownian_motion(
        S0=100.0, mu=mu, sigma=sigma, T=years, n_steps=252, n_paths=20_000, seed=7
    )
    log_return = np.log(paths[-1] / paths[0])
    assert log_return.mean() == pytest.approx((mu - 0.5 * sigma**2) * years, abs=0.01)
    assert log_return.std(ddof=1) == pytest.approx(sigma * np.sqrt(years), abs=0.01)


def test_gbm_seed_does_not_touch_global_rng() -> None:
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    first = geometric_brownian_motion(50.0, 0.05, 0.1, 1.0, 30, 8, seed=4)
    second = geometric_brownian_motion(50.0, 0.05, 0.1, 1.0, 30, 8, seed=4)
    after = np.random.random()
    assert np.allclose(first, second)
    assert before == after


def test_historical_var_uses_empirical_quantile_and_inclusive_tail() -> None:
    # 10 sorted returns. At 80% confidence the tail is the worst
    # round-half-up(0.2 * 10) = 2 observations. VaR is the less extreme of
    # those two (-0.04) and CVaR is their mean. The old index
    # int((1 - 0.8) * 10) collapsed to 1 because 1 - 0.8 is not exact.
    values = np.array(
        [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    )
    var, cvar = var_cvar(pd.Series(values), 0.80)
    assert var == pytest.approx(-0.04)
    assert cvar == pytest.approx((-0.05 + -0.04) / 2)

    # Exactly 5% of 100 observations: worst 5 returns, VaR at the 5th.
    grid = np.linspace(-1.0, 1.0, 100)
    var95, cvar95 = var_cvar(grid, 0.95)
    worst = np.sort(grid)[:5]
    assert var95 == pytest.approx(worst[-1])
    assert cvar95 == pytest.approx(worst.mean())
    assert cvar95 <= var95


def test_var_cvar_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="between"):
        var_cvar(pd.Series([0.01, -0.02]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        var_cvar(pd.Series([np.nan, np.inf]), 0.95)


def test_max_drawdown_on_a_known_path() -> None:
    values = pd.Series([100.0, 120.0, 90.0, 95.0])
    # Peak 120, trough 90 → -25%.
    assert max_drawdown(values) == pytest.approx(-0.25)


def test_max_drawdown_with_zero_peak_is_not_infinite() -> None:
    result = max_drawdown(pd.Series([0.0, 0.0]))
    assert not np.isinf(result)


def test_gbm_zero_volatility_is_deterministic_drift() -> None:
    mu, years, steps = 0.10, 1.0, 4
    paths = geometric_brownian_motion(100.0, mu, 0.0, years, steps, 3, seed=1)
    expected_terminal = 100.0 * np.exp(mu * years)
    assert np.allclose(paths[-1], expected_terminal)
