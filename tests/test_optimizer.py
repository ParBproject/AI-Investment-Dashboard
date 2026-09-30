import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    compute_portfolio_metrics,
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    minimum_variance_frontier,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "LOW_VOL": rng.normal(0.0004, 0.010, 1000),
            "MID_VOL": rng.normal(0.0005, 0.020, 1000),
            "HIGH_VOL": rng.normal(0.0003, 0.030, 1000),
        }
    )


def _short_frontier(sample_returns: pd.DataFrame) -> dict:
    return efficient_frontier(
        sample_returns,
        n_portfolios=250,
        allow_short=True,
        random_state=7,
    )


def test_short_frontier_is_reproducible(sample_returns: pd.DataFrame) -> None:
    first = _short_frontier(sample_returns)
    second = _short_frontier(sample_returns)
    assert np.allclose(first["weights"], second["weights"])


def test_short_frontier_weights_sum_to_one(sample_returns: pd.DataFrame) -> None:
    frontier = _short_frontier(sample_returns)
    assert np.allclose(frontier["weights"].sum(axis=1), 1.0, atol=1e-10)


def test_short_frontier_respects_asset_bounds(sample_returns: pd.DataFrame) -> None:
    frontier = _short_frontier(sample_returns)
    assert frontier["weights"].min() >= -1.0 - 1e-10
    assert frontier["weights"].max() <= 1.0 + 1e-10


def test_short_frontier_metrics_are_finite(sample_returns: pd.DataFrame) -> None:
    frontier = _short_frontier(sample_returns)
    assert np.isfinite(frontier["rets"]).all()
    assert np.isfinite(frontier["vols"]).all()
    assert np.isfinite(frontier["sharpes"]).all()


def test_long_only_frontier_is_reproducible(sample_returns: pd.DataFrame) -> None:
    first = efficient_frontier(
        sample_returns,
        n_portfolios=100,
        random_state=19,
    )
    second = efficient_frontier(
        sample_returns,
        n_portfolios=100,
        random_state=19,
    )

    assert np.allclose(first["weights"], second["weights"])
    assert np.all(first["weights"] >= 0)
    assert np.all(first["weights"] <= 1)


def test_optimizers_return_valid_allocations(sample_returns: pd.DataFrame) -> None:
    sharpe_weights, _, _, sharpe = max_sharpe_weights(
        sample_returns,
        random_state=11,
    )
    min_var_weights, _, min_var_vol = min_variance_weights(sample_returns)

    assert np.isclose(sharpe_weights.sum(), 1.0, atol=1e-8)
    assert np.isclose(min_var_weights.sum(), 1.0, atol=1e-8)
    assert np.all(sharpe_weights >= -1e-10)
    assert np.all(min_var_weights >= -1e-10)
    assert np.isfinite(sharpe)
    assert min_var_vol > 0


def test_invalid_frontier_size_is_rejected(sample_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_portfolios"):
        efficient_frontier(sample_returns, n_portfolios=0)


def test_portfolio_metrics_annualise_return_and_variance() -> None:
    weights = np.array([0.25, 0.75])
    mean_returns = np.array([0.001, 0.002])
    cov = np.array([[0.0001, 0.00002], [0.00002, 0.0004]])
    ann_ret, ann_vol = compute_portfolio_metrics(weights, mean_returns, cov, trading_days=252)

    daily_return = 0.25 * 0.001 + 0.75 * 0.002
    daily_var = float(weights @ cov @ weights)
    assert ann_ret == pytest.approx(daily_return * 252)
    assert ann_vol == pytest.approx(np.sqrt(daily_var * 252))


def test_minimum_variance_frontier_is_the_upper_limb(sample_returns: pd.DataFrame) -> None:
    curve = minimum_variance_frontier(sample_returns, n_points=12)
    assert curve["weights"].shape[0] >= 2
    assert np.allclose(curve["weights"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(curve["weights"] >= -1e-6)
    assert np.all(curve["weights"] <= 1.0 + 1e-6)
    assert np.all(np.diff(curve["rets"]) >= -1e-8)
    # Past the minimum-variance point, risk should not fall as return rises.
    assert np.all(np.diff(curve["vols"]) >= -1e-4)
    assert curve["vols"][0] == pytest.approx(np.min(curve["vols"]), abs=1e-6)


def test_minimum_variance_weights_match_inverse_variance() -> None:
    rng = np.random.default_rng(0)
    # Independent assets. The long-only minimum-variance mix is inverse variance.
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.01, 40_000),
            "B": rng.normal(0.0, 0.02, 40_000),
        }
    )
    weights, _, _ = min_variance_weights(returns)
    variance = returns.var().to_numpy()
    expected = (1.0 / variance)
    expected = expected / expected.sum()
    assert weights == pytest.approx(expected, abs=0.02)


def test_zero_volatility_sharpe_is_undefined() -> None:
    # Constant returns have no risk. Reporting Sharpe 0 would look like a
    # finished calculation instead of an undefined ratio.
    flat = pd.DataFrame({"A": np.ones(12), "B": np.full(12, 2.0)})
    _, _, ann_vol, sharpe = max_sharpe_weights(flat, random_state=0)
    assert ann_vol == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(sharpe)

    cloud = efficient_frontier(flat, n_portfolios=5, random_state=1)
    assert np.isnan(cloud["sharpes"]).all()


def test_max_sharpe_rejects_a_non_positive_start_count(sample_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_starts"):
        max_sharpe_weights(sample_returns, random_state=1, n_starts=0)


def test_max_sharpe_accepts_a_warm_start(sample_returns: pd.DataFrame) -> None:
    weights, _, _, sharpe = max_sharpe_weights(
        sample_returns,
        random_state=3,
        n_starts=2,
        initial_weights=np.array([0.2, 0.3, 0.5]),
    )
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    assert np.isfinite(sharpe)


def test_non_finite_returns_are_rejected(sample_returns: pd.DataFrame) -> None:
    invalid = sample_returns.copy()
    invalid.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        efficient_frontier(invalid, n_portfolios=10)
