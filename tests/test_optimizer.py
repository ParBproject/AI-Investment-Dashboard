import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_risk_contributions,
    risk_parity_weights,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Deterministic synthetic returns with deliberately different volatilities."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "LOW_VOL": rng.normal(0.0004, 0.010, 1000),
            "MID_VOL": rng.normal(0.0005, 0.020, 1000),
            "HIGH_VOL": rng.normal(0.0003, 0.030, 1000),
        }
    )


def test_risk_parity_equalizes_risk_contributions() -> None:
    # Orthogonal, zero-mean return patterns produce a diagonal covariance
    # matrix with a known equal-risk solution. This keeps the test independent
    # of random-sample covariance differences across numerical-library builds.
    risk_parity_returns = pd.DataFrame(
        {
            "LOW_VOL": np.array([1, -1, 1, -1], dtype=float) * 0.01,
            "MID_VOL": np.array([1, 1, -1, -1], dtype=float) * 0.02,
            "HIGH_VOL": np.array([1, -1, -1, 1], dtype=float) * 0.03,
        }
    )

    weights, _, volatility, contributions = risk_parity_weights(
        risk_parity_returns
    )

    assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    assert np.all(weights >= -1e-10)
    assert volatility > 0

    contribution_share = contributions / contributions.sum()
    expected_share = np.repeat(1 / len(weights), len(weights))
    assert np.allclose(contribution_share, expected_share, atol=1e-6)

    expected_inverse_vol = np.array([1 / 0.01, 1 / 0.02, 1 / 0.03])
    expected_inverse_vol /= expected_inverse_vol.sum()
    assert np.allclose(weights, expected_inverse_vol, atol=1e-6)


def test_risk_contributions_sum_to_portfolio_volatility(
    sample_returns: pd.DataFrame,
) -> None:
    weights = np.array([0.5, 0.3, 0.2])
    cov_matrix = sample_returns.cov().values
    contributions = portfolio_risk_contributions(weights, cov_matrix)

    portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights * 252)
    assert np.isclose(contributions.sum(), portfolio_volatility, rtol=1e-10)


def test_short_frontier_respects_bounds_and_is_reproducible(
    sample_returns: pd.DataFrame,
) -> None:
    first = efficient_frontier(
        sample_returns,
        n_portfolios=250,
        allow_short=True,
        random_state=7,
    )
    second = efficient_frontier(
        sample_returns,
        n_portfolios=250,
        allow_short=True,
        random_state=7,
    )

    assert np.allclose(first["weights"], second["weights"])
    assert np.allclose(first["weights"].sum(axis=1), 1.0, atol=1e-10)
    assert first["weights"].min() >= -1.0 - 1e-10
    assert first["weights"].max() <= 1.0 + 1e-10
    assert np.isfinite(first["rets"]).all()
    assert np.isfinite(first["vols"]).all()
    assert np.isfinite(first["sharpes"]).all()


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
