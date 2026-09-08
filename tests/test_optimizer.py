import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
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


def test_non_finite_returns_are_rejected(sample_returns: pd.DataFrame) -> None:
    invalid = sample_returns.copy()
    invalid.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        efficient_frontier(invalid, n_portfolios=10)
