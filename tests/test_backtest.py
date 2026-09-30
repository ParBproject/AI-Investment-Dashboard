"""Walk-forward execution: next bar, costs, and no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    performance_summary,
    select_lookback,
    simulate_scheduled_weights,
    walk_forward_comparison,
    walk_forward_targets,
)


def test_entry_cost_and_drifted_next_day_match_hand_calculation() -> None:
    returns = pd.DataFrame(
        {"A": [-0.10, 0.05], "B": [0.20, -0.05]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    paths = simulate_scheduled_weights(
        returns,
        {0: np.array([0.5, 0.5])},
        commission_bps=5.0,
        slippage_bps=5.0,
    )
    cost_rate = (5.0 + 5.0) / 10_000.0
    assert paths["turnover"].iloc[0] == pytest.approx(1.0)
    assert paths["cost"].iloc[0] == pytest.approx(cost_rate)
    assert paths["turnover"].iloc[1] == pytest.approx(0.0)
    assert paths["cost"].iloc[1] == pytest.approx(0.0)

    factor0 = (1.0 - cost_rate) * 1.05
    drifted = np.array([0.5 * 0.90, 0.5 * 1.20])
    drifted = drifted / drifted.sum()
    gross1 = float(drifted @ np.array([0.05, -0.05]))
    assert paths["gross_return"].iloc[1] == pytest.approx(gross1)
    assert paths["wealth"].iloc[-1] == pytest.approx(factor0 * (1.0 + gross1))


def test_commission_and_slippage_are_both_charged_once() -> None:
    returns = pd.DataFrame({"A": [0.01], "B": [0.02]})
    weights = {0: np.array([0.25, 0.75])}
    both = simulate_scheduled_weights(returns, weights, commission_bps=10, slippage_bps=10)
    commission_only = simulate_scheduled_weights(
        returns, weights, commission_bps=20, slippage_bps=0
    )
    slippage_only = simulate_scheduled_weights(
        returns, weights, commission_bps=0, slippage_bps=20
    )
    free = simulate_scheduled_weights(returns, weights, commission_bps=0, slippage_bps=0)
    assert both["wealth"].iloc[-1] == pytest.approx(commission_only["wealth"].iloc[-1])
    assert both["wealth"].iloc[-1] == pytest.approx(slippage_only["wealth"].iloc[-1])
    assert both["wealth"].iloc[-1] < free["wealth"].iloc[-1]


def test_rebalance_charges_both_legs_of_drifted_weights() -> None:
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [-0.10, 0.00]})
    paths = simulate_scheduled_weights(
        returns,
        {0: np.array([0.5, 0.5]), 1: np.array([0.5, 0.5])},
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    # Day-0 return is zero, so weights drift from 50/50 to 55/45.
    # Restoring 50/50 buys 0.05 and sells 0.05.
    assert paths["gross_return"].iloc[0] == pytest.approx(0.0)
    assert paths["turnover"].iloc[1] == pytest.approx(0.10)


def test_short_entry_charges_the_long_and_the_short() -> None:
    returns = pd.DataFrame({"A": [0.00], "B": [0.00]})
    paths = simulate_scheduled_weights(
        returns,
        {0: np.array([1.2, -0.2])},
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    assert paths["turnover"].iloc[0] == pytest.approx(1.4)


def test_weights_do_not_see_the_return_they_earn() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    returns = pd.DataFrame(
        {
            "A": [0.00, 0.00, 0.00, 1.00],
            "B": [0.02, 0.02, 0.02, -0.40],
        },
        index=index,
    )
    seen: list[pd.Timestamp] = []

    def pick_higher_mean(window: pd.DataFrame) -> np.ndarray:
        seen.append(window.index[-1])
        choice = int(np.argmax(window.mean().to_numpy()))
        weights = np.zeros(window.shape[1])
        weights[choice] = 1.0
        return weights

    targets = walk_forward_targets(
        returns, lookback=3, rebalance_every=3, weight_fn=pick_higher_mean
    )
    assert list(targets) == [3]
    assert seen == [index[2]]
    assert targets[3] == pytest.approx([0.0, 1.0])

    shocked = returns.copy()
    shocked.iloc[3, 0] = 5.0
    shocked_targets = walk_forward_targets(
        shocked, lookback=3, rebalance_every=3, weight_fn=pick_higher_mean
    )
    assert shocked_targets[3] == pytest.approx(targets[3])

    paths = simulate_scheduled_weights(
        returns, targets, commission_bps=0.0, slippage_bps=0.0
    )
    # A peek at day 4 would buy A and earn +100%. The schedule holds B.
    assert paths["gross_return"].iloc[3] == pytest.approx(-0.40)
    assert paths["net_return"].iloc[:3].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_reversed_dates_are_rejected() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "B": [0.0, 0.01, -0.01]},
        index=pd.to_datetime(["2024-01-04", "2024-01-03", "2024-01-02"]),
    )
    with pytest.raises(ValueError, match="time-ordered"):
        walk_forward_targets(
            returns,
            lookback=2,
            rebalance_every=1,
            weight_fn=lambda window: np.ones(window.shape[1]) / window.shape[1],
        )


def test_performance_summary_pins_cagr_sharpe_and_drawdown() -> None:
    net = np.array([0.01, -0.02, 0.03])
    paths = pd.DataFrame(
        {
            "net_return": net,
            "gross_return": net,
            "turnover": [1.0, 0.0, 0.0],
            "rebalance": [1, 0, 0],
        }
    )
    summary = performance_summary(paths, risk_free_rate=0.0, trading_days=252)
    terminal = 1.01 * 0.98 * 1.03
    assert summary["cagr"] == pytest.approx(terminal ** (252 / 3) - 1.0)
    volatility = float(pd.Series(net).std(ddof=1) * np.sqrt(252))
    assert summary["volatility"] == pytest.approx(volatility)
    assert summary["sharpe"] == pytest.approx(float(net.mean() * 252) / volatility)
    assert summary["max_drawdown"] == pytest.approx(-0.02)
    assert summary["n_days"] == 3
    assert summary["avg_turnover"] == pytest.approx(1.0)


def test_select_lookback_leaves_two_held_out_days() -> None:
    assert select_lookback(300) == 252
    assert select_lookback(254) == 252
    assert select_lookback(253) == 63
    assert select_lookback(65) == 63
    assert select_lookback(64) is None
    assert select_lookback(10) is None


def _two_asset_history(rows: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    shared = rng.normal(0.0004, 0.01, rows)
    return pd.DataFrame(
        {
            "A": shared,
            "B": shared + rng.normal(0.0, 0.002, rows),
        },
        index=pd.bdate_range("2022-01-03", periods=rows),
    )


def test_short_history_skips_the_walk_forward() -> None:
    short = _two_asset_history(40)
    assert walk_forward_comparison(short) is None
    with pytest.raises(ValueError, match="longer than the lookback"):
        walk_forward_comparison(short, lookback=63)


def test_walk_forward_is_reproducible_and_costs_reduce_wealth() -> None:
    history = _two_asset_history()
    kwargs = dict(lookback=63, rebalance_every=21, n_starts=3, random_state=9, risk_free_rate=0.01)
    first = walk_forward_comparison(history, commission_bps=0, slippage_bps=0, **kwargs)
    second = walk_forward_comparison(history, commission_bps=0, slippage_bps=0, **kwargs)
    costly = walk_forward_comparison(history, commission_bps=25, slippage_bps=25, **kwargs)
    assert first is not None and second is not None and costly is not None
    assert first["strategy_wealth"].iloc[-1] == pytest.approx(second["strategy_wealth"].iloc[-1])
    assert costly["strategy_wealth"].iloc[-1] < first["strategy_wealth"].iloc[-1]
    assert first["strategy"]["n_days"] == len(history) - 63
    assert first["n_rebalances"] == len(range(63, len(history), 21))


def test_identical_assets_match_equal_weight_before_costs() -> None:
    rng = np.random.default_rng(0)
    shared = rng.normal(0.0005, 0.01, 120)
    history = pd.DataFrame({"A": shared, "B": shared})
    result = walk_forward_comparison(
        history,
        risk_free_rate=0.0,
        lookback=63,
        rebalance_every=21,
        commission_bps=0.0,
        slippage_bps=0.0,
        n_starts=3,
        random_state=1,
    )
    assert result is not None
    assert result["strategy_wealth"].iloc[-1] == pytest.approx(
        result["benchmark_wealth"].iloc[-1]
    )


def test_walk_forward_beats_equal_weight_when_one_asset_dominates() -> None:
    rng = np.random.default_rng(2)
    history = pd.DataFrame(
        {
            "GOOD": rng.normal(0.003, 0.005, 180),
            "BAD": rng.normal(-0.002, 0.02, 180),
        }
    )
    result = walk_forward_comparison(
        history,
        risk_free_rate=0.0,
        allow_short=False,
        lookback=63,
        rebalance_every=21,
        commission_bps=0.0,
        slippage_bps=0.0,
        n_starts=4,
        random_state=0,
    )
    assert result is not None
    assert result["strategy"]["cagr"] > result["benchmark"]["cagr"]
