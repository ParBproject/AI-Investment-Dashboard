"""
backtest.py
===========
Walk-forward portfolio backtest with next-bar execution and trading costs.

A weight vector chosen from returns strictly before date ``t`` earns the
close-to-close simple return that ends on date ``t``. Costs are charged on
the rebalance, before that return, on both purchases and sales.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.models import max_drawdown
from src.optimizer import TRADING_DAYS, max_sharpe_weights

# A full year of trading days when the sample has one. Shorter uploads still
# get a walk-forward, but only when a 63-day window leaves a held-out tail.
PREFERRED_LOOKBACK = 252
MINIMUM_LOOKBACK = 63
DEFAULT_REBALANCE_EVERY = 21
DEFAULT_COMMISSION_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_WALK_FORWARD_STARTS = 8


def select_lookback(
    n_observations: int,
    preferred: int = PREFERRED_LOOKBACK,
    minimum: int = MINIMUM_LOOKBACK,
) -> Optional[int]:
    """
    Training length that leaves at least two out-of-sample returns.

    Two held-out days is the smallest sample with a defined volatility.
    Returns None when the history cannot support ``minimum`` plus those days.
    """
    if preferred < 2 or minimum < 2:
        raise ValueError("lookback must be at least 2")
    if n_observations - preferred >= 2:
        return preferred
    if n_observations - minimum >= 2:
        return minimum
    return None


def _require_time_ordered(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        raise ValueError("returns must be a DataFrame with at least one asset")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns must be time-ordered")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must contain only finite numeric values")


def _as_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).reshape(-1)
    if vector.shape != (n_assets,) or not np.isfinite(vector).all():
        raise ValueError("weights must be a finite vector, one weight per asset")
    total = float(vector.sum())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("weights must sum to 1")
    return vector / total


def traded_notional(previous: np.ndarray, target: np.ndarray) -> float:
    """
    Gross traded fraction of NAV: purchases plus sales.

    Entering a fully invested long-only portfolio from cash trades 1.0.
    A rebalance that sells 5% and buys 5% trades 0.10. Both legs are included
    because commission and slippage are charged on each.
    """
    delta = np.asarray(target, dtype=float) - np.asarray(previous, dtype=float)
    buys = float(delta[delta > 0.0].sum())
    sells = float((-delta[delta < 0.0]).sum())
    return buys + sells


def _cost_fraction(
    previous: np.ndarray,
    target: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[float, float]:
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("commission and slippage must be non-negative")
    turnover = traded_notional(previous, target)
    rate = (commission_bps + slippage_bps) / 10_000.0
    return turnover, turnover * rate


def _drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    """Mark weights to market. Costs are a cash drag and do not change mix."""
    growth = weights * (1.0 + asset_returns)
    gross = float(growth.sum())
    if gross <= 1e-12:
        return np.zeros_like(weights)
    return growth / gross


def simulate_scheduled_weights(
    returns: pd.DataFrame,
    weights_by_iloc: dict[int, np.ndarray],
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """
    Apply a precomputed weight schedule.

    ``weights_by_iloc[i]`` is chosen using information available before
    ``returns.iloc[i]`` is known. It replaces the drifted holdings at that
    close, pays costs, and then earns ``returns.iloc[i]``. Days with no
    scheduled trade keep the drifted weights and pay nothing. Wealth starts
    at 1 in cash, so rows before the first trade earn zero.
    """
    _require_time_ordered(returns)
    n_assets = returns.shape[1]
    held = np.zeros(n_assets)
    wealth = 1.0
    rows: list[dict] = []

    for i, (_, asset_returns) in enumerate(returns.iterrows()):
        day = asset_returns.to_numpy(dtype=float)
        turnover = 0.0
        cost = 0.0
        if i in weights_by_iloc:
            target = _as_weights(weights_by_iloc[i], n_assets)
            turnover, cost = _cost_fraction(held, target, commission_bps, slippage_bps)
            held = target

        gross = float(held @ day) if held.any() else 0.0
        if cost >= 1.0 or (1.0 + gross) <= 0.0:
            net_factor = 0.0
            held = np.zeros(n_assets)
        else:
            net_factor = (1.0 - cost) * (1.0 + gross)
            if held.any():
                held = _drift_weights(held, day)
        wealth *= net_factor
        rows.append(
            {
                "gross_return": gross,
                "net_return": net_factor - 1.0,
                "cost": cost,
                "turnover": turnover,
                "rebalance": int(i in weights_by_iloc),
                "wealth": wealth,
            }
        )

    return pd.DataFrame(rows, index=returns.index)


def walk_forward_targets(
    returns: pd.DataFrame,
    lookback: int,
    rebalance_every: int,
    weight_fn: Callable[[pd.DataFrame], np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Fit weights on a trailing window that ends the day before they are used.

    The window for application index ``i`` is ``returns.iloc[i - lookback:i]``.
    It does not include ``returns.iloc[i]`` or any later row.
    """
    _require_time_ordered(returns)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    if len(returns) <= lookback:
        raise ValueError("returns must be longer than the lookback window")

    targets: dict[int, np.ndarray] = {}
    for i in range(lookback, len(returns), rebalance_every):
        window = returns.iloc[i - lookback:i]
        if len(window) != lookback:
            raise ValueError("training window is shorter than lookback")
        targets[i] = _as_weights(weight_fn(window), returns.shape[1])
    if not targets:
        raise ValueError("schedule produced no rebalances")
    return targets


def performance_summary(
    paths: pd.DataFrame,
    risk_free_rate: float,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """
    CAGR, volatility, Sharpe, and max drawdown for one path frame.

    CAGR is compounded growth. Sharpe uses the annualised arithmetic mean
    minus ``risk_free_rate`` (already annual) over annualised volatility.
    The frame is the invested sample only; leading flat days must already
    have been dropped.
    """
    if trading_days < 1:
        raise ValueError("trading_days must be positive")
    net = paths["net_return"].to_numpy(dtype=float)
    gross = paths["gross_return"].to_numpy(dtype=float)
    if net.size == 0:
        raise ValueError("performance sample is empty")

    wealth = np.cumprod(np.r_[1.0, 1.0 + net])
    years = net.size / trading_days
    terminal = float(wealth[-1])
    if terminal > 0.0:
        cagr = float(terminal ** (1.0 / years) - 1.0)
    elif terminal == 0.0:
        cagr = -1.0
    else:
        cagr = float("nan")

    gross_terminal = float(np.prod(1.0 + gross))
    if gross_terminal > 0.0:
        cagr_gross = float(gross_terminal ** (1.0 / years) - 1.0)
    elif gross_terminal == 0.0:
        cagr_gross = -1.0
    else:
        cagr_gross = float("nan")

    if net.size < 2:
        volatility = float("nan")
        sharpe = float("nan")
    else:
        volatility = float(np.std(net, ddof=1) * np.sqrt(trading_days))
        ann_mean = float(np.mean(net) * trading_days)
        sharpe = (
            (ann_mean - risk_free_rate) / volatility if volatility > 1e-12 else float("nan")
        )

    rebalance_turnover = paths.loc[paths["rebalance"] == 1, "turnover"]
    avg_turnover = float(rebalance_turnover.mean()) if len(rebalance_turnover) else float("nan")

    return {
        "cagr": cagr,
        "cagr_gross": cagr_gross,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(pd.Series(wealth)),
        "n_days": int(net.size),
        "avg_turnover": avg_turnover,
    }


def _slice_from_first_trade(paths: pd.DataFrame) -> pd.DataFrame:
    traded = np.flatnonzero(paths["rebalance"].to_numpy() == 1)
    if traded.size == 0:
        raise ValueError("no rebalance in the path")
    first = int(traded[0])
    held = paths.iloc[first:].copy()
    start_wealth = 1.0 if first == 0 else float(paths["wealth"].iloc[first - 1])
    if start_wealth <= 0.0:
        raise ValueError("wealth was already wiped out before the first trade")
    held["wealth"] = held["wealth"] / start_wealth
    return held


def _max_sharpe_fn(
    risk_free_rate: float,
    allow_short: bool,
    random_state: Optional[int],
    n_starts: int,
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Max-Sharpe on each window, warm-started from the previous target."""
    previous: dict[str, Optional[np.ndarray]] = {"weights": None}

    def estimate(window: pd.DataFrame) -> np.ndarray:
        weights, _, _, _ = max_sharpe_weights(
            window,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            random_state=random_state,
            n_starts=n_starts,
            initial_weights=previous["weights"],
        )
        previous["weights"] = np.asarray(weights, dtype=float).copy()
        return previous["weights"]

    return estimate


def walk_forward_comparison(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    allow_short: bool = False,
    lookback: Optional[int] = None,
    rebalance_every: int = DEFAULT_REBALANCE_EVERY,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    n_starts: int = DEFAULT_WALK_FORWARD_STARTS,
    random_state: Optional[int] = 42,
) -> Optional[dict]:
    """
    Compare walk-forward max-Sharpe with equal weight under the same rules.

    Both sleeves rebalance on the same dates, pay the same cost schedule, and
    earn the next bar only. Returns None when the sample cannot support the
    minimum lookback. ``lookback=None`` picks 252 days when the history is
    longer than that, otherwise 63.
    """
    _require_time_ordered(returns)
    if lookback is None:
        lookback = select_lookback(len(returns))
        if lookback is None:
            return None

    sharpe_targets = walk_forward_targets(
        returns,
        lookback=lookback,
        rebalance_every=rebalance_every,
        weight_fn=_max_sharpe_fn(risk_free_rate, allow_short, random_state, n_starts),
    )
    equal_targets = walk_forward_targets(
        returns,
        lookback=lookback,
        rebalance_every=rebalance_every,
        weight_fn=lambda window: np.ones(window.shape[1]) / window.shape[1],
    )
    strategy_paths = simulate_scheduled_weights(
        returns, sharpe_targets, commission_bps, slippage_bps
    )
    benchmark_paths = simulate_scheduled_weights(
        returns, equal_targets, commission_bps, slippage_bps
    )
    strategy = _slice_from_first_trade(strategy_paths)
    benchmark = _slice_from_first_trade(benchmark_paths)
    return {
        "lookback": int(lookback),
        "rebalance_every": int(rebalance_every),
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
        "n_rebalances": int(strategy["rebalance"].sum()),
        "strategy": performance_summary(strategy, risk_free_rate),
        "benchmark": performance_summary(benchmark, risk_free_rate),
        "strategy_wealth": strategy["wealth"],
        "benchmark_wealth": benchmark["wealth"],
    }
