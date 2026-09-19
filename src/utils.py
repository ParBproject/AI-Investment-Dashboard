"""Shared formatting and professional Plotly chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY = "#0F172A"
TEAL = "#0F766E"
CYAN = "#38BDF8"
ROSE = "#E11D48"
SLATE = "#64748B"
GRID = "#E2E8F0"
PAPER = "#FFFFFF"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_dollar(value: float, decimals: int = 2) -> str:
    """Format a float as a dollar string."""
    return "$" + f"{value:,.{decimals}f}"


def _base_layout(title: str, height: int = 380) -> dict:
    return {
        "title": {
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": NAVY},
        },
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": "Inter, Arial, sans-serif", "color": NAVY},
        "margin": {"l": 55, "r": 30, "t": 65, "b": 55},
    }


def weights_pie_chart(
    weights: np.ndarray,
    labels: list[str],
    title: str = "Optimal Portfolio Weights",
) -> go.Figure:
    """Create a clean donut chart of portfolio weights."""
    values = np.asarray(weights, dtype=float)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=np.abs(values),
            hole=0.58,
            textinfo="label+percent",
            textfont={"size": 12},
            marker={
                "colors": [
                    TEAL,
                    "#14B8A6",
                    CYAN,
                    "#2563EB",
                    "#7C3AED",
                    ROSE,
                    "#F59E0B",
                    "#84CC16",
                ][: len(labels)]
            },
            hovertemplate="%{label}: %{value:.2%}<extra></extra>",
        )
    )
    layout = _base_layout(title, height=390)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def correlation_heatmap(
    returns: pd.DataFrame,
    labels: list[str],
    title: str = "Asset Correlation Matrix",
) -> go.Figure:
    """Create an annotated correlation heatmap."""
    corr = returns.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#DBEAFE"],
                [0.5, "#F8FAFC"],
                [1.0, "#0F766E"],
            ],
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            colorbar={"title": "ρ", "thickness": 12, "outlinewidth": 0},
            hovertemplate="%{y} / %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    layout = _base_layout(title, height=390)
    layout["xaxis"] = {"side": "bottom"}
    layout["yaxis"] = {"autorange": "reversed"}
    fig.update_layout(**layout)
    return fig


def returns_histogram(
    values: np.ndarray,
    title: str = "Return Distribution",
    bins: int = 60,
    color: str = TEAL,
) -> go.Figure:
    """Histogram with a kernel-density overlay."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        raise ValueError("values must contain at least two finite observations")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=bins,
            marker={"color": color},
            opacity=0.72,
            name="Frequency",
        )
    )

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(clean, bw_method="scott")
    x_range = np.linspace(clean.min(), clean.max(), 300)
    y_kde = kde(x_range)
    count, _ = np.histogram(clean, bins=bins)
    scale = count.max() / y_kde.max()

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde * scale,
            mode="lines",
            line={"color": NAVY, "width": 2.2},
            name="Density",
        )
    )

    layout = _base_layout(title, height=380)
    layout["xaxis"] = {"title": "Value", "gridcolor": GRID}
    layout["yaxis"] = {"title": "Count", "gridcolor": GRID}
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def candlestick_chart(
    prices: pd.DataFrame,
    ticker: str,
    title: str | None = None,
) -> go.Figure:
    """Create a Plotly OHLC candlestick chart."""
    fig = go.Figure(
        go.Candlestick(
            x=prices.index,
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=prices["Close"],
            increasing_line_color=TEAL,
            decreasing_line_color=ROSE,
            name=ticker,
        )
    )
    layout = _base_layout(title or f"{ticker} Price", height=420)
    layout["xaxis"] = {"title": "", "rangeslider": {"visible": False}}
    layout["yaxis"] = {"title": "Price ($)", "gridcolor": GRID}
    fig.update_layout(**layout)
    return fig


def drawdown_chart(
    portfolio_values: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """Plot rolling drawdown from peak."""
    rolling_max = portfolio_values.cummax()
    drawdown = portfolio_values / rolling_max - 1.0

    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(225,29,72,0.12)",
            line={"color": ROSE, "width": 1.8},
            name="Drawdown",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>",
        )
    )
    layout = _base_layout(title, height=320)
    layout["xaxis"] = {"title": "", "gridcolor": GRID}
    layout["yaxis"] = {"title": "Drawdown", "tickformat": ".1%", "gridcolor": GRID}
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig
