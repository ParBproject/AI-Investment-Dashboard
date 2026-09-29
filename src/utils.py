"""
utils.py
========
Shared utility functions and Plotly chart builders for the
AI Investment Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string (e.g. 0.0342 → '3.42%')."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def format_dollar(value: float, decimals: int = 2) -> str:
    """Format a float as a dollar string (e.g. 12345.6 → '$12,345.60')."""
    return f"${value:,.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

def weights_pie_chart(
    weights: np.ndarray,
    labels: list[str],
    title: str = "Optimal Portfolio Weights",
) -> go.Figure:
    """
    Create a Plotly donut chart of portfolio weights.

    Parameters
    ----------
    weights : np.ndarray  Portfolio allocation fractions.
    labels : list[str]    Asset names.
    title : str           Chart title.

    Negative weights (short sales) are drawn as a bar chart. A pie of signed
    weights either drops the shorts or plots their absolute value.
    """
    signed = np.asarray(weights, dtype=float)
    if np.any(signed < -1e-8):
        fig = go.Figure(go.Bar(
            x=labels,
            y=np.round(signed * 100, 2),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=360,
            yaxis_title="Weight (%)",
            showlegend=False,
        )
        return fig

    fig = go.Figure(go.Pie(
        labels=labels,
        values=np.round(np.clip(signed, 0.0, None) * 100, 2),
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        showlegend=False,
    )
    return fig


def correlation_heatmap(
    returns: pd.DataFrame,
    labels: list[str],
    title: str = "Asset Correlation Matrix",
) -> go.Figure:
    """
    Create an annotated Plotly heatmap of pairwise return correlations.

    Parameters
    ----------
    returns : pd.DataFrame  Daily returns.
    labels : list[str]      Column labels.
    title : str             Chart title.
    """
    corr = returns.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
    )
    return fig


def returns_histogram(
    values: np.ndarray,
    title: str = "Return Distribution",
    bins: int = 60,
    color: str = "#636EFA",
) -> go.Figure:
    """
    Plotly histogram with a KDE overlay for a distribution of values.

    Parameters
    ----------
    values : np.ndarray  Data to plot (e.g. simulated portfolio values).
    title : str          Chart title.
    bins : int           Number of histogram bins.
    color : str          Bar fill colour.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=bins,
        marker_color=color,
        opacity=0.75,
        name="Frequency",
    ))

    # A zero-width sample (for example a constant price path) has no density.
    if values.size >= 2 and float(np.std(values)) > 0.0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values, bw_method="scott")
        x_range = np.linspace(values.min(), values.max(), 300)
        y_kde = kde(x_range)
        count, _ = np.histogram(values, bins=bins)
        if y_kde.max() > 0.0 and count.max() > 0:
            scale = count.max() / y_kde.max()
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde * scale,
                mode="lines",
                line=dict(color="white", width=2),
                name="KDE",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        template="plotly_dark",
        height=360,
        showlegend=False,
    )
    return fig


def candlestick_chart(
    prices: pd.DataFrame,
    ticker: str,
    title: str | None = None,
) -> go.Figure:
    """
    Create a Plotly OHLC candlestick chart.

    Parameters
    ----------
    prices : pd.DataFrame  Must have columns: Open, High, Low, Close.
    ticker : str           Ticker symbol for the label.
    title : str, optional  Chart title.
    """
    fig = go.Figure(go.Candlestick(
        x=prices.index,
        open=prices["Open"],
        high=prices["High"],
        low=prices["Low"],
        close=prices["Close"],
        name=ticker,
    ))
    fig.update_layout(
        title=title or f"{ticker} Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=400,
    )
    return fig


def drawdown_chart(
    portfolio_values: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """
    Plot rolling drawdown from peak.

    Parameters
    ----------
    portfolio_values : pd.Series  Cumulative portfolio value series.
    title : str  Chart title.
    """
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy",
        line=dict(color="red", width=1),
        name="Drawdown (%)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=300,
    )
    return fig
