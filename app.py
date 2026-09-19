"""Professional quantitative investment research dashboard."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import fetch_price_data, load_csv_prices
from src.derivatives import (
    black_scholes_price,
    compare_option_pricers,
    implied_volatility,
    put_call_parity_gap,
)
from src.models import (
    black_scholes,
    gmm_scenario_returns,
    max_drawdown,
    monte_carlo_paths,
    var_cvar,
)
from src.optimizer import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
)
from src.utils import (
    correlation_heatmap,
    format_dollar,
    format_pct,
    returns_histogram,
    weights_pie_chart,
)


NAVY = "#0F172A"
TEAL = "#0F766E"
CYAN = "#38BDF8"
ROSE = "#E11D48"
SLATE = "#64748B"
GRID = "#E2E8F0"


st.set_page_config(
    page_title="Quant Investment Research Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F8FAFC; }
    .block-container {
        max-width: 1480px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2.05rem 2.25rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 12%, rgba(56,189,248,.22), transparent 31%),
            linear-gradient(135deg, #0F172A 0%, #172554 58%, #0F766E 125%);
        color: white;
        box-shadow: 0 18px 48px rgba(15,23,42,.15);
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        font-size: .78rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #99F6E4;
        font-weight: 750;
        margin-bottom: .55rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        line-height: 1.08;
        letter-spacing: -.035em;
    }
    .hero p {
        margin: .8rem 0 0;
        max-width: 900px;
        color: #DCE7F4;
        font-size: 1rem;
        line-height: 1.65;
    }
    .signal-card {
        padding: 1rem 1.08rem;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        min-height: 104px;
        box-shadow: 0 5px 18px rgba(15,23,42,.04);
    }
    .signal-label {
        color: #64748B;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 750;
    }
    .signal-value {
        color: #0F172A;
        font-size: 1.45rem;
        font-weight: 760;
        margin-top: .28rem;
    }
    .signal-note {
        color: #64748B;
        font-size: .79rem;
        margin-top: .18rem;
    }
    .method-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: .8rem;
        line-height: 1.55;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: .85rem 1rem;
        box-shadow: 0 4px 14px rgba(15,23,42,.035);
    }
    section[data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def signal_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="signal-label">{label}</div>
          <div class="signal-value">{value}</div>
          <div class="signal-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def line_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    y_title: str,
    height: int = 420,
) -> go.Figure:
    colors = [NAVY, TEAL, CYAN, "#2563EB", "#7C3AED", ROSE, "#F59E0B"]
    fig = go.Figure()
    for index, column in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=str(column),
                line={"width": 2.3, "color": colors[index % len(colors)]},
            )
        )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, Arial, sans-serif", "color": NAVY},
        xaxis={"title": "", "gridcolor": GRID},
        yaxis={"title": y_title, "gridcolor": GRID},
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 55, "r": 25, "t": 65, "b": 65},
    )
    return fig


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Quantitative Investment Research</div>
      <h1>Investment Research Lab</h1>
      <p>
        Real-market portfolio analytics, optimization, tail-risk simulation,
        multi-method derivatives pricing, implied volatility, and regime-aware
        stress scenarios in one reproducible research interface.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Research configuration")
    input_mode = st.radio(
        "Data source",
        ["Ticker symbols", "Upload CSV"],
        horizontal=True,
    )

    uploaded = None
    if input_mode == "Ticker symbols":
        tickers_raw = st.text_input(
            "Asset universe",
            value="AAPL, MSFT, GOOGL, AMZN, SPY",
        )
        tickers = [
            ticker.strip().upper()
            for ticker in tickers_raw.split(",")
            if ticker.strip()
        ]
    else:
        uploaded = st.file_uploader(
            "Upload price CSV",
            type=["csv"],
            help="First column: date. Remaining columns: adjusted prices.",
        )
        tickers = []

    start_date = st.date_input("History start", value=date(2021, 1, 1))
    end_date = st.date_input("History end", value=date.today())

    st.divider()
    st.markdown("#### Portfolio")
    risk_free_rate = st.slider(
        "Risk-free rate",
        0.0,
        8.0,
        4.0,
        0.25,
    ) / 100.0
    allow_short = st.checkbox("Allow bounded short selling", value=False)
    n_portfolios = st.slider(
        "Opportunity-set simulations",
        250,
        5000,
        1250,
        250,
    )

    st.divider()
    st.markdown("#### Risk simulation")
    n_mc_paths = st.slider("Monte Carlo paths", 1000, 10000, 3000, 500)
    mc_horizon = st.slider("Risk horizon (trading days)", 63, 504, 252, 21)

    st.divider()
    st.markdown("#### Derivatives")
    option_type = st.selectbox("Option type", ["call", "put"])
    opt_spot = st.number_input("Spot price", min_value=0.01, value=150.0, step=1.0)
    opt_strike = st.number_input("Strike price", min_value=0.01, value=155.0, step=1.0)
    opt_maturity = st.number_input(
        "Maturity (years)",
        min_value=0.01,
        value=0.50,
        step=0.05,
    )
    opt_vol = st.number_input(
        "Volatility (%)",
        min_value=0.01,
        value=25.0,
        step=1.0,
    ) / 100.0
    opt_rate = st.number_input(
        "Option risk-free rate (%)",
        value=4.0,
        step=0.25,
    ) / 100.0
    observed_option_price = st.number_input(
        "Observed market option price (0 = skip IV)",
        min_value=0.0,
        value=0.0,
        step=0.25,
    )
    binomial_steps = st.slider("Binomial tree steps", 50, 1000, 400, 50)
    option_mc_paths = st.slider(
        "Option Monte Carlo paths",
        5000,
        150000,
        50000,
        5000,
    )

    st.divider()
    st.markdown("#### Regime scenario")
    shock_pct = st.slider("One-time market shock", -50, 20, -20) / 100.0
    rate_hike_bps = st.slider("Rate shock (bps)", -200, 300, 100)
    scenario_paths = st.slider("Scenario paths", 500, 5000, 1500, 500)

    run_btn = st.button(
        "Run research suite",
        type="primary",
        use_container_width=True,
    )


if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        signal_card("Portfolio", "Mean–variance", "Max Sharpe + minimum variance")
    with c2:
        signal_card("Risk", "VaR + ES", "Monte Carlo and drawdown diagnostics")
    with c3:
        signal_card("Derivatives", "3 pricers", "Black–Scholes, tree, Monte Carlo")
    with c4:
        signal_card("Scenarios", "GMM regimes", "Baseline and shocked distributions")

    st.markdown("### Research modules")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="method-box">
            <b>Portfolio analytics</b><br><br>
            Historical returns, dependence structure, constrained optimization,
            opportunity-set simulation, and risk-adjusted allocation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="method-box">
            <b>Derivatives workbench</b><br><br>
            Black–Scholes Greeks, implied volatility, CRR binomial trees,
            risk-neutral Monte Carlo, parity checks, and pricing sensitivity.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="method-box">
            <b>Scenario & risk lab</b><br><br>
            Historical VaR / Expected Shortfall, stochastic portfolio paths,
            regime-mixture scenarios, and explicit market/rate shocks.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


if start_date >= end_date:
    st.error("History start must be earlier than history end.")
    st.stop()

with st.spinner("Loading market data and running quantitative models..."):
    if input_mode == "Upload CSV":
        if uploaded is None:
            st.error("Upload a CSV file before running the research suite.")
            st.stop()
        prices = load_csv_prices(uploaded)
        tickers = list(prices.columns)
    else:
        if not tickers:
            st.error("Enter at least one ticker.")
            st.stop()
        prices = fetch_price_data(
            tuple(tickers),
            str(start_date),
            str(end_date),
        )

if prices is None or prices.empty or prices.shape[1] == 0:
    st.error("No usable price history was returned.")
    st.stop()

returns = prices.pct_change(fill_method=None).dropna()
if len(returns) < 30:
    st.error("At least 30 complete return observations are required.")
    st.stop()

asset_names = list(prices.columns)
n_assets = len(asset_names)

frontier = efficient_frontier(
    returns,
    n_portfolios=n_portfolios,
    risk_free_rate=risk_free_rate,
    allow_short=allow_short,
    random_state=42,
)
opt_weights, opt_ret, opt_vol, opt_sharpe = max_sharpe_weights(
    returns,
    risk_free_rate=risk_free_rate,
    allow_short=allow_short,
    random_state=42,
)
min_weights, min_ret, min_vol = min_variance_weights(
    returns,
    allow_short=allow_short,
)
min_sharpe = (
    (min_ret - risk_free_rate) / min_vol
    if min_vol > 0
    else 0.0
)

portfolio_returns = returns @ opt_weights
portfolio_wealth = (1.0 + portfolio_returns).cumprod()
portfolio_drawdown = max_drawdown(portfolio_wealth)
var_95, es_95 = var_cvar(portfolio_returns, 0.95)
var_99, es_99 = var_cvar(portfolio_returns, 0.99)

paths = monte_carlo_paths(
    portfolio_returns,
    n_paths=n_mc_paths,
    horizon=mc_horizon,
    initial_value=1.0,
    seed=42,
)

call_price, put_price, greeks = black_scholes(
    opt_spot,
    opt_strike,
    opt_maturity,
    opt_rate,
    opt_vol,
)
pricing = compare_option_pricers(
    opt_spot,
    opt_strike,
    opt_maturity,
    opt_rate,
    opt_vol,
    option_type=option_type,
    binomial_steps=binomial_steps,
    monte_carlo_paths=option_mc_paths,
    seed=42,
)
parity_gap = put_call_parity_gap(
    opt_spot,
    opt_strike,
    opt_maturity,
    opt_rate,
    call_price,
    put_price,
)

iv_result = None
if observed_option_price > 0:
    try:
        iv_result = implied_volatility(
            observed_option_price,
            opt_spot,
            opt_strike,
            opt_maturity,
            opt_rate,
            option_type=option_type,
        )
    except ValueError as exc:
        st.sidebar.warning(f"Implied volatility unavailable: {exc}")

shocked_paths, normal_paths = gmm_scenario_returns(
    portfolio_returns,
    n_paths=scenario_paths,
    horizon=252,
    shock_pct=shock_pct,
    rate_shock=rate_hike_bps / 10_000.0,
    risk_free_rate=risk_free_rate,
    n_components=min(3, max(1, len(portfolio_returns) // 30)),
    seed=42,
)


h1, h2, h3, h4 = st.columns(4)
with h1:
    signal_card("Assets", str(n_assets), " / ".join(asset_names[:4]) + ("…" if n_assets > 4 else ""))
with h2:
    signal_card("Observations", f"{len(returns):,}", "Complete daily return rows")
with h3:
    signal_card("Max-Sharpe", f"{opt_sharpe:.2f}", f"{format_pct(opt_ret)} expected return")
with h4:
    signal_card("95% Expected Shortfall", format_pct(es_95), "Historical daily tail loss")

st.write("")

overview_tab, portfolio_tab, risk_tab, derivatives_tab, scenario_tab, methodology_tab = st.tabs(
    [
        "Market Overview",
        "Portfolio Lab",
        "Risk Lab",
        "Derivatives Lab",
        "Regime Scenarios",
        "Methodology",
    ]
)


with overview_tab:
    normalized = prices / prices.iloc[0] * 100.0
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            line_chart(
                normalized,
                title="Normalized Market Performance",
                y_title="Indexed value (100 = start)",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            correlation_heatmap(returns, asset_names),
            use_container_width=True,
        )

    annual_returns = returns.mean() * 252
    annual_vols = returns.std() * np.sqrt(252)
    total_returns = prices.iloc[-1] / prices.iloc[0] - 1.0
    stats = pd.DataFrame(
        {
            "Asset": asset_names,
            "Annualized Return": annual_returns.values,
            "Annualized Volatility": annual_vols.values,
            "Total Return": total_returns.values,
            "Approx. Sharpe": (
                (annual_returns - risk_free_rate) / annual_vols.replace(0, np.nan)
            ).values,
        }
    )
    st.markdown("#### Market statistics")
    st.dataframe(
        stats.style.format(
            {
                "Annualized Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Total Return": "{:.2%}",
                "Approx. Sharpe": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


with portfolio_tab:
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Max-Sharpe return", format_pct(opt_ret))
    p2.metric("Max-Sharpe volatility", format_pct(opt_vol))
    p3.metric("Max-Sharpe ratio", f"{opt_sharpe:.2f}")
    p4.metric("Min-var return", format_pct(min_ret))
    p5.metric("Min-var volatility", format_pct(min_vol))
    p6.metric("Min-var Sharpe", f"{min_sharpe:.2f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["vols"],
            y=frontier["rets"],
            mode="markers",
            marker={
                "size": 5,
                "color": frontier["sharpes"],
                "colorscale": "Tealgrn",
                "opacity": 0.42,
                "colorbar": {"title": "Sharpe", "thickness": 12},
            },
            name="Simulated portfolios",
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode="markers",
            marker={"size": 19, "color": ROSE, "symbol": "star"},
            name="Maximum Sharpe",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_vol],
            y=[min_ret],
            mode="markers",
            marker={"size": 15, "color": TEAL, "symbol": "diamond"},
            name="Minimum variance",
        )
    )
    fig.update_layout(
        title={"text": "Portfolio Opportunity Set", "x": 0.02},
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, Arial, sans-serif", "color": NAVY},
        xaxis={"title": "Annualized volatility", "tickformat": ".1%", "gridcolor": GRID},
        yaxis={"title": "Annualized return", "tickformat": ".1%", "gridcolor": GRID},
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 55, "r": 30, "t": 65, "b": 70},
    )

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if allow_short:
            weight_frame = pd.DataFrame(
                {
                    "Asset": asset_names,
                    "Weight": opt_weights,
                }
            ).sort_values("Weight")
            weight_fig = go.Figure(
                go.Bar(
                    x=weight_frame["Weight"],
                    y=weight_frame["Asset"],
                    orientation="h",
                    marker_color=[
                        ROSE if value < 0 else TEAL
                        for value in weight_frame["Weight"]
                    ],
                    text=[f"{value:.1%}" for value in weight_frame["Weight"]],
                    textposition="outside",
                )
            )
            weight_fig.update_layout(
                title={"text": "Signed Max-Sharpe Weights", "x": 0.02},
                height=390,
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis={"tickformat": ".0%", "gridcolor": GRID},
                margin={"l": 40, "r": 30, "t": 65, "b": 40},
            )
            st.plotly_chart(weight_fig, use_container_width=True)
        else:
            st.plotly_chart(
                weights_pie_chart(opt_weights, asset_names),
                use_container_width=True,
            )

    allocation = pd.DataFrame(
        {
            "Asset": asset_names,
            "Maximum Sharpe": opt_weights,
            "Minimum Variance": min_weights,
        }
    ).sort_values("Maximum Sharpe", ascending=False)
    st.dataframe(
        allocation.style.format(
            {
                "Maximum Sharpe": "{:.2%}",
                "Minimum Variance": "{:.2%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


with risk_tab:
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("95% VaR", format_pct(var_95))
    r2.metric("95% Expected Shortfall", format_pct(es_95))
    r3.metric("99% VaR", format_pct(var_99))
    r4.metric("99% Expected Shortfall", format_pct(es_99))
    r5.metric("Historical max drawdown", format_pct(portfolio_drawdown))

    sample_count = min(180, paths.shape[1])
    fig_paths = go.Figure()
    x_days = np.arange(paths.shape[0])
    for index in range(sample_count):
        fig_paths.add_trace(
            go.Scatter(
                x=x_days,
                y=paths[:, index],
                mode="lines",
                line={"width": 0.55, "color": "rgba(15,118,110,0.08)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    p05 = np.percentile(paths, 5, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    fig_paths.add_trace(
        go.Scatter(
            x=x_days,
            y=p50,
            name="Median",
            line={"color": NAVY, "width": 2.6},
        )
    )
    fig_paths.add_trace(
        go.Scatter(
            x=x_days,
            y=p95,
            name="95th percentile",
            line={"color": TEAL, "width": 1.8, "dash": "dash"},
        )
    )
    fig_paths.add_trace(
        go.Scatter(
            x=x_days,
            y=p05,
            name="5th percentile",
            line={"color": ROSE, "width": 1.8, "dash": "dash"},
        )
    )
    fig_paths.update_layout(
        title={"text": "Portfolio Monte Carlo Paths", "x": 0.02},
        height=470,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"title": "Trading day", "gridcolor": GRID},
        yaxis={"title": "Portfolio value", "gridcolor": GRID},
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 55, "r": 30, "t": 65, "b": 70},
    )

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig_paths, use_container_width=True)
    with right:
        st.plotly_chart(
            returns_histogram(
                paths[-1],
                title=f"Terminal Value Distribution ({mc_horizon} days)",
            ),
            use_container_width=True,
        )

    terminal = paths[-1]
    risk_summary = pd.DataFrame(
        {
            "Metric": [
                "Mean terminal value",
                "Median terminal value",
                "5th percentile",
                "95th percentile",
                "Probability below starting value",
            ],
            "Value": [
                f"{terminal.mean():.3f}",
                f"{np.median(terminal):.3f}",
                f"{np.percentile(terminal, 5):.3f}",
                f"{np.percentile(terminal, 95):.3f}",
                f"{np.mean(terminal < 1.0):.2%}",
            ],
        }
    )
    st.dataframe(risk_summary, hide_index=True, use_container_width=True)


with derivatives_tab:
    selected_bs = black_scholes_price(
        opt_spot,
        opt_strike,
        opt_maturity,
        opt_rate,
        opt_vol,
        option_type=option_type,
    )

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Black–Scholes", format_dollar(selected_bs))
    d2.metric("Binomial tree", format_dollar(pricing.binomial))
    d3.metric(
        "Risk-neutral Monte Carlo",
        format_dollar(pricing.monte_carlo),
        delta=f"SE {format_dollar(pricing.monte_carlo_standard_error, 3)}",
        delta_color="off",
    )
    d4.metric("Put–call parity gap", format_dollar(parity_gap, 6))
    d5.metric(
        "Implied volatility",
        "—" if iv_result is None else format_pct(iv_result),
    )

    comparison = pd.DataFrame(
        {
            "Method": ["Black–Scholes", "CRR Binomial", "Risk-neutral Monte Carlo"],
            "Price": [
                pricing.black_scholes,
                pricing.binomial,
                pricing.monte_carlo,
            ],
            "Difference vs Black–Scholes": [
                0.0,
                pricing.binomial_difference,
                pricing.monte_carlo_difference,
            ],
        }
    )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### Cross-method pricing")
        st.dataframe(
            comparison.style.format(
                {
                    "Price": format_dollar,
                    "Difference vs Black–Scholes": lambda value: format_dollar(value, 4),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        greek_rows = []
        for key, value in greeks.items():
            if key not in {"d1", "d2"}:
                greek_rows.append(
                    {
                        "Greek": key.replace("_", " ").title(),
                        "Value": value,
                    }
                )
        st.markdown("#### Black–Scholes Greeks")
        st.dataframe(
            pd.DataFrame(greek_rows).style.format({"Value": "{:.5f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        spot_grid = np.linspace(opt_spot * 0.75, opt_spot * 1.25, 21)
        vol_grid = np.linspace(max(0.05, opt_vol * 0.45), opt_vol * 1.75, 21)
        surface = np.empty((len(vol_grid), len(spot_grid)))
        for row_index, volatility in enumerate(vol_grid):
            for col_index, spot in enumerate(spot_grid):
                surface[row_index, col_index] = black_scholes_price(
                    spot,
                    opt_strike,
                    opt_maturity,
                    opt_rate,
                    volatility,
                    option_type=option_type,
                )

        heatmap = go.Figure(
            go.Heatmap(
                z=surface,
                x=spot_grid,
                y=vol_grid,
                colorscale=[
                    [0.0, "#ECFEFF"],
                    [0.45, "#67E8F9"],
                    [1.0, "#0F172A"],
                ],
                colorbar={"title": "Price", "thickness": 12},
                hovertemplate=(
                    "Spot: %{x:.2f}<br>Volatility: %{y:.1%}"
                    "<br>Option price: %{z:.2f}<extra></extra>"
                ),
            )
        )
        heatmap.update_layout(
            title={"text": f"{option_type.title()} Price Sensitivity Surface", "x": 0.02},
            height=540,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis={"title": "Spot price"},
            yaxis={"title": "Volatility", "tickformat": ".0%"},
            margin={"l": 55, "r": 30, "t": 65, "b": 55},
        )
        st.plotly_chart(heatmap, use_container_width=True)

    st.caption(
        "European Black–Scholes, CRR binomial, and risk-neutral Monte Carlo "
        "should converge under consistent assumptions. Differences are shown explicitly."
    )


with scenario_tab:
    normal_terminal = normal_paths[-1]
    shocked_terminal = shocked_paths[-1]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Baseline median", f"{np.median(normal_terminal):.3f}")
    s2.metric("Shocked median", f"{np.median(shocked_terminal):.3f}")
    s3.metric(
        "Median impact",
        f"{np.median(shocked_terminal) / np.median(normal_terminal) - 1.0:.2%}",
    )
    s4.metric(
        "Shocked probability of loss",
        f"{np.mean(shocked_terminal < 1.0):.2%}",
    )

    days = np.arange(normal_paths.shape[0])
    normal_median = np.median(normal_paths, axis=1)
    shocked_median = np.median(shocked_paths, axis=1)
    normal_p10 = np.percentile(normal_paths, 10, axis=1)
    normal_p90 = np.percentile(normal_paths, 90, axis=1)
    shocked_p10 = np.percentile(shocked_paths, 10, axis=1)
    shocked_p90 = np.percentile(shocked_paths, 90, axis=1)

    scenario_fig = go.Figure()
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=normal_p90,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=normal_p10,
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.10)",
            line={"width": 0},
            name="Baseline 10–90%",
            hoverinfo="skip",
        )
    )
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=normal_median,
            line={"color": TEAL, "width": 2.5},
            name="Baseline median",
        )
    )
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=shocked_p90,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=shocked_p10,
            fill="tonexty",
            fillcolor="rgba(225,29,72,0.09)",
            line={"width": 0},
            name="Shocked 10–90%",
            hoverinfo="skip",
        )
    )
    scenario_fig.add_trace(
        go.Scatter(
            x=days,
            y=shocked_median,
            line={"color": ROSE, "width": 2.5},
            name="Shocked median",
        )
    )
    scenario_fig.update_layout(
        title={"text": "Regime-Mixture Scenario Paths", "x": 0.02},
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"title": "Trading day", "gridcolor": GRID},
        yaxis={"title": "Portfolio value", "gridcolor": GRID},
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 55, "r": 30, "t": 65, "b": 70},
    )
    st.plotly_chart(scenario_fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="method-box">
        <b>Scenario specification</b><br><br>
        One-time market shock: <b>{shock_pct:.1%}</b><br>
        Annual drift adjustment from rate shock: <b>{rate_hike_bps} bps</b><br>
        Distribution model: <b>Gaussian Mixture Model</b> fit to historical optimized-portfolio returns.
        </div>
        """,
        unsafe_allow_html=True,
    )


with methodology_tab:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """
            <div class="method-box">
            <b>Portfolio construction</b><br><br>
            Historical daily returns feed mean–variance optimization. Maximum-Sharpe
            and minimum-variance allocations use bounded SLSQP optimization, while
            the opportunity set is simulated reproducibly.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Risk modeling</b><br><br>
            Historical VaR / Expected Shortfall summarize observed downside tails.
            Parametric Monte Carlo simulates portfolio-value uncertainty using
            historical mean and volatility.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="method-box">
            <b>Derivatives validation</b><br><br>
            European option values are cross-checked across Black–Scholes,
            Cox–Ross–Rubinstein binomial trees, and risk-neutral Monte Carlo.
            Put–call parity and implied volatility add internal consistency checks.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Regime scenarios</b><br><br>
            A Gaussian mixture learns multiple return regimes from history.
            Shock scenarios apply an explicit one-time market move and rate-driven
            drift adjustment rather than hiding assumptions inside a narrative.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Limitations")
    st.markdown(
        """
        Historical inputs are regime-dependent. Portfolio optimization can be
        sensitive to estimation error. Monte Carlo and option models rely on
        distributional assumptions. Scenario probabilities are not forecasts,
        and the application does not model every implementation, liquidity,
        tax, or regulatory constraint.
        """
    )

st.caption(
    "Educational quantitative-finance research project. Historical, simulated, "
    "and theoretical results do not constitute investment advice."
)
