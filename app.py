"""
AI-Driven Investment Dashboard
================================
A comprehensive Streamlit web application combining portfolio optimization,
Black-Scholes option pricing, Monte Carlo simulations, and AI-driven
"what-if" scenario generation.

⚠️ DISCLAIMER: For educational/simulation purposes only. Not financial advice.
Backtest results do not guarantee future performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy libs loaded only when needed) ──────────────────────────
@st.cache_resource
def load_heavy_libs():
    import yfinance as yf
    from scipy.optimize import minimize
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture
    return yf, minimize, norm, GaussianMixture

# ── Module imports ─────────────────────────────────────────────────────────────
from src.data_loader import fetch_price_data, load_csv_prices
from src.optimizer import (
    compute_portfolio_metrics,
    efficient_frontier,
    max_sharpe_weights,
)
from src.models import (
    black_scholes,
    monte_carlo_paths,
    var_cvar,
    gmm_scenario_returns,
)
from src.utils import (
    format_pct,
    weights_pie_chart,
    correlation_heatmap,
    returns_histogram,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://via.placeholder.com/260x60?text=AI+Investment+Dashboard",
             use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")

    input_mode = st.radio("Data Source", ["Ticker Symbols", "Upload CSV"],
                          horizontal=True)

    if input_mode == "Ticker Symbols":
        tickers_raw = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, TSLA",
        )
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV (Date index, ticker columns)",
                                    type=["csv"])
        tickers = []

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date",
                                   value=pd.Timestamp("2021-01-01"))
    with col2:
        end_date = st.date_input("End Date",
                                 value=pd.Timestamp("2024-12-31"))

    risk_free_rate = st.number_input("Risk-Free Rate (annual, %)", value=4.5,
                                     step=0.1) / 100
    allow_short = st.checkbox("Allow Short Selling", value=False)
    n_portfolios = st.slider("Frontier Portfolios", 100, 2000, 500, step=100)

    st.markdown("---")
    st.subheader("🎯 What-If Scenario")
    shock_pct = st.slider("Market Shock (%)", -50, 50, -20)
    rate_hike = st.slider("Interest Rate Change (bps)", -200, 200, 100)
    n_mc_paths = st.slider("Monte Carlo Paths", 500, 5000, 1000, step=500)

    st.markdown("---")
    st.subheader("🔮 Options Pricing")
    opt_spot = st.number_input("Spot Price (S)", value=150.0)
    opt_strike = st.number_input("Strike Price (K)", value=155.0)
    opt_maturity = st.number_input("Maturity (years, T)", value=0.5,
                                   step=0.25)
    opt_vol = st.number_input("Volatility (σ, %)", value=25.0) / 100
    opt_r = st.number_input("Risk-Free Rate for Option (%)",
                            value=4.5) / 100

    run_btn = st.button("🚀 Run Analysis", type="primary",
                        use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📈 AI-Driven Investment Dashboard")
st.caption(
    "Portfolio Optimization · Monte Carlo · Black-Scholes · "
    "AI Scenario Generation  |  ⚠️ Educational use only"
)

# Tabs
tabs = st.tabs([
    "🏠 Overview",
    "📊 Portfolio Optimizer",
    "🎲 Monte Carlo & Risk",
    "🔮 Options Pricing",
    "🤖 AI What-If Scenarios",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    return fetch_price_data(tickers, str(start), str(end))


if not run_btn:
    with tabs[0]:
        st.info("👈 Configure your portfolio in the sidebar, then click **Run Analysis**.")
        st.markdown("""
        ### Features
        | Module | Description |
        |--------|-------------|
        | **Portfolio Optimizer** | Mean-variance optimization, Efficient Frontier, Max-Sharpe weights |
        | **Monte Carlo & Risk** | Simulated price paths, VaR/CVaR, drawdown analysis |
        | **Options Pricing** | Black-Scholes call/put prices, Greeks, payoff diagrams |
        | **AI What-If Scenarios** | GMM-based synthetic return generation, shock scenarios |
        """)
    st.stop()

# ── Load prices ────────────────────────────────────────────────────────────────
with st.spinner("Fetching price data…"):
    if input_mode == "Upload CSV" and uploaded is not None:
        prices = load_csv_prices(uploaded)
        tickers = list(prices.columns)
    elif tickers:
        prices = load_data(tuple(tickers), start_date, end_date)
    else:
        st.error("Please enter tickers or upload a CSV.")
        st.stop()

if prices is None or prices.empty:
    st.error("Could not retrieve data. Check tickers/dates and try again.")
    st.stop()

returns = prices.pct_change().dropna()
n_assets = len(prices.columns)
asset_names = list(prices.columns)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📋 Portfolio Overview")

    # KPI metrics row
    total_returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    annual_vols = returns.std() * np.sqrt(252) * 100
    ann_rets = returns.mean() * 252 * 100

    cols = st.columns(min(n_assets, 5))
    for i, ticker in enumerate(asset_names[:5]):
        with cols[i]:
            st.metric(
                label=ticker,
                value=f"${prices[ticker].iloc[-1]:.2f}",
                delta=f"{total_returns[ticker]:.1f}% total return",
            )

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Normalised price chart
        fig_norm = go.Figure()
        norm_prices = prices / prices.iloc[0] * 100
        for ticker in asset_names:
            fig_norm.add_trace(
                go.Scatter(x=norm_prices.index, y=norm_prices[ticker],
                           name=ticker, mode="lines")
            )
        fig_norm.update_layout(
            title="Normalized Price Performance (Base=100)",
            xaxis_title="Date", yaxis_title="Indexed Price",
            template="plotly_dark", height=380,
        )
        st.plotly_chart(fig_norm, use_container_width=True)

    with col_right:
        # Correlation heatmap
        fig_corr = correlation_heatmap(returns, asset_names)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Summary stats table
    st.subheader("📊 Summary Statistics")
    stats_df = pd.DataFrame({
        "Ticker": asset_names,
        "Ann. Return (%)": ann_rets.round(2).values,
        "Ann. Volatility (%)": annual_vols.round(2).values,
        "Total Return (%)": total_returns.round(2).values,
        "Sharpe (approx)": ((ann_rets - risk_free_rate * 100) /
                            annual_vols).round(3).values,
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PORTFOLIO OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🎯 Mean-Variance Portfolio Optimization")

    with st.spinner("Computing efficient frontier…"):
        frontier_results = efficient_frontier(
            returns, n_portfolios, risk_free_rate, allow_short
        )
        opt_weights, opt_ret, opt_vol, opt_sharpe = max_sharpe_weights(
            returns, risk_free_rate, allow_short
        )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Max Sharpe Ratio", f"{opt_sharpe:.3f}")
    col_b.metric("Expected Annual Return", format_pct(opt_ret))
    col_c.metric("Expected Annual Volatility", format_pct(opt_vol))

    st.markdown("---")
    col_left2, col_right2 = st.columns([3, 2])

    with col_left2:
        # Efficient frontier scatter
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier_results["vols"] * 100,
            y=frontier_results["rets"] * 100,
            mode="markers",
            marker=dict(
                color=frontier_results["sharpes"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                size=4,
                opacity=0.6,
            ),
            name="Random Portfolios",
        ))
        # Highlight max-Sharpe
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol * 100], y=[opt_ret * 100],
            mode="markers+text",
            marker=dict(color="red", size=14, symbol="star"),
            text=["Max Sharpe"],
            textposition="top center",
            name="Optimal Portfolio",
        ))
        fig_ef.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    with col_right2:
        fig_pie = weights_pie_chart(opt_weights, asset_names)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Weights table
    st.subheader("Optimal Weights")
    w_df = pd.DataFrame({
        "Asset": asset_names,
        "Weight (%)": (opt_weights * 100).round(2),
    }).sort_values("Weight (%)", ascending=False)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MONTE CARLO & RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🎲 Monte Carlo Simulation & Risk Metrics")

    # Portfolio returns using optimized weights
    port_returns = returns @ opt_weights

    with st.spinner("Running Monte Carlo…"):
        last_price = 1.0  # normalized portfolio value
        paths = monte_carlo_paths(port_returns, n_mc_paths, 252, last_price)

    var_95, cvar_95 = var_cvar(port_returns, 0.95)
    var_99, cvar_99 = var_cvar(port_returns, 0.99)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95% (daily)", format_pct(var_95))
    c2.metric("CVaR 95% (daily)", format_pct(cvar_95))
    c3.metric("VaR 99% (daily)", format_pct(var_99))
    c4.metric("CVaR 99% (daily)", format_pct(var_99))

    st.markdown("---")
    col_mc1, col_mc2 = st.columns([3, 2])

    with col_mc1:
        # MC paths chart
        fig_mc = go.Figure()
        x_days = list(range(paths.shape[0]))
        for i in range(min(200, n_mc_paths)):
            fig_mc.add_trace(go.Scatter(
                x=x_days, y=paths[:, i],
                mode="lines", line=dict(width=0.5, color="rgba(99,110,250,0.15)"),
                showlegend=False,
            ))
        # Median + percentile bands
        p5 = np.percentile(paths, 5, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p95 = np.percentile(paths, 95, axis=1)
        fig_mc.add_trace(go.Scatter(x=x_days, y=p50, name="Median",
                                    line=dict(color="white", width=2)))
        fig_mc.add_trace(go.Scatter(x=x_days, y=p95, name="95th pct",
                                    line=dict(color="green", width=1.5,
                                              dash="dash")))
        fig_mc.add_trace(go.Scatter(x=x_days, y=p5, name="5th pct",
                                    line=dict(color="red", width=1.5,
                                              dash="dash")))
        fig_mc.update_layout(
            title=f"Monte Carlo Simulation ({n_mc_paths} paths, 1-year horizon)",
            xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_mc2:
        # Final-value distribution
        final_vals = paths[-1, :]
        fig_hist = returns_histogram(final_vals,
                                     title="Distribution of Final Portfolio Value")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Risk metrics table
    st.subheader("📋 Risk Summary")
    risk_data = {
        "Metric": ["Ann. Return", "Ann. Volatility",
                   "VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%",
                   "Max Simulated Gain", "Max Simulated Loss"],
        "Value": [
            format_pct(port_returns.mean() * 252),
            format_pct(port_returns.std() * np.sqrt(252)),
            format_pct(var_95), format_pct(cvar_95),
            format_pct(var_99), format_pct(cvar_99),
            format_pct(final_vals.max() - 1),
            format_pct(final_vals.min() - 1),
        ],
    }
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True,
                 hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — OPTIONS PRICING
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔮 Black-Scholes Option Pricing")

    call_price, put_price, greeks = black_scholes(
        opt_spot, opt_strike, opt_maturity, opt_r, opt_vol
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Call Price", f"${call_price:.4f}")
    c2.metric("Put Price", f"${put_price:.4f}")
    c3.metric("Delta (Call)", f"{greeks['delta_call']:.4f}")
    c4.metric("Gamma", f"{greeks['gamma']:.4f}")
    c5.metric("Theta (Call)", f"{greeks['theta_call']:.4f}")

    st.markdown("---")

    # Payoff diagram
    spot_range = np.linspace(opt_spot * 0.5, opt_spot * 1.5, 300)
    call_payoffs = np.maximum(spot_range - opt_strike, 0) - call_price
    put_payoffs = np.maximum(opt_strike - spot_range, 0) - put_price

    fig_opt = go.Figure()
    fig_opt.add_trace(go.Scatter(x=spot_range, y=call_payoffs, name="Long Call",
                                 line=dict(color="green", width=2)))
    fig_opt.add_trace(go.Scatter(x=spot_range, y=put_payoffs, name="Long Put",
                                 line=dict(color="red", width=2)))
    fig_opt.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_opt.add_vline(x=opt_strike, line_dash="dot", line_color="yellow",
                      annotation_text="Strike")
    fig_opt.update_layout(
        title="Option Payoff at Expiration",
        xaxis_title="Spot Price at Expiry ($)",
        yaxis_title="P&L ($)",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_opt, use_container_width=True)

    # Greeks vs Spot
    vols_range = np.linspace(0.05, 0.80, 100)
    call_prices_vol = [black_scholes(opt_spot, opt_strike, opt_maturity,
                                     opt_r, v)[0] for v in vols_range]
    put_prices_vol = [black_scholes(opt_spot, opt_strike, opt_maturity,
                                    opt_r, v)[1] for v in vols_range]

    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=vols_range * 100, y=call_prices_vol,
                                   name="Call Price", line=dict(color="green")))
    fig_vega.add_trace(go.Scatter(x=vols_range * 100, y=put_prices_vol,
                                   name="Put Price", line=dict(color="red")))
    fig_vega.update_layout(
        title="Option Price vs Implied Volatility",
        xaxis_title="Volatility (%)", yaxis_title="Price ($)",
        template="plotly_dark", height=320,
    )
    st.plotly_chart(fig_vega, use_container_width=True)

    # Greeks table
    st.subheader("Greeks Summary")
    greeks_df = pd.DataFrame([{
        "Call Price": f"${call_price:.4f}",
        "Put Price": f"${put_price:.4f}",
        "Delta (Call)": f"{greeks['delta_call']:.4f}",
        "Delta (Put)": f"{greeks['delta_put']:.4f}",
        "Gamma": f"{greeks['gamma']:.6f}",
        "Theta (Call)": f"{greeks['theta_call']:.4f}",
        "Theta (Put)": f"{greeks['theta_put']:.4f}",
        "Vega": f"{greeks['vega']:.4f}",
        "Rho (Call)": f"{greeks['rho_call']:.4f}",
        "Rho (Put)": f"{greeks['rho_put']:.4f}",
    }])
    st.dataframe(greeks_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AI WHAT-IF SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🤖 AI-Driven What-If Scenario Analysis")
    st.caption(
        "Uses Gaussian Mixture Models (GMM) to learn the return distribution "
        "and generate synthetic price paths with user-defined shocks."
    )

    with st.spinner("Fitting GMM and generating scenarios…"):
        scenario_paths, scenario_normal = gmm_scenario_returns(
            port_returns,
            n_paths=n_mc_paths,
            horizon=252,
            shock_pct=shock_pct / 100,
            rate_shock=rate_hike / 10000,
            risk_free_rate=risk_free_rate,
        )

    # Comparison chart
    x_days = list(range(scenario_paths.shape[0]))
    fig_scen = make_subplots(rows=1, cols=2,
                              subplot_titles=["Normal Conditions",
                                              f"Shock: {shock_pct:+}% market, "
                                              f"{rate_hike:+}bps rate"])

    for title, data, col in [("Normal", scenario_normal, 1),
                              ("Shocked", scenario_paths, 2)]:
        for i in range(min(100, n_mc_paths)):
            fig_scen.add_trace(
                go.Scatter(x=x_days, y=data[:, i], mode="lines",
                           line=dict(width=0.5,
                                     color=("rgba(99,110,250,0.1)" if col == 1
                                            else "rgba(255,80,80,0.1)")),
                           showlegend=False),
                row=1, col=col,
            )
        p50 = np.percentile(data, 50, axis=1)
        p5 = np.percentile(data, 5, axis=1)
        p95 = np.percentile(data, 95, axis=1)
        color = "blue" if col == 1 else "red"
        fig_scen.add_trace(go.Scatter(x=x_days, y=p50, name=f"{title} Median",
                                       line=dict(color=color, width=2)),
                           row=1, col=col)
        fig_scen.add_trace(go.Scatter(x=x_days, y=p95, name=f"{title} 95th",
                                       line=dict(color=color, width=1,
                                                 dash="dash")),
                           row=1, col=col)
        fig_scen.add_trace(go.Scatter(x=x_days, y=p5, name=f"{title} 5th",
                                       line=dict(color=color, width=1,
                                                 dash="dot")),
                           row=1, col=col)

    fig_scen.update_layout(template="plotly_dark", height=460,
                            title="Scenario Comparison: Normal vs Shocked")
    st.plotly_chart(fig_scen, use_container_width=True)

    # Before/after metrics table
    st.subheader("📋 Before vs After Shock Metrics")
    final_normal = scenario_normal[-1, :]
    final_shocked = scenario_paths[-1, :]

    def scenario_metrics(vals, label):
        return {
            "Scenario": label,
            "Median Return (%)": f"{(np.median(vals) - 1) * 100:.2f}%",
            "Mean Return (%)": f"{(np.mean(vals) - 1) * 100:.2f}%",
            "Prob. of Loss (%)": f"{(vals < 1).mean() * 100:.1f}%",
            "5th Pct Value ($)": f"${np.percentile(vals, 5):.3f}",
            "95th Pct Value ($)": f"${np.percentile(vals, 95):.3f}",
        }

    metrics_df = pd.DataFrame([
        scenario_metrics(final_normal, "Normal"),
        scenario_metrics(final_shocked, f"Shock {shock_pct:+}% / {rate_hike:+}bps"),
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Distribution comparison
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=(final_normal - 1) * 100, name="Normal",
                                     opacity=0.6, nbinsx=60,
                                     marker_color="blue"))
    fig_dist.add_trace(go.Histogram(x=(final_shocked - 1) * 100, name="Shocked",
                                     opacity=0.6, nbinsx=60,
                                     marker_color="red"))
    fig_dist.update_layout(
        barmode="overlay",
        title="Final Portfolio Return Distribution: Normal vs Shocked",
        xaxis_title="1-Year Return (%)", yaxis_title="Frequency",
        template="plotly_dark", height=360,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This dashboard is for educational and simulation "
    "purposes only. It does not constitute financial advice. Past performance "
    "and simulated results do not guarantee future returns. "
    "Always consult a qualified financial professional before investing."
)
