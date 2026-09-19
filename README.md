# Quantitative Investment Research Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![Derivatives](https://img.shields.io/badge/Derivatives-BS%20%7C%20CRR%20%7C%20Monte%20Carlo-0F766E)](src/derivatives.py)
[![Risk](https://img.shields.io/badge/Risk-VaR%20%7C%20ES%20%7C%20Scenarios-2563EB)](src/models.py)
[![CI](https://github.com/ParBproject/AI-Investment-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/AI-Investment-Dashboard/actions/workflows/ci.yml)

A polished quantitative-finance research application combining **real market data, portfolio optimization, risk analytics, multi-method derivatives pricing, implied volatility, Monte Carlo simulation, and regime-aware stress scenarios**.

The project is designed to demonstrate both **Quantitative Specialist** and **Data Analyst** skills: data ingestion, statistical analysis, optimization, simulation, numerical validation, interactive reporting, and explicit model assumptions.

## Employer snapshot

| Capability | Evidence |
|---|---|
| Real financial data | Historical adjusted prices via Yahoo Finance or CSV |
| Portfolio analytics | Maximum-Sharpe and minimum-variance allocation |
| Dependence analysis | Correlations, covariance-aware volatility, diversification |
| Market risk | Historical VaR, Expected Shortfall, drawdown |
| Monte Carlo | Reproducible portfolio path simulation |
| Derivatives | Black–Scholes, Greeks, CRR binomial tree, risk-neutral Monte Carlo |
| Implied volatility | Bracketed root-solving with no-arbitrage validation |
| Model validation | Cross-pricer differences and put–call parity residual |
| Scenario analysis | Gaussian Mixture Model regime simulation with explicit shocks |
| Reporting | Professional Streamlit research workbench and Plotly visualizations |
| Engineering | Modular Python, regression tests, CI on Python 3.10/3.12 |

## Research workbenches

The redesigned application is divided into six focused research areas:

**Market Overview**  
Normalized price performance, descriptive statistics, annualized risk/return, and correlation structure.

**Portfolio Lab**  
Maximum-Sharpe and minimum-variance portfolios, opportunity-set simulation, signed allocation views, and concentration analysis.

**Risk Lab**  
Historical VaR / Expected Shortfall, maximum drawdown, seeded Monte Carlo paths, terminal distributions, and probability-of-loss diagnostics.

**Derivatives Lab**  
Black–Scholes, Greeks, Cox–Ross–Rubinstein binomial pricing, risk-neutral Monte Carlo, implied volatility, put–call parity, and a two-dimensional option-price sensitivity surface.

**Regime Scenarios**  
Gaussian-mixture return regimes, explicit market shocks, rate-driven drift changes, terminal distributions, and downside probability.

**Methodology**  
Model assumptions, validation logic, and limitations surfaced directly in the product.

## Why the derivatives module is different

A basic portfolio dashboard often shows one Black–Scholes number.

This project cross-checks the same European option using three independent methods:

```text
Black–Scholes closed form
        ↓ compare
Cox–Ross–Rubinstein binomial tree
        ↓ compare
Risk-neutral Monte Carlo
```

The application also reports:

- Monte Carlo standard error;
- tree-vs-Black–Scholes pricing difference;
- Monte-Carlo-vs-Black–Scholes pricing difference;
- put–call parity residual;
- implied volatility from an observed option price.

That makes model consistency visible instead of treating one formula as ground truth.

## Derivatives pricing workflow

```text
Spot / strike / maturity / rate / volatility
        ↓
Black–Scholes + Greeks
        ↓
CRR binomial tree
        ↓
Risk-neutral Monte Carlo
        ↓
Cross-method price comparison
        ↓
Put–call parity check
        ↓
Observed market price (optional)
        ↓
Implied volatility
        ↓
Spot × volatility sensitivity surface
```

## Portfolio research workflow

```text
Historical adjusted prices
        ↓
Daily returns
        ↓
Mean + covariance estimates
        ↓
Constrained optimization
        ↓
Maximum Sharpe / Minimum Variance
        ↓
Portfolio return series
        ↓
VaR / Expected Shortfall / Drawdown
        ↓
Monte Carlo + regime scenarios
```

## Application outputs

Existing repository screenshots show the major analysis modules:

### Market overview

![Investment dashboard overview](assets/screenshots/01_overview.png)

### Portfolio opportunity set

![Portfolio efficient frontier](assets/screenshots/02_efficient_frontier.png)

### Monte Carlo risk analysis

![Monte Carlo portfolio simulation](assets/screenshots/03_monte_carlo.png)

### Options analytics

![Options analysis](assets/screenshots/04_options_pricing.png)

### Scenario analysis

![Investment scenarios](assets/screenshots/05_what_if_scenarios.png)

The current application has been redesigned into a unified light-mode research interface; screenshots should be regenerated after deployment to reflect the latest visual system.

## Option-pricing methods

### Black–Scholes

The shared model computes European call and put values plus:

- Delta;
- Gamma;
- Vega;
- Theta;
- Rho.

### CRR binomial tree

`src/derivatives.py` implements a recombining Cox–Ross–Rubinstein tree with configurable step count.

It supports both European and American early-exercise logic.

### Risk-neutral Monte Carlo

European terminal values are simulated under risk-neutral GBM using antithetic random variates.

The function returns both:

- estimated option price;
- Monte Carlo standard error.

### Implied volatility

The implied-volatility solver uses a bracketed Brent method and validates European no-arbitrage bounds before solving.

## Scenario modeling

The scenario engine fits a Gaussian Mixture Model to historical optimized-portfolio returns.

This allows more than one return regime to influence simulated outcomes.

The user can apply:

- a one-time market shock;
- an interest-rate shock expressed in basis points.

The dashboard reports baseline vs shocked medians and probability of ending below starting capital.

These are conditional scenario analyses, not event-probability forecasts.

## Repository architecture

```text
AI-Investment-Dashboard/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── derivatives.py
│   ├── models.py
│   ├── optimizer.py
│   └── utils.py
├── tests/
│   ├── test_derivatives.py
│   └── test_optimizer.py
├── notebooks/
│   └── 01_model_exploration.ipynb
├── assets/screenshots/
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── METHODOLOGY.md
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/ParBproject/AI-Investment-Dashboard.git
cd AI-Investment-Dashboard

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

Run the regression suite:

```bash
python -m pytest -q
```

## Quantitative validation

The automated tests include checks that:

- optimizer weights remain bounded and fully invested;
- short-enabled simulations do not create accidental extreme leverage;
- CRR binomial prices converge near Black–Scholes European prices;
- Black–Scholes prices satisfy put–call parity;
- implied volatility recovers a known input volatility;
- risk-neutral Monte Carlo prices are statistically consistent with Black–Scholes;
- American puts are not valued below equivalent European puts;
- option-pricing comparison outputs remain positive and finite.

GitHub Actions runs linting, compilation, the full regression suite, and import checks on Python **3.10 and 3.12**.

## Skills demonstrated

**Quantitative finance:** portfolio optimization, covariance risk, VaR, Expected Shortfall, Monte Carlo, Black–Scholes, Greeks, binomial trees, implied volatility, regime scenarios.

**Data analysis:** pandas, NumPy, real market-data transformation, correlations, descriptive statistics, scenario comparison, interactive reporting.

**Numerical methods:** SciPy optimization, root solving, backward induction, stochastic simulation, antithetic variates.

**Software engineering:** modular Python, dataclasses, deterministic seeds, unit tests, CI/CD, explicit validation and error handling.

**Communication:** decision-oriented dashboard design, methodology documentation, visible model assumptions and limitations.

## Methodology

See **[METHODOLOGY.md](METHODOLOGY.md)** for formulas, modeling assumptions, validation logic, and production limitations.

## Limitations

- Historical return estimates are sample-dependent and regime-dependent.
- Mean–variance optimization can be sensitive to estimation error.
- Core Black–Scholes assumes no dividends and constant volatility/rates.
- Monte Carlo and GMM outputs depend on distributional assumptions.
- Scenarios do not estimate the probability that a shock will occur.
- Transaction costs, liquidity constraints, taxes, market impact, and every regulatory constraint are not fully modeled.

## Responsible use

Educational quantitative-finance project only. Historical, simulated, and theoretical results do not constitute investment advice.
