# AI-Driven Investment Dashboard

## For a data analyst application

**Do not lead a resume with this title.** The word “AI” here is scenario math (Monte Carlo, options, a frontier), not a language model. Keep it as a finance-dashboard extra behind Advanced Financial Models.

<p align="center"><img src="assets/screenshots/01_overview.png" alt="Investment dashboard overview" width="100%"></p>
<p align="center"><img src="assets/screenshots/02_efficient_frontier.png" alt="Efficient frontier" width="100%"></p>
<p align="center"><img src="assets/screenshots/03_monte_carlo.png" alt="Monte Carlo paths" width="100%"></p>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![Analytics](https://img.shields.io/badge/Analytics-Optimization_%7C_Monte_Carlo_%7C_Options-1f6feb)](#capabilities)

An interactive financial-analytics application combining market-data exploration, portfolio optimization, Monte Carlo simulation, Black–Scholes option pricing, and Gaussian-mixture scenario analysis. There is no language model in this repository.

## Capabilities

| Area | What the application provides |
|---|---|
| Market overview | Normalized prices, returns, correlations, and descriptive statistics |
| Portfolio optimization | Minimum-variance efficient frontier, maximum-Sharpe allocation, random feasible portfolios, and weight visualization |
| Monte Carlo | Seeded portfolio paths from daily (not annualised) drift and volatility, plus distribution-based risk measures |
| Options | Black–Scholes prices and Greeks (vega and rho per 1 percentage point, theta per calendar day) |
| Scenario analysis | Gaussian-mixture paths with a one-day market shock and an annual rate shock scaled by 1/252 |
| Reporting | Interactive Plotly charts and decision-oriented KPI cards |

## Application Preview

### Market Overview

![Investment dashboard overview](assets/screenshots/01_overview.png)

### Efficient Frontier

![Portfolio efficient frontier](assets/screenshots/02_efficient_frontier.png)

### Monte Carlo Simulation

![Monte Carlo portfolio simulation](assets/screenshots/03_monte_carlo.png)

### Options Pricing

![Black-Scholes options analysis](assets/screenshots/04_options_pricing.png)

### What-If Scenarios

![Investment what-if scenarios](assets/screenshots/05_what_if_scenarios.png)

## Architecture

~~~text
Market Data
    ↓
Data Loader → Analytics & Risk Metrics
    ↓                    ↓
Optimizer           Scenario Models
    ↓                    ↓
       Streamlit + Plotly Dashboard
~~~

## Run Locally

~~~bash
git clone https://github.com/ParBproject/AI-Investment-Dashboard.git
cd AI-Investment-Dashboard

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

Requires Python 3.11 or newer. `requirements.txt` pins the libraries the tests were run against. Matplotlib is included for `notebooks/01_model_exploration.ipynb`.

Open http://localhost:8501.

## Repository Structure

~~~text
AI-Investment-Dashboard/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── optimizer.py
│   └── utils.py
├── notebooks/01_model_exploration.ipynb
├── tests/
├── assets/screenshots/
├── .github/workflows/ci.yml
└── requirements.txt
~~~

## Skills Demonstrated

Python, pandas, NumPy, SciPy, scikit-learn, financial modelling, portfolio optimization, Monte Carlo methods, option pricing, scenario analysis, Streamlit, Plotly, and modular application design.

## Responsible Use

This project is for educational and simulation purposes and is not financial advice. Historical and simulated results do not guarantee future performance. Production use would require validated data, model governance, monitoring, security controls, and professional risk review.
