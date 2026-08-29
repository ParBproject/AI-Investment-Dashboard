# AI-Driven Investment Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![Analytics](https://img.shields.io/badge/Analytics-Optimization_%7C_Monte_Carlo_%7C_Options-1f6feb)](#capabilities)

An interactive financial-analytics application combining market-data exploration, portfolio optimization, Monte Carlo simulation, Black–Scholes option pricing, and AI-assisted scenario analysis.

## Capabilities

| Area | What the application provides |
|---|---|
| Market overview | Normalized prices, returns, correlations, and descriptive statistics |
| Portfolio optimization | Efficient frontier, maximum-Sharpe allocation, and weight visualization |
| Monte Carlo | Simulated portfolio paths and distribution-based risk measures |
| Options | Black–Scholes pricing and sensitivity analysis |
| Scenario analysis | Configurable what-if shocks and AI-style scenario generation |
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
├── assets/screenshots/
└── requirements.txt
~~~

## Skills Demonstrated

Python, pandas, NumPy, SciPy, financial modelling, portfolio optimization, Monte Carlo methods, option pricing, scenario analysis, Streamlit, Plotly, and modular application design.

## Responsible Use

This project is for educational and simulation purposes and is not financial advice. Historical and simulated results do not guarantee future performance. Production use would require validated data, model governance, monitoring, security controls, and professional risk review.
