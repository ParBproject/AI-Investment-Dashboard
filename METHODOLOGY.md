# Quantitative Methodology

This document summarizes the analytical methods, assumptions, and validation controls used by the Investment Research Lab.

## 1. Market data

Historical adjusted prices are retrieved through `yfinance` or supplied through a user CSV.

Daily simple returns are calculated as:

```text
r_t = P_t / P_(t-1) - 1
```

Historical estimates can change materially with the selected sample period.

## 2. Portfolio optimization

The optimizer estimates annualized portfolio return and volatility from historical daily returns.

For weights **w**, mean daily returns **μ**, and covariance matrix **Σ**:

```text
Expected annual return = 252 × wᵀμ

Annual variance = 252 × wᵀΣw
```

The application solves:

- maximum-Sharpe allocation;
- global minimum-variance allocation;
- a simulated opportunity set for visual comparison.

Long-only mode constrains weights to [0,1]. Short-enabled mode constrains each asset to [-1,1] while requiring net weights to sum to one.

## 3. Historical risk metrics

Historical Value at Risk and Expected Shortfall are estimated directly from observed portfolio returns.

They are descriptive sample statistics, not guaranteed future loss limits.

## 4. Portfolio Monte Carlo

Portfolio paths use historical daily mean and volatility with normally distributed daily shocks.

The simulation is seeded for reproducibility.

This is a parametric uncertainty model and does not fully capture fat tails, volatility clustering, or structural breaks.

## 5. Black–Scholes

European call and put prices use the Black–Scholes model under the standard assumptions of lognormal diffusion, constant volatility, constant risk-free rate, frictionless trading, and no dividends.

The application reports:

- call/put price;
- Delta;
- Gamma;
- Vega;
- Theta;
- Rho.

## 6. Implied volatility

Implied volatility is recovered by solving for the Black–Scholes volatility that matches an observed market option price.

A bracketed Brent root solver is used.

The observed price must satisfy European no-arbitrage bounds.

## 7. CRR binomial tree

The Cox–Ross–Rubinstein tree provides a discrete-time pricing cross-check.

For each time step:

```text
u = exp(σ √Δt)
d = 1/u
p = (exp(r Δt) - d) / (u - d)
```

European values are obtained by discounted risk-neutral backward induction.

The same tree can also evaluate early exercise for American options.

## 8. Risk-neutral Monte Carlo option pricing

European option prices are also estimated under risk-neutral geometric Brownian motion:

```text
S_T = S_0 exp((r - 0.5σ²)T + σ√T Z)
```

Discounted terminal payoffs produce the price estimate.

Antithetic random variates reduce sampling noise, and the dashboard reports the Monte Carlo standard error.

## 9. Cross-method validation

For European options under matching assumptions:

- Black–Scholes;
- a sufficiently fine CRR tree;
- risk-neutral Monte Carlo

should produce similar values.

The application reports pricing differences explicitly rather than hiding them.

Put–call parity is also checked:

```text
C - P = S - K exp(-rT)
```

A material residual indicates inconsistent inputs or model assumptions.

## 10. Regime-mixture scenarios

A Gaussian Mixture Model is fit to historical optimized-portfolio returns.

This allows the simulated return distribution to contain multiple regimes rather than one Gaussian state.

The scenario layer compares:

- baseline GMM paths;
- a one-time market shock;
- a rate-shock drift adjustment.

These scenarios are conditional simulations. They do not assign a probability that the specified shock will occur.

## 11. Reproducibility

Where stochastic models support it, a fixed random seed is used for demonstrations and tests.

GitHub Actions runs linting, compilation, and regression tests on Python 3.10 and 3.12.

## 12. Limitations

The project does not model every production consideration, including:

- bid–ask spreads;
- market impact;
- taxes;
- liquidity constraints;
- changing volatility surfaces;
- stochastic interest rates;
- dividends in the core Black–Scholes implementation;
- counterparty risk;
- margin requirements;
- full model governance and independent validation.

The repository is educational quantitative research and not investment advice.
