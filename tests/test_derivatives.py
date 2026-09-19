import unittest

from src.derivatives import (
    binomial_option_price,
    black_scholes_price,
    compare_option_pricers,
    implied_volatility,
    monte_carlo_option_price,
    put_call_parity_gap,
)


class DerivativesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spot = 100.0
        self.strike = 105.0
        self.maturity = 0.75
        self.rate = 0.04
        self.volatility = 0.24

    def test_binomial_converges_near_black_scholes(self) -> None:
        bs = black_scholes_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="call",
        )
        tree = binomial_option_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            steps=800,
            option_type="call",
        )
        self.assertAlmostEqual(tree, bs, delta=0.08)

    def test_implied_volatility_recovers_input_sigma(self) -> None:
        market_price = black_scholes_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="put",
        )
        recovered = implied_volatility(
            market_price,
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            option_type="put",
        )
        self.assertAlmostEqual(recovered, self.volatility, places=6)

    def test_put_call_parity_for_black_scholes_prices(self) -> None:
        call = black_scholes_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="call",
        )
        put = black_scholes_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="put",
        )
        gap = put_call_parity_gap(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            call,
            put,
        )
        self.assertAlmostEqual(gap, 0.0, places=9)

    def test_monte_carlo_is_consistent_with_black_scholes(self) -> None:
        bs = black_scholes_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="call",
        )
        mc = monte_carlo_option_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            n_paths=80_000,
            option_type="call",
            seed=123,
        )
        tolerance = max(0.10, 4.0 * mc.standard_error)
        self.assertAlmostEqual(mc.price, bs, delta=tolerance)

    def test_american_put_not_below_european_put(self) -> None:
        european = binomial_option_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            steps=400,
            option_type="put",
            american=False,
        )
        american = binomial_option_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            steps=400,
            option_type="put",
            american=True,
        )
        self.assertGreaterEqual(american, european)

    def test_comparison_exposes_model_differences(self) -> None:
        result = compare_option_pricers(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            option_type="call",
            binomial_steps=400,
            monte_carlo_paths=20_000,
            seed=7,
        )
        self.assertGreater(result.black_scholes, 0.0)
        self.assertGreater(result.binomial, 0.0)
        self.assertGreater(result.monte_carlo, 0.0)


if __name__ == "__main__":
    unittest.main()
