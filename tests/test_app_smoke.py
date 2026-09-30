"""Smoke tests for the Streamlit entry point. No market-data download."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _price_csv() -> bytes:
    rng = np.random.default_rng(0)
    n_days = 80
    shocks = rng.normal(0.0004, 0.01, size=(n_days, 3))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    index = pd.bdate_range("2021-01-04", periods=n_days)
    lines = ["Date,AAA,BBB,CCC"]
    for day, row in zip(index, prices):
        lines.append(f"{day.date()},{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}")
    return "\n".join(lines).encode()


def test_dashboard_opens_before_run() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception


def test_empty_tickers_show_an_error_instead_of_crashing() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    app.text_input[0].set_value("")
    app.button[0].click().run()
    assert not app.exception
    assert any("ticker" in error.value.lower() for error in app.error)


def test_csv_analysis_runs_without_a_download() -> None:
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.run()
    app.radio[0].set_value("Upload CSV").run()
    app.file_uploader[0].set_value(("prices.csv", _price_csv(), "text/csv")).run()
    app.button[0].click().run()

    assert not app.exception
    assert app.error == []
    labels = [metric.label for metric in app.metric]
    assert "CVaR 99% (daily)" in labels
    assert "Max Sharpe Ratio" in labels
    assert "Call Price" in labels
