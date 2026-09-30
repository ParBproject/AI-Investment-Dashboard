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


def _text(elements) -> str:
    return "\n".join(element.value for element in elements)


def test_dashboard_opens_before_run() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception


def test_scenario_tab_is_labeled_as_a_gaussian_mixture() -> None:
    """The what-if tab is a Gaussian mixture, not an AI feature."""
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception

    tab_labels = [tab.label for tab in app.tabs]
    assert "📉 What-If Scenarios" in tab_labels
    assert "🤖 AI What-If Scenarios" not in tab_labels

    captions = _text(app.caption)
    overview = _text(app.markdown)
    assert "Gaussian-Mixture Scenarios" in captions
    assert "AI Scenario Generation" not in captions
    assert "**What-If Scenarios**" in overview
    assert "Gaussian-mixture synthetic returns and shock scenarios" in overview
    assert "walk-forward max-Sharpe" in overview
    assert "in-sample portfolio" in overview
    assert "AI What-If Scenarios" not in overview

    readme = (APP.parent / "README.md").read_text()
    assert "Gaussian-mixture scenario analysis" in readme
    assert "Gaussian-mixture paths fit on that same in-sample series" in readme
    assert "| Walk-forward check |" in readme
    assert "AI What-If Scenarios" not in readme
    assert "AI-assisted scenario analysis" not in readme


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
    assert "CVaR 99% (daily, in-sample)" in labels
    assert "In-sample Sharpe" in labels
    assert "Walk-forward Sharpe" in labels
    assert "Equal-weight Sharpe" in labels
    assert "Call Price" in labels
    subheaders = _text(app.subheader)
    assert "What-If Scenario Analysis" in subheaders
    assert "AI-Driven What-If" not in subheaders
    assert "full in-sample return series" in _text(app.caption)
