"""Smoke tests for the Streamlit entry point. No market-data download."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


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
