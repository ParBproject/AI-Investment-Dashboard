from io import StringIO

import pandas as pd
import pytest

import yfinance as yf

from src.data_loader import extract_close_prices, fetch_price_data, load_csv_prices


def _dates() -> pd.DatetimeIndex:
    return pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])


def test_extract_close_from_yfinance_multiindex() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("Close", "MSFT"),
            ("High", "AAPL"),
            ("High", "MSFT"),
        ],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[10.0, 20.0, 11.0, 21.0], [11.0, 22.0, 12.0, 23.0], [12.0, 21.0, 13.0, 22.0]],
        index=_dates(),
        columns=columns,
    )
    prices = extract_close_prices(raw, ("AAPL", "MSFT"))
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert prices.iloc[0, 0] == 10.0
    assert "High" not in prices.columns


def test_extract_close_renames_a_single_ticker() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("High", "AAPL")],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]],
        index=_dates(),
        columns=columns,
    )
    prices = extract_close_prices(raw, ("AAPL",))
    assert list(prices.columns) == ["AAPL"]


def test_extract_close_accepts_a_flat_frame() -> None:
    raw = pd.DataFrame({"Open": [1, 2, 3], "Close": [4, 5, 6]}, index=_dates())
    prices = extract_close_prices(raw, ("SPY",))
    assert list(prices.columns) == ["SPY"]
    assert prices.iloc[-1, 0] == 6


def test_extract_close_rejects_an_empty_download() -> None:
    with pytest.raises(ValueError, match="No price data"):
        extract_close_prices(pd.DataFrame(), ("AAPL",))


def test_fetch_price_data_rejects_an_empty_ticker_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fetch_price_data((), "2024-01-01", "2024-02-01")


def test_load_csv_prices_parses_dates_and_sorts() -> None:
    raw = StringIO(
        "Date,AAPL,MSFT\n"
        "2024-01-03,11,21\n"
        "2024-01-02,10,20\n"
        "2024-01-04,12,22\n"
    )
    prices = load_csv_prices(raw)
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert list(prices.index) == list(_dates())
    assert prices.iloc[0, 0] == 10


def test_fetch_includes_the_selected_end_date(monkeypatch: pytest.MonkeyPatch) -> None:
    # yfinance treats ``end`` as exclusive, so an unshifted 2024-12-31 request
    # would stop on 2024-12-30 and drop the session the user selected.
    captured: dict[str, str] = {}

    def fake_download(tickers, start, end, auto_adjust, progress):
        captured["start"] = start
        captured["end"] = end
        index = _dates()
        columns = pd.MultiIndex.from_product([["Close"], ["AAPL"]])
        return pd.DataFrame([[10.0], [11.0], [12.0]], index=index, columns=columns)

    monkeypatch.setattr(yf, "download", fake_download)
    prices = fetch_price_data(("AAPL",), "2024-01-02", "2024-01-04")
    assert captured["end"] == "2024-01-05"
    assert prices.index[-1] == pd.Timestamp("2024-01-04")


def test_fetch_rejects_an_unparseable_end_date(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_download(*args, **kwargs):
        raise AssertionError("download should not run when the end date is invalid")

    monkeypatch.setattr(yf, "download", fail_download)
    with pytest.raises(ValueError, match="End date"):
        fetch_price_data(("AAPL",), "2024-01-02", "not-a-date")


def test_load_csv_prices_rejects_a_file_with_no_numbers() -> None:
    raw = StringIO("Date,Note\n2024-01-02,hello\n2024-01-03,world\n")
    with pytest.raises(ValueError, match="no usable"):
        load_csv_prices(raw)
