"""
data_loader.py
==============
Handles price data retrieval from yfinance and CSV uploads.

This module stays free of Streamlit. Caching and error display belong in the
app, so a failed download is not stored for the cache TTL and the notebook
can call these functions without a Streamlit runtime.
"""

import pandas as pd
import numpy as np


def extract_close_prices(
    raw: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """
    Pull adjusted close prices out of a yfinance download frame.

    Current yfinance builds a MultiIndex of ``(Price, Ticker)`` even for a
    single symbol. Older downloads used a flat ``Close`` column. Both shapes
    are accepted. The result is forward-filled and rows that are still empty
    are dropped.
    """
    if raw is None or raw.empty:
        raise ValueError("No price data returned. Check the tickers and date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data did not include Close prices.")
        prices = raw["Close"]
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data did not include Close prices.")
        prices = raw[["Close"]].copy()

    if isinstance(prices, pd.Series):
        name = tickers[0] if len(tickers) == 1 else "Close"
        prices = prices.to_frame(name=name)

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise ValueError("No price data returned. Check the tickers and date range.")

    if len(tickers) == 1:
        prices = prices.copy()
        prices.columns = [tickers[0]]

    prices = prices.ffill().dropna()
    if prices.empty:
        raise ValueError("Price history is empty after cleaning missing values.")
    return prices


def fetch_price_data(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : tuple of str
        Stock/ETF ticker symbols (e.g. ('AAPL', 'MSFT')).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format, inclusive. yfinance's own ``end``
        argument is exclusive; this function shifts it by one day.

    Returns
    -------
    pd.DataFrame
        DataFrame with a Date index and one column per ticker.

    Raises
    ------
    ValueError
        If no ticker is provided or the download yields no usable prices.
    """
    if not tickers:
        raise ValueError("Provide at least one ticker.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError("yfinance is required to download prices.") from exc

    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=_yfinance_exclusive_end(end),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(f"Error fetching data: {exc}") from exc

    return extract_close_prices(raw, tuple(tickers))


def _yfinance_exclusive_end(end: str) -> str:
    """
    Convert an inclusive end date to the exclusive bound yfinance expects.

    ``yf.download(..., end="2024-12-31")`` stops on 2024-12-30. The sidebar
    date is the last session the user asked for, so the request uses the
    following calendar day.
    """
    try:
        day = pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"End date is not a valid date: {end}") from exc
    if pd.isna(day):
        raise ValueError(f"End date is not a valid date: {end}")
    return (day.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def load_csv_prices(uploaded_file) -> pd.DataFrame:
    """
    Load price data from a user-uploaded CSV file.

    Expected format:
    - First column: Date (parseable by pd.to_datetime)
    - Remaining columns: One per asset, containing adjusted close prices

    Parameters
    ----------
    uploaded_file : file-like
        Streamlit UploadedFile, path, or buffer.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame with a DatetimeIndex.

    Raises
    ------
    ValueError
        If the file cannot be read or has no usable numeric prices.
    """
    try:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    except Exception as exc:
        raise ValueError(f"Error reading CSV: {exc}") from exc

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.ffill().dropna()
    if df.empty:
        raise ValueError("CSV has no usable numeric price columns.")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


def get_benchmark_data(
    benchmark: str = "SPY",
    start: str = "2021-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """
    Fetch a benchmark index (default SPY) for comparison.

    Returns
    -------
    pd.Series
        Daily returns of the benchmark.
    """
    prices = fetch_price_data((benchmark,), start, end)
    return prices.iloc[:, 0].pct_change().dropna()
