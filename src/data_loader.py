"""
data_loader.py
==============
Handles price data retrieval from yfinance and CSV uploads.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Fetch adjusted closing prices for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : tuple of str
        Stock/ETF ticker symbols (e.g. ('AAPL', 'MSFT')).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and one column per ticker, or None on error.
    """
    try:
        import yfinance as yf

        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            return None

        # yfinance multi-ticker returns MultiIndex columns; extract 'Close'
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        # Rename single-ticker case
        if len(tickers) == 1:
            prices.columns = list(tickers)

        # Drop columns that are entirely NaN
        prices = prices.dropna(axis=1, how="all")

        # Forward-fill sporadic missing values then drop remaining NaNs
        prices = prices.ffill().dropna()

        return prices

    except Exception as exc:
        st.error(f"Error fetching data: {exc}")
        return None


def load_csv_prices(uploaded_file) -> pd.DataFrame:
    """
    Load price data from a user-uploaded CSV file.

    Expected format:
    - First column: Date (parseable by pd.to_datetime)
    - Remaining columns: One per asset, containing adjusted close prices

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame with DatetimeIndex.
    """
    try:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill().dropna()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as exc:
        import streamlit as st
        st.error(f"Error reading CSV: {exc}")
        return pd.DataFrame()


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
    if prices is not None and not prices.empty:
        return prices.iloc[:, 0].pct_change().dropna()
    return pd.Series(dtype=float)
