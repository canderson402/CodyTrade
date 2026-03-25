"""Fetch intraday price bars from Alpaca for scalp trading.

Uses Alpaca's market data API instead of yfinance because we need
minute-level bars that yfinance doesn't reliably provide.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, SCALP_BAR_INTERVAL, SCALP_LOOKBACK_BARS

logger = logging.getLogger(__name__)

# Map config string to Alpaca TimeFrame object
_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1Min": TimeFrame.Minute,
    "5Min": TimeFrame(5, TimeFrame.Unit.Minute) if hasattr(TimeFrame, "Unit") else TimeFrame.Minute,
    "15Min": TimeFrame(15, TimeFrame.Unit.Minute) if hasattr(TimeFrame, "Unit") else TimeFrame.Minute,
    "30Min": TimeFrame(30, TimeFrame.Unit.Minute) if hasattr(TimeFrame, "Unit") else TimeFrame.Hour,
    "1Hour": TimeFrame.Hour,
}


def _get_data_client() -> StockHistoricalDataClient | None:
    """Create an Alpaca data client. Returns None if keys are missing."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API keys not set — check your .env file")
        return None

    try:
        return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except Exception as e:
        logger.error("Failed to create Alpaca data client: %s", e)
        return None


def get_intraday_bars(ticker: str, interval: str = SCALP_BAR_INTERVAL,
                      bars: int = SCALP_LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch intraday OHLCV bars from Alpaca. Returns DataFrame with lowercase columns."""
    client = _get_data_client()
    if client is None:
        return pd.DataFrame()

    try:
        timeframe = _TIMEFRAME_MAP.get(interval, TimeFrame.Minute)

        # Fetch enough calendar time to cover the requested number of bars.
        # Market is open ~6.5 hours/day, so 5-min bars = ~78 bars/day.
        # Fetch 3 days to ensure we get enough bars even over weekends.
        end = datetime.now()
        start = end - timedelta(days=3)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=bars,
        )

        bar_set = client.get_stock_bars(request)
        df = bar_set.df

        if df.empty:
            logger.warning("No intraday data returned for %s", ticker)
            return pd.DataFrame()

        # Flatten multi-index if present (Alpaca returns symbol as top-level index)
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Keep only OHLCV columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in expected_cols if c in df.columns]
        df = df[available_cols]

        return df.tail(bars)

    except Exception as e:
        logger.error("Failed to fetch intraday bars for %s: %s", ticker, e)
        return pd.DataFrame()


def get_latest_price(ticker: str) -> float | None:
    """Get the most recent intraday price from Alpaca. Returns a float or None."""
    df = get_intraday_bars(ticker, bars=1)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Intraday Data Demo ===\n")

    ticker = "AAPL"
    print(f"Fetching {SCALP_BAR_INTERVAL} bars for {ticker}...")
    df = get_intraday_bars(ticker)

    if not df.empty:
        print(f"Got {len(df)} bars")
        print(df.tail(10))
        price = get_latest_price(ticker)
        print(f"\nLatest price: ${price:.2f}" if price else "\nNo price available")
    else:
        print("No data returned — market may be closed or API keys not set.")
