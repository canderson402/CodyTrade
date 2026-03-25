"""Fetch historical and current stock price data using yfinance.

This is the primary data source for Phase 1. Alpaca can replace yfinance
for production use if lower latency or real-time streaming is needed.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import HISTORY_DAYS

logger = logging.getLogger(__name__)


def get_historical_bars(ticker: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Fetch daily OHLCV bars for the last N days. Returns a DataFrame with lowercase columns."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"))

        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame()

        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Keep only the columns we need for analysis
        expected_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in expected_cols if c in df.columns]
        df = df[available_cols]

        return df

    except Exception as e:
        logger.error("Failed to fetch historical bars for %s: %s", ticker, e)
        return pd.DataFrame()


def get_current_price(ticker: str) -> float | None:
    """Get the most recent closing price for a ticker. Returns a float or None on failure."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            logger.warning("No price data returned for %s", ticker)
            return None

        # Use the last closing price available
        price = float(hist["Close"].iloc[-1])
        return price

    except Exception as e:
        logger.error("Failed to fetch current price for %s: %s", ticker, e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Market Data Demo ===")
    print("\nFetching AAPL historical bars (90 days)...")
    bars = get_historical_bars("AAPL")
    if not bars.empty:
        print(bars.head())
        print(f"\nTotal rows: {len(bars)}")
    else:
        print("No data returned.")

    print("\nFetching AAPL current price...")
    price = get_current_price("AAPL")
    if price is not None:
        print(f"AAPL current price: ${price:.2f}")
    else:
        print("Could not fetch price.")
