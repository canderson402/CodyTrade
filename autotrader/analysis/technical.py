"""Compute technical indicators and derive a bullish/bearish score from price data.

Uses pandas-ta for all indicator calculations. The technical_score function
implements rule-based scoring — each rule is commented with *why* it matters.
"""

import logging

import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    ta = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RSI_PERIOD, EMA_SHORT, EMA_LONG, BB_PERIOD, BB_STD

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, and EMA columns to a price DataFrame. Returns the enriched DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to compute_indicators")
        return df

    if ta is None:
        logger.warning("pandas-ta not installed — skipping indicators")
        return df

    # RSI — momentum oscillator that flags overbought/oversold conditions
    df["rsi"] = df.ta.rsi(length=RSI_PERIOD)

    # MACD — trend-following momentum indicator (fast/slow/signal)
    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
    df["macd"] = macd_df.iloc[:, 0]         # MACD line
    df["macd_signal"] = macd_df.iloc[:, 2]  # Signal line

    # Bollinger Bands — volatility envelope around a moving average
    bb_df = df.ta.bbands(length=BB_PERIOD, std=BB_STD)
    df["bb_lower"] = bb_df.iloc[:, 0]   # Lower band
    df["bb_upper"] = bb_df.iloc[:, 2]   # Upper band

    # Exponential Moving Averages — short-term vs long-term trend
    df["ema_20"] = df.ta.ema(length=EMA_SHORT)
    df["ema_50"] = df.ta.ema(length=EMA_LONG)

    return df


def technical_score(df: pd.DataFrame) -> float:
    """Score the latest row of indicators from 0.0 (strong sell) to 1.0 (strong buy)."""
    if df.empty:
        return 0.5  # No data → neutral

    # Use the most recent row for scoring
    latest = df.iloc[-1]

    # Start at neutral — each rule nudges the score toward bullish or bearish
    score = 0.5

    # --- RSI Rule ---
    # RSI < 30 means the stock is oversold — buyers often step in here (mean reversion)
    # RSI > 70 means overbought — selling pressure tends to increase
    rsi = latest.get("rsi")
    if pd.notna(rsi):
        if rsi < 30:
            score += 0.2
        elif rsi > 70:
            score -= 0.2

    # --- EMA Rule ---
    # Price above the 50-day EMA suggests an established uptrend;
    # below it suggests a downtrend. This is the simplest trend filter.
    close = latest.get("close")
    ema_50 = latest.get("ema_50")
    if pd.notna(close) and pd.notna(ema_50):
        if close > ema_50:
            score += 0.15
        else:
            score -= 0.15

    # --- MACD Rule ---
    # When MACD crosses above its signal line, momentum is shifting bullish.
    # Below the signal line means bearish momentum is building.
    macd = latest.get("macd")
    macd_sig = latest.get("macd_signal")
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig:
            score += 0.15
        else:
            score -= 0.15

    # --- Bollinger Band Rule ---
    # Price near the lower band suggests the stock is stretched to the downside
    # and may bounce. Near the upper band, it may be overextended upward.
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")
    if pd.notna(close) and pd.notna(bb_lower) and pd.notna(bb_upper):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            # bb_pct: 0.0 = at lower band, 1.0 = at upper band
            bb_pct = (close - bb_lower) / bb_range
            if bb_pct < 0.2:
                score += 0.1
            elif bb_pct > 0.8:
                score -= 0.1

    # Clamp to valid range — scores can't exceed certainty
    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.market import get_historical_bars

    print("=== Technical Analysis Demo ===")
    print("\nFetching AAPL data and computing indicators...")
    df = get_historical_bars("AAPL")

    if not df.empty:
        df = compute_indicators(df)
        print("\nLatest indicator values:")
        latest = df.iloc[-1]
        print(f"  RSI:         {latest.get('rsi', 'N/A'):.2f}")
        print(f"  MACD:        {latest.get('macd', 'N/A'):.4f}")
        print(f"  MACD Signal: {latest.get('macd_signal', 'N/A'):.4f}")
        print(f"  BB Lower:    {latest.get('bb_lower', 'N/A'):.2f}")
        print(f"  BB Upper:    {latest.get('bb_upper', 'N/A'):.2f}")
        print(f"  EMA 20:      {latest.get('ema_20', 'N/A'):.2f}")
        print(f"  EMA 50:      {latest.get('ema_50', 'N/A'):.2f}")

        score = technical_score(df)
        print(f"\n  Technical Score: {score:.2f}")
    else:
        print("No data returned.")
