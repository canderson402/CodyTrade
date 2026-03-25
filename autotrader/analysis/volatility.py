"""Intraday volatility indicators for the scalp trading strategy.

Computes ATR (how much the stock moves per bar), VWAP (fair value line),
and intraday Bollinger Bands. These tell the scalper when price is
stretched and likely to snap back.
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCALP_ATR_PERIOD, SCALP_BB_PERIOD, SCALP_BB_STD

logger = logging.getLogger(__name__)


def compute_scalp_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, VWAP, and Bollinger Bands to an intraday DataFrame. Returns enriched DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to compute_scalp_indicators")
        return df

    # ATR — Average True Range: measures how much price moves per bar.
    # Used to set dynamic stop-loss and take-profit levels.
    df["atr"] = df.ta.atr(length=SCALP_ATR_PERIOD)

    # VWAP — Volume-Weighted Average Price: the "fair value" for the day.
    # Institutional traders use this as a benchmark. Price below VWAP
    # is considered cheap, above VWAP is considered expensive.
    if "volume" in df.columns:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        df["vwap"] = np.where(
            cumulative_vol > 0,
            cumulative_tp_vol / cumulative_vol,
            df["close"],
        )
    else:
        df["vwap"] = df["close"]

    # Intraday Bollinger Bands — volatility envelope for mean reversion signals
    bb_df = df.ta.bbands(length=SCALP_BB_PERIOD, std=SCALP_BB_STD)
    if bb_df is not None and not bb_df.empty:
        df["bb_lower"] = bb_df.iloc[:, 0]
        df["bb_mid"] = bb_df.iloc[:, 1]
        df["bb_upper"] = bb_df.iloc[:, 2]
    else:
        df["bb_lower"] = df["close"]
        df["bb_mid"] = df["close"]
        df["bb_upper"] = df["close"]

    return df


def scalp_signal(df: pd.DataFrame) -> dict:
    """Generate a scalp signal from the latest intraday indicators. Returns a signal dict."""
    if df.empty or len(df) < SCALP_BB_PERIOD:
        return {"action": "WAIT", "reason": "Not enough data"}

    latest = df.iloc[-1]
    close = latest.get("close")
    vwap = latest.get("vwap")
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")
    atr = latest.get("atr")

    # Need all indicators to make a decision
    if any(pd.isna(v) for v in [close, vwap, bb_lower, bb_upper, atr]):
        return {"action": "WAIT", "reason": "Indicators still warming up"}

    # --- BUY Signal ---
    # Price dropped below lower Bollinger Band AND is below VWAP.
    # This means price is stretched to the downside and likely to snap back.
    if close <= bb_lower and close < vwap:
        return {
            "action": "BUY",
            "reason": "Price below BB lower + below VWAP (oversold bounce expected)",
            "entry": close,
            "stop_loss": close - (atr * 1.0),  # Will be overridden by config in scalper
            "take_profit": close + (atr * 1.5),
            "atr": atr,
        }

    # --- SELL Signal ---
    # Price pushed above upper Bollinger Band AND is above VWAP.
    # Overextended to the upside — likely to pull back.
    if close >= bb_upper and close > vwap:
        return {
            "action": "SELL",
            "reason": "Price above BB upper + above VWAP (overbought pullback expected)",
            "entry": close,
            "stop_loss": close + (atr * 1.0),
            "take_profit": close - (atr * 1.5),
            "atr": atr,
        }

    # --- No Signal ---
    # Price is between the bands — no clear edge. Wait for a setup.
    return {"action": "WAIT", "reason": "Price within bands — no edge"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.intraday import get_intraday_bars

    print("=== Volatility Analysis Demo ===\n")

    ticker = "AAPL"
    print(f"Fetching intraday bars for {ticker}...")
    df = get_intraday_bars(ticker)

    if not df.empty:
        df = compute_scalp_indicators(df)
        latest = df.iloc[-1]
        print(f"\nLatest indicator values:")
        print(f"  Close:    ${latest.get('close', 0):.2f}")
        print(f"  VWAP:     ${latest.get('vwap', 0):.2f}")
        print(f"  ATR:      ${latest.get('atr', 0):.4f}")
        print(f"  BB Lower: ${latest.get('bb_lower', 0):.2f}")
        print(f"  BB Upper: ${latest.get('bb_upper', 0):.2f}")

        signal = scalp_signal(df)
        print(f"\n  Scalp Signal: {signal['action']}")
        print(f"  Reason:       {signal['reason']}")
    else:
        print("No intraday data — market may be closed.")
