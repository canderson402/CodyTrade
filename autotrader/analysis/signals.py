"""Combine technical and sentiment scores into a single composite trading signal.

This is the decision layer — it takes the output of technical.py and sentiment.py,
weights them according to config, and produces a clear BUY / HOLD / SELL label.
"""

import logging
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    WEIGHT_TECHNICAL,
    WEIGHT_SENTIMENT,
    SIGNAL_STRONG_BUY_THRESHOLD,
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    SIGNAL_STRONG_SELL_THRESHOLD,
    HISTORY_DAYS,
)
from data.market import get_historical_bars
from data.news import get_news_headlines
from analysis.technical import compute_indicators, technical_score
from analysis.sentiment import score_headlines

logger = logging.getLogger(__name__)


def _classify_signal(composite: float) -> str:
    """Map a composite score to a human-readable signal label."""
    # Thresholds create five zones from strong sell to strong buy.
    # The wider "hold" zone in the middle prevents over-trading on weak signals.
    if composite >= SIGNAL_STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    elif composite >= SIGNAL_BUY_THRESHOLD:
        return "BUY"
    elif composite > SIGNAL_SELL_THRESHOLD:
        return "HOLD"
    elif composite > SIGNAL_STRONG_SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG_SELL"


def compute_composite_signal(ticker: str) -> dict:
    """Fetch data, compute all scores, and return a complete signal dict for a ticker."""
    # Step 1: Get price data and compute technical indicators
    df = get_historical_bars(ticker, days=HISTORY_DAYS)
    if not df.empty:
        df = compute_indicators(df)
    tech_score = technical_score(df)

    # Step 2: Get news headlines and compute sentiment
    headlines = get_news_headlines(ticker)
    sent_score = score_headlines(headlines)

    # Step 3: Weighted average — heavier on technicals because price data
    # is more reliable than headline sentiment on short timeframes
    composite = (WEIGHT_TECHNICAL * tech_score) + (WEIGHT_SENTIMENT * sent_score)

    # Step 4: Classify into a discrete signal label
    signal_label = _classify_signal(composite)

    return {
        "ticker": ticker,
        "technical_score": round(tech_score, 4),
        "sentiment_score": round(sent_score, 4),
        "composite_score": round(composite, 4),
        "signal": signal_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Composite Signal Demo ===\n")
    signal = compute_composite_signal("AAPL")

    print(f"Ticker:          {signal['ticker']}")
    print(f"Technical Score:  {signal['technical_score']}")
    print(f"Sentiment Score:  {signal['sentiment_score']}")
    print(f"Composite Score:  {signal['composite_score']}")
    print(f"Signal:           {signal['signal']}")
    print(f"Timestamp:        {signal['timestamp']}")
