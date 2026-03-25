"""Fetch financial news headlines for a given stock ticker using Finnhub.

Headlines are returned as plain strings — sentiment scoring happens
separately in analysis/sentiment.py.
"""

import logging
from datetime import datetime, timedelta

import finnhub

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FINNHUB_API_KEY, MAX_HEADLINES

logger = logging.getLogger(__name__)


def get_news_headlines(ticker: str, days: int = 1) -> list[str]:
    """Fetch recent news headlines for a ticker. Returns a list of headline strings, max MAX_HEADLINES."""
    try:
        if not FINNHUB_API_KEY:
            logger.error("FINNHUB_API_KEY is not set — check your .env file")
            return []

        client = finnhub.Client(api_key=FINNHUB_API_KEY)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Finnhub company_news expects dates as "YYYY-MM-DD" strings
        news = client.company_news(
            ticker,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
        )

        if not news:
            logger.info("No news found for %s in the last %d day(s)", ticker, days)
            return []

        # Extract only the headline text from each article,
        # capped at MAX_HEADLINES to keep API usage reasonable
        headlines = [article["headline"] for article in news if "headline" in article]
        return headlines[:MAX_HEADLINES]

    except Exception as e:
        logger.error("Failed to fetch news for %s: %s", ticker, e)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== News Headlines Demo ===")
    print("\nFetching AAPL headlines (last 1 day)...")
    headlines = get_news_headlines("AAPL")

    if headlines:
        for i, h in enumerate(headlines, 1):
            print(f"  {i}. {h}")
        print(f"\nTotal headlines: {len(headlines)}")
    else:
        print("No headlines returned (check FINNHUB_API_KEY in .env).")
