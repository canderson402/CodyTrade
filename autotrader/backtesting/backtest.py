"""Backtest the signal engine on historical data using Backtrader.

Uses only the technical score for backtesting because historical news
sentiment data isn't available through our free APIs. This is standard —
sentiment backtesting requires expensive historical news datasets.
"""

import logging
from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    WATCHLIST,
    BACKTEST_CASH,
    BACKTEST_COMMISSION,
    BACKTEST_POSITION_PCT,
)
from data.market import get_historical_bars
from analysis.technical import compute_indicators, technical_score

logger = logging.getLogger(__name__)

# Extra calendar days to fetch before the backtest start date so
# indicators (especially 50-day EMA) have time to stabilize.
_WARMUP_DAYS = 80


class SignalStrategy(bt.Strategy):
    """Buy when technical score exceeds buy threshold, sell when it drops below sell threshold."""

    # Scores dict is passed in at runtime — maps dates to precomputed scores
    params = (("scores", {}),)

    def __init__(self) -> None:
        """Initialize order tracking."""
        self.order = None
        self.trade_count = 0

    def next(self) -> None:
        """Evaluate the signal score on each bar and place orders accordingly."""
        # Don't stack orders — wait for the pending one to fill or cancel
        if self.order:
            return

        current_date = self.data.datetime.date(0)
        score = self.params.scores.get(current_date, 0.5)

        if score >= SIGNAL_BUY_THRESHOLD and not self.position:
            # Allocate a fixed % of portfolio to this position
            cash_to_use = self.broker.getvalue() * BACKTEST_POSITION_PCT
            size = int(cash_to_use / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)

        elif score <= SIGNAL_SELL_THRESHOLD and self.position:
            # Exit the entire position
            self.order = self.close()

    def notify_order(self, order: bt.Order) -> None:
        """Track completed trades and clear the pending order reference."""
        if order.status in [order.Completed]:
            self.trade_count += 1
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def _prepare_data(
    ticker: str, start_date: datetime, end_date: datetime
) -> tuple[pd.DataFrame, dict]:
    """Fetch data, compute indicators and scores. Returns (DataFrame, score_dict)."""
    total_days = (end_date - start_date).days + _WARMUP_DAYS

    df = get_historical_bars(ticker, days=total_days)
    if df.empty:
        return pd.DataFrame(), {}

    df = compute_indicators(df)

    # Compute the technical score for each bar individually.
    # Each row is scored using only its own indicator values.
    scores = [technical_score(df.iloc[[i]]) for i in range(len(df))]
    df["signal_score"] = scores

    # Backtrader needs an openinterest column even if unused
    df["openinterest"] = 0

    # Remove timezone info — Backtrader doesn't handle tz-aware indices
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Build lookup dict before filtering (strategy uses dates to look up scores)
    score_dict = {
        date.date(): score for date, score in zip(df.index, df["signal_score"])
    }

    # Trim to the backtest window (warmup rows are still in score_dict if needed)
    df = df.loc[start_date:end_date]

    return df, score_dict


def run_backtest(
    ticker: str, start_date: datetime, end_date: datetime
) -> dict | None:
    """Run a backtest for one ticker over a date range. Returns a results dict or None on failure."""
    try:
        df, score_dict = _prepare_data(ticker, start_date, end_date)

        if df.empty or len(df) < 10:
            logger.error("Not enough data to backtest %s", ticker)
            return None

        cerebro = bt.Cerebro()

        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        cerebro.addstrategy(SignalStrategy, scores=score_dict)
        cerebro.broker.setcash(BACKTEST_CASH)
        cerebro.broker.setcommission(commission=BACKTEST_COMMISSION)

        results = cerebro.run()
        strategy = results[0]

        final_value = cerebro.broker.getvalue()
        return_pct = (final_value - BACKTEST_CASH) / BACKTEST_CASH * 100

        return {
            "ticker": ticker,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "starting_cash": BACKTEST_CASH,
            "final_value": round(final_value, 2),
            "return_pct": round(return_pct, 2),
            "num_trades": strategy.trade_count,
        }

    except Exception as e:
        logger.error("Backtest failed for %s: %s", ticker, e)
        return None


def run_all_backtests(days: int = 365) -> list[dict]:
    """Run backtests for every ticker in WATCHLIST over the last N days. Returns a list of result dicts."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    results = []
    for ticker in WATCHLIST:
        print(f"  Backtesting {ticker}...")
        result = run_backtest(ticker, start_date, end_date)
        if result:
            results.append(result)

    return results


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table of backtest results."""
    header = f"{'Ticker':<8} {'Return %':>10} {'Final Value':>13} {'Trades':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['ticker']:<8} {r['return_pct']:>9.2f}% "
            f"${r['final_value']:>11,.2f} {r['num_trades']:>8}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Backtest Demo ===\n")
    print(f"Running backtests for {WATCHLIST} over the last year...\n")

    results = run_all_backtests()

    if results:
        print("\n=== Results ===\n")
        _print_results_table(results)
    else:
        print("No backtest results returned.")
