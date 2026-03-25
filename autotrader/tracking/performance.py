"""Performance tracking and reporting.

Computes win rate, return, Sharpe ratio, and max drawdown from trade history.
Uses quantstats for portfolio analytics and comparison against SPY benchmark.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import quantstats as qs

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracking.database import get_recent_signals, get_recent_trades
from data.market import get_historical_bars

logger = logging.getLogger(__name__)


def _build_returns_from_trades(trades: list[dict], days: int = 90) -> pd.Series:
    """Build a daily returns series from trade history. Returns a pandas Series indexed by date."""
    if not trades:
        return pd.Series(dtype=float)

    # Get unique tickers from trades
    tickers = list({t["ticker"] for t in trades})

    # Fetch price data for all traded tickers
    all_prices = {}
    for ticker in tickers:
        df = get_historical_bars(ticker, days=days)
        if not df.empty:
            all_prices[ticker] = df["close"]

    if not all_prices:
        return pd.Series(dtype=float)

    # Simple approach: compute average daily returns across all traded tickers
    # This approximates portfolio performance when we don't have exact position data
    returns_list = []
    for ticker, prices in all_prices.items():
        daily_returns = prices.pct_change().dropna()
        returns_list.append(daily_returns)

    if not returns_list:
        return pd.Series(dtype=float)

    combined = pd.concat(returns_list, axis=1).mean(axis=1)
    combined.index = combined.index.tz_localize(None) if combined.index.tz else combined.index
    return combined


def compute_metrics(trades: list[dict]) -> dict:
    """Compute basic performance metrics from trade history. Returns a metrics dict."""
    if not trades:
        return {"win_rate": 0.0, "total_trades": 0, "total_return_pct": 0.0}

    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]

    # Match buy/sell pairs to compute wins vs losses
    wins = 0
    losses = 0
    for i, sell in enumerate(sell_trades):
        # Find matching buy for the same ticker before this sell
        matching_buys = [
            b for b in buy_trades
            if b["ticker"] == sell["ticker"]
            and b["price"] is not None
            and sell["price"] is not None
        ]
        if matching_buys:
            buy_price = matching_buys[-1]["price"]
            sell_price = sell["price"]
            if sell_price > buy_price:
                wins += 1
            else:
                losses += 1

    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    return {
        "win_rate": round(win_rate, 1),
        "wins": wins,
        "losses": losses,
        "total_trades": len(trades),
        "open_trades": len(buy_trades) - len(sell_trades),
    }


def generate_report() -> str:
    """Generate a performance summary report. Returns the report as a formatted string."""
    trades = get_recent_trades(limit=100)
    signals = get_recent_signals(limit=100)

    lines = []
    lines.append("=" * 50)
    lines.append("  AUTOTRADER PERFORMANCE REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 50)

    # --- Trade Metrics ---
    metrics = compute_metrics(trades)
    lines.append("\n--- Trade Metrics ---")
    lines.append(f"  Total trades logged:  {metrics['total_trades']}")
    lines.append(f"  Win rate:             {metrics['win_rate']}%")
    lines.append(f"  Wins / Losses:        {metrics.get('wins', 0)} / {metrics.get('losses', 0)}")

    # --- Signal Summary ---
    lines.append("\n--- Recent Signals ---")
    if signals:
        for s in signals[:5]:
            lines.append(f"  {s['ticker']:>5s}  {s['signal']:>12s}  "
                         f"composite={s['composite_score']:.2f}  "
                         f"({s['timestamp'][:16]})")
    else:
        lines.append("  No signals recorded yet.")

    # --- Quantstats Analysis ---
    lines.append("\n--- Portfolio Analytics ---")
    returns = _build_returns_from_trades(trades)
    if not returns.empty and len(returns) > 5:
        try:
            sharpe = qs.stats.sharpe(returns)
            max_dd = qs.stats.max_drawdown(returns)
            total_ret = qs.stats.comp(returns)
            lines.append(f"  Sharpe Ratio:   {sharpe:.2f}")
            lines.append(f"  Max Drawdown:   {max_dd:.2%}")
            lines.append(f"  Total Return:   {total_ret:.2%}")
        except Exception as e:
            lines.append(f"  Could not compute analytics: {e}")
    else:
        lines.append("  Not enough trade data for analytics yet.")
        lines.append("  (Run the system for a few days to build up history.)")

    lines.append("\n" + "=" * 50)

    report = "\n".join(lines)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    report = generate_report()
    print(report)
