"""Daily summary report — market overview, signals, trades, portfolio, and prediction accuracy.

Called by the evening pipeline after all metrics are updated. Reports are saved
to reports/YYYY-MM-DD.txt and also printed to console for scheduler logs.
"""

import logging
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WATCHLIST
from data.market import get_current_price, get_historical_bars
from analysis.signals import compute_composite_signal
from trading.paper_trader import get_portfolio_status
from tracking.database import get_recent_signals, get_recent_trades
from tracking.performance import compute_metrics

logger = logging.getLogger(__name__)

_REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))


def _market_overview() -> list[str]:
    """Generate market overview section — SPY direction and basic context."""
    lines = ["--- Market Overview ---"]
    spy_price = get_current_price("SPY")
    spy_bars = get_historical_bars("SPY", days=5)

    if spy_price and not spy_bars.empty:
        prev_close = spy_bars["close"].iloc[-2] if len(spy_bars) >= 2 else spy_price
        change = spy_price - prev_close
        change_pct = (change / prev_close) * 100
        direction = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
        lines.append(f"  SPY: ${spy_price:.2f} ({direction} {change_pct:+.2f}%)")
    else:
        lines.append("  SPY: data unavailable")

    return lines


def _watchlist_signals() -> list[str]:
    """Generate signal summary for each ticker in the watchlist."""
    lines = ["\n--- Watchlist Signals ---"]
    for ticker in WATCHLIST:
        try:
            signal = compute_composite_signal(ticker)
            lines.append(
                f"  {ticker:>5s}:  {signal['signal']:>12s}  "
                f"(tech={signal['technical_score']:.2f}  "
                f"sent={signal['sentiment_score']:.2f}  "
                f"comp={signal['composite_score']:.2f})"
            )
        except Exception as e:
            lines.append(f"  {ticker:>5s}:  ERROR — {e}")
    return lines


def _trade_activity() -> list[str]:
    """Summarize today's trade activity."""
    lines = ["\n--- Trade Activity ---"]
    trades = get_recent_trades(limit=10)
    today = datetime.now().strftime("%Y-%m-%d")

    todays_trades = [t for t in trades if t.get("timestamp", "").startswith(today)]
    if todays_trades:
        for t in todays_trades:
            price_str = f"${t['price']:.2f}" if t.get("price") else "pending"
            lines.append(f"  {t['action']:>4s} {t['ticker']} x{t['qty']} @ {price_str}")
    else:
        lines.append("  No trades executed today.")

    return lines


def _portfolio_snapshot() -> list[str]:
    """Get current portfolio state from Alpaca."""
    lines = ["\n--- Portfolio Snapshot ---"]
    status = get_portfolio_status()

    if status:
        lines.append(f"  Cash:         ${float(status['cash']):>12,.2f}")
        lines.append(f"  Equity:       ${float(status['equity']):>12,.2f}")
        lines.append(f"  Buying Power: ${float(status['buying_power']):>12,.2f}")
        if status["positions"]:
            lines.append(f"  Positions:")
            for p in status["positions"]:
                lines.append(
                    f"    {p['ticker']:>5s}: {p['qty']} shares "
                    f"@ ${float(p['current_price']):,.2f} "
                    f"(P&L: ${float(p['unrealized_pl']):+,.2f})"
                )
        else:
            lines.append("  Positions:    none")
    else:
        lines.append("  Could not connect to Alpaca.")

    return lines


def _prediction_accuracy() -> list[str]:
    """Compare recent signals against actual price movement."""
    lines = ["\n--- Prediction Accuracy ---"]
    trades = get_recent_trades(limit=50)
    metrics = compute_metrics(trades)

    lines.append(f"  Win Rate:     {metrics['win_rate']}%")
    lines.append(f"  Total Trades: {metrics['total_trades']}")
    lines.append(f"  Wins:         {metrics.get('wins', 0)}")
    lines.append(f"  Losses:       {metrics.get('losses', 0)}")

    return lines


def generate_daily_report() -> str:
    """Produce the full daily report. Saves to file and returns the text."""
    lines = []
    lines.append("=" * 55)
    lines.append("  AUTOTRADER DAILY REPORT")
    lines.append(f"  {datetime.now().strftime('%A, %B %d, %Y  %I:%M %p')}")
    lines.append("=" * 55)

    lines.extend(_market_overview())
    lines.extend(_watchlist_signals())
    lines.extend(_trade_activity())
    lines.extend(_portfolio_snapshot())
    lines.extend(_prediction_accuracy())

    lines.append("\n" + "=" * 55)

    report = "\n".join(lines)

    # Save to reports directory
    try:
        os.makedirs(_REPORTS_DIR, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d") + ".txt"
        filepath = os.path.join(_REPORTS_DIR, filename)
        with open(filepath, "w") as f:
            f.write(report)
        logger.info("Daily report saved to %s", filepath)
    except Exception as e:
        logger.error("Failed to save report: %s", e)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    report = generate_daily_report()
    print(report)
