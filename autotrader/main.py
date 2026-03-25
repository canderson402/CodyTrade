"""AutoTrader entry point — runs the full pipeline on a schedule.

Three scheduled runs per weekday:
  9:00 AM CT  — Morning: fetch data, compute signals, execute trades
 12:00 PM CT  — Midday:  refresh signals with updated news (no trades)
  4:30 PM CT  — Evening: evaluate accuracy, generate daily report

Start with: python main.py
Stop with:  Ctrl+C
"""

import logging
import sys
import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.insert(0, os.path.dirname(__file__))
from config import WATCHLIST, PAPER_TRADING
from analysis.signals import compute_composite_signal
from prediction.model import train_model, predict_direction
from trading.paper_trader import execute_signal
from tracking.database import log_signal, log_trade
from tracking.report import generate_daily_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autotrader")


def morning_pipeline() -> None:
    """Pre-market run: fetch data, compute signals, retrain models on Mondays, execute trades."""
    logger.info("=== MORNING PIPELINE START ===")

    # Retrain ML models every Monday to stay current with recent patterns
    if datetime.now().weekday() == 0:
        logger.info("Monday — retraining ML models")
        for ticker in WATCHLIST:
            train_model(ticker)

    for ticker in WATCHLIST:
        try:
            # Compute composite signal (technical + sentiment)
            signal = compute_composite_signal(ticker)
            log_signal(signal)
            logger.info("%s signal: %s (composite=%.2f)",
                        ticker, signal["signal"], signal["composite_score"])

            # Get ML prediction for context
            direction, confidence = predict_direction(ticker)
            logger.info("%s prediction: %s (confidence=%.2f)", ticker, direction, confidence)

            # Execute paper trade if signal warrants it
            if PAPER_TRADING:
                result = execute_signal(signal)
                if result:
                    log_trade(result)
                    logger.info("%s trade executed: %s", ticker, result["action"])
            else:
                logger.warning("PAPER_TRADING is False — skipping execution")

        except Exception as e:
            logger.error("Morning pipeline failed for %s: %s", ticker, e)

    logger.info("=== MORNING PIPELINE COMPLETE ===")


def midday_pipeline() -> None:
    """Midday refresh: recompute signals with updated news. No trades to avoid overtrading."""
    logger.info("=== MIDDAY PIPELINE START ===")

    for ticker in WATCHLIST:
        try:
            signal = compute_composite_signal(ticker)
            log_signal(signal)
            logger.info("%s midday signal: %s (composite=%.2f)",
                        ticker, signal["signal"], signal["composite_score"])
        except Exception as e:
            logger.error("Midday pipeline failed for %s: %s", ticker, e)

    logger.info("=== MIDDAY PIPELINE COMPLETE ===")


def evening_pipeline() -> None:
    """Post-market run: evaluate accuracy and generate daily report."""
    logger.info("=== EVENING PIPELINE START ===")

    # Log end-of-day signals for accuracy tracking
    for ticker in WATCHLIST:
        try:
            signal = compute_composite_signal(ticker)
            log_signal(signal)
        except Exception as e:
            logger.error("Evening signal failed for %s: %s", ticker, e)

    # Generate and print the daily report
    report = generate_daily_report()
    logger.info("\n%s", report)

    logger.info("=== EVENING PIPELINE COMPLETE ===")


def run_once() -> None:
    """Run all three pipelines once immediately — useful for testing."""
    morning_pipeline()
    midday_pipeline()
    evening_pipeline()


def start_scheduler() -> None:
    """Start the APScheduler with all three daily jobs."""
    scheduler = BlockingScheduler(timezone="US/Central")

    # Every weekday at 9:00 AM CT — before market open
    scheduler.add_job(
        morning_pipeline, "cron",
        day_of_week="mon-fri", hour=9, minute=0,
        id="morning", name="Morning Pipeline",
    )

    # Every weekday at 12:00 PM CT — midday refresh
    scheduler.add_job(
        midday_pipeline, "cron",
        day_of_week="mon-fri", hour=12, minute=0,
        id="midday", name="Midday Pipeline",
    )

    # Every weekday at 4:30 PM CT — after market close
    scheduler.add_job(
        evening_pipeline, "cron",
        day_of_week="mon-fri", hour=16, minute=30,
        id="evening", name="Evening Pipeline",
    )

    logger.info("AutoTrader scheduler started. Press Ctrl+C to stop.")
    logger.info("Scheduled jobs:")
    logger.info("  Morning Pipeline:  Mon-Fri 9:00 AM CT")
    logger.info("  Midday Pipeline:   Mon-Fri 12:00 PM CT")
    logger.info("  Evening Pipeline:  Mon-Fri 4:30 PM CT")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run all pipelines once immediately (for testing)
        print("Running all pipelines once...\n")
        run_once()
    else:
        # Start the scheduled loop
        start_scheduler()
