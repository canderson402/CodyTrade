"""Database for logging signals, trades, and manual predictions.

Uses SQLAlchemy so the same code works with SQLite (local development)
and PostgreSQL (Supabase in production). Set DATABASE_URL in .env to switch.
"""

import logging
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# If DATABASE_URL is set (e.g. Supabase PostgreSQL), use it.
# Otherwise fall back to local SQLite for development.
# Check both os.environ (local .env) and Streamlit secrets (cloud).
_DATABASE_URL = os.getenv("DATABASE_URL", "")
if not _DATABASE_URL:
    try:
        import streamlit as st
        _DATABASE_URL = st.secrets.get("DATABASE_URL", "")
    except Exception:
        pass
if _DATABASE_URL:
    _ENGINE = create_engine(_DATABASE_URL, echo=False)
    logger.info("Using remote database (PostgreSQL)")
else:
    _DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autotrader.db"))
    _ENGINE = create_engine(f"sqlite:///{_DB_PATH}", echo=False)
    logger.info("Using local database (SQLite)")

_Session = sessionmaker(bind=_ENGINE)

Base = declarative_base()


class Signal(Base):
    """Every composite signal computed by the analysis engine."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    technical_score = Column(Float)
    sentiment_score = Column(Float)
    composite_score = Column(Float)
    signal = Column(String)         # BUY, SELL, HOLD, etc.
    timestamp = Column(DateTime, default=datetime.now)


class Trade(Base):
    """Every paper trade executed via Alpaca."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    action = Column(String)         # BUY, SELL, or CLOSE
    price = Column(Float)
    qty = Column(Integer)
    order_id = Column(String)
    strategy = Column(String, default="swing")  # "swing" or "scalp"
    timestamp = Column(DateTime, default=datetime.now)


class ManualPrediction(Base):
    """Your own directional picks — for comparing human vs algorithm accuracy."""
    __tablename__ = "manual_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    direction = Column(String)      # UP, DOWN, or FLAT
    confidence = Column(Float)      # 0.0 to 1.0
    date = Column(DateTime, default=datetime.now)


class GoalProgress(Base):
    """Weekly and long-term goal tracking per strategy."""
    __tablename__ = "goal_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False)   # "swing", "scalp", or "overall"
    period = Column(String, nullable=False)      # "weekly" or "long_term"
    week_start = Column(String)                  # "2026-03-23" for weekly entries
    target = Column(Float)
    actual = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)


# Create all tables if they don't exist yet
Base.metadata.create_all(_ENGINE)


def log_signal(signal_dict: dict) -> None:
    """Write a signal record to the database."""
    try:
        session = _Session()
        record = Signal(
            ticker=signal_dict.get("ticker", ""),
            technical_score=signal_dict.get("technical_score"),
            sentiment_score=signal_dict.get("sentiment_score"),
            composite_score=signal_dict.get("composite_score"),
            signal=signal_dict.get("signal"),
            timestamp=datetime.fromisoformat(signal_dict["timestamp"])
            if "timestamp" in signal_dict else datetime.now(),
        )
        session.add(record)
        session.commit()
        session.close()
        logger.info("Logged signal: %s %s", signal_dict.get("ticker"), signal_dict.get("signal"))
    except Exception as e:
        logger.error("Failed to log signal: %s", e)


def log_trade(trade_dict: dict) -> None:
    """Write a trade record to the database."""
    try:
        session = _Session()
        record = Trade(
            ticker=trade_dict.get("ticker", ""),
            action=trade_dict.get("action"),
            price=trade_dict.get("price"),
            qty=trade_dict.get("qty"),
            order_id=trade_dict.get("order_id"),
            strategy=trade_dict.get("strategy", "swing"),
            timestamp=datetime.now(),
        )
        session.add(record)
        session.commit()
        session.close()
        logger.info("Logged trade: %s %s", trade_dict.get("action"), trade_dict.get("ticker"))
    except Exception as e:
        logger.error("Failed to log trade: %s", e)


def log_manual_prediction(pred_dict: dict) -> None:
    """Write a manual prediction record to the database."""
    try:
        session = _Session()
        record = ManualPrediction(
            ticker=pred_dict.get("ticker", ""),
            direction=pred_dict.get("direction"),
            confidence=pred_dict.get("confidence"),
            date=pred_dict.get("date", datetime.now()),
        )
        session.add(record)
        session.commit()
        session.close()
        logger.info("Logged prediction: %s %s", pred_dict.get("ticker"), pred_dict.get("direction"))
    except Exception as e:
        logger.error("Failed to log prediction: %s", e)


def get_recent_signals(limit: int = 20) -> list[dict]:
    """Fetch the most recent signal records. Returns a list of dicts."""
    try:
        session = _Session()
        records = session.query(Signal).order_by(Signal.timestamp.desc()).limit(limit).all()
        session.close()
        return [
            {
                "ticker": r.ticker,
                "technical_score": r.technical_score,
                "sentiment_score": r.sentiment_score,
                "composite_score": r.composite_score,
                "signal": r.signal,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]
    except Exception as e:
        logger.error("Failed to fetch signals: %s", e)
        return []


def get_recent_trades(limit: int = 20) -> list[dict]:
    """Fetch the most recent trade records. Returns a list of dicts."""
    try:
        session = _Session()
        records = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
        session.close()
        return [
            {
                "ticker": r.ticker,
                "action": r.action,
                "price": r.price,
                "qty": r.qty,
                "order_id": r.order_id,
                "strategy": r.strategy,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]
    except Exception as e:
        logger.error("Failed to fetch trades: %s", e)
        return []


def get_trades_by_strategy(strategy: str, limit: int = 50) -> list[dict]:
    """Fetch recent trades filtered by strategy ('swing' or 'scalp')."""
    try:
        session = _Session()
        records = (session.query(Trade)
                   .filter(Trade.strategy == strategy)
                   .order_by(Trade.timestamp.desc())
                   .limit(limit).all())
        session.close()
        return [
            {
                "ticker": r.ticker,
                "action": r.action,
                "price": r.price,
                "qty": r.qty,
                "strategy": r.strategy,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]
    except Exception as e:
        logger.error("Failed to fetch %s trades: %s", strategy, e)
        return []


def log_goal_progress(strategy: str, period: str, target: float, actual: float,
                      week_start: str = "") -> None:
    """Log weekly or long-term goal progress for a strategy."""
    try:
        session = _Session()
        record = GoalProgress(
            strategy=strategy,
            period=period,
            week_start=week_start,
            target=target,
            actual=actual,
        )
        session.add(record)
        session.commit()
        session.close()
        logger.info("Logged goal: %s %s — $%.2f / $%.2f", strategy, period, actual, target)
    except Exception as e:
        logger.error("Failed to log goal progress: %s", e)


def get_goal_progress(strategy: str, period: str) -> list[dict]:
    """Fetch goal progress records for a strategy and period."""
    try:
        session = _Session()
        records = (session.query(GoalProgress)
                   .filter(GoalProgress.strategy == strategy,
                           GoalProgress.period == period)
                   .order_by(GoalProgress.timestamp.desc())
                   .limit(10).all())
        session.close()
        return [
            {
                "strategy": r.strategy,
                "period": r.period,
                "week_start": r.week_start,
                "target": r.target,
                "actual": r.actual,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]
    except Exception as e:
        logger.error("Failed to fetch goal progress: %s", e)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Database Demo ===\n")
    print(f"Database path: {_DB_PATH}\n")

    # Log a sample signal
    sample_signal = {
        "ticker": "AAPL",
        "technical_score": 0.72,
        "sentiment_score": 0.61,
        "composite_score": 0.68,
        "signal": "BUY",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    log_signal(sample_signal)
    print(f"Logged sample signal: {sample_signal['ticker']} = {sample_signal['signal']}")

    # Log a sample trade
    sample_trade = {"ticker": "AAPL", "action": "BUY", "price": 250.0, "qty": 1}
    log_trade(sample_trade)
    print(f"Logged sample trade: {sample_trade['action']} {sample_trade['ticker']}")

    # Log a sample prediction
    sample_pred = {"ticker": "AAPL", "direction": "UP", "confidence": 0.75}
    log_manual_prediction(sample_pred)
    print(f"Logged sample prediction: {sample_pred['ticker']} {sample_pred['direction']}")

    # Read them back
    print(f"\nRecent signals: {len(get_recent_signals())} records")
    print(f"Recent trades:  {len(get_recent_trades())} records")
