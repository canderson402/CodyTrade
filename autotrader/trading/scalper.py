"""Intraday scalp trader with built-in risk management.

Executes many small trades based on volatility signals. Every trade has a
stop-loss and take-profit set at entry. Circuit breakers prevent catastrophic
losses on bad days.
"""

import logging
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    PAPER_TRADING,
    SCALP_STOP_LOSS_ATR,
    SCALP_TAKE_PROFIT_ATR,
    SCALP_RISK_PER_TRADE_PCT,
    SCALP_MAX_TRADES_PER_DAY,
    SCALP_MAX_DAILY_LOSS,
    SCALP_MAX_CONCURRENT,
)
from tracking.database import log_trade

logger = logging.getLogger(__name__)


class ScalpSession:
    """Tracks state for one day of scalp trading. Enforces all risk limits."""

    def __init__(self) -> None:
        """Initialize a fresh daily session with zero trades and zero P&L."""
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.active_positions: dict[str, dict] = {}  # ticker -> position info
        self.date: str = datetime.now().strftime("%Y-%m-%d")

    def can_trade(self) -> tuple[bool, str]:
        """Check if all risk limits allow another trade. Returns (allowed, reason)."""
        if self.trades_today >= SCALP_MAX_TRADES_PER_DAY:
            return False, f"Max trades reached ({SCALP_MAX_TRADES_PER_DAY}/day)"

        if self.daily_pnl <= -SCALP_MAX_DAILY_LOSS:
            return False, f"Max daily loss hit (${SCALP_MAX_DAILY_LOSS:.2f})"

        if len(self.active_positions) >= SCALP_MAX_CONCURRENT:
            return False, f"Max concurrent positions ({SCALP_MAX_CONCURRENT})"

        return True, "OK"


def _get_client() -> TradingClient | None:
    """Create an Alpaca TradingClient for paper trading."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API keys not set")
        return None
    try:
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except Exception as e:
        logger.error("Failed to create Alpaca client: %s", e)
        return None


def _calculate_position_size(client: TradingClient, atr: float) -> int:
    """Calculate how many shares to buy based on risk-per-trade and ATR."""
    try:
        account = client.get_account()
        portfolio_value = float(account.equity)

        # Risk amount = portfolio × risk percentage
        risk_amount = portfolio_value * SCALP_RISK_PER_TRADE_PCT

        # Dollar risk per share = ATR × stop-loss multiplier
        risk_per_share = atr * SCALP_STOP_LOSS_ATR

        if risk_per_share <= 0:
            return 0

        # Shares = total risk / risk per share
        shares = int(risk_amount / risk_per_share)
        return max(1, shares)  # At least 1 share

    except Exception as e:
        logger.error("Position size calculation failed: %s", e)
        return 1


def execute_scalp(signal: dict, session: ScalpSession) -> dict | None:
    """Execute a scalp trade based on a volatility signal. Returns trade info or None."""
    # SAFETY GATE
    if not PAPER_TRADING:
        logger.critical("PAPER_TRADING is False — refusing to scalp")
        return None

    action = signal.get("action", "WAIT")
    if action == "WAIT":
        return None

    # Check all risk limits before proceeding
    allowed, reason = session.can_trade()
    if not allowed:
        logger.warning("Trade blocked: %s", reason)
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        ticker = signal.get("ticker", "")
        entry = signal.get("entry", 0)
        atr = signal.get("atr", 0)

        # Calculate position size based on portfolio risk
        qty = _calculate_position_size(client, atr)

        # Determine order side
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_request)

        # Calculate stop and target using config multipliers
        if action == "BUY":
            stop_loss = entry - (atr * SCALP_STOP_LOSS_ATR)
            take_profit = entry + (atr * SCALP_TAKE_PROFIT_ATR)
        else:
            stop_loss = entry + (atr * SCALP_STOP_LOSS_ATR)
            take_profit = entry - (atr * SCALP_TAKE_PROFIT_ATR)

        trade_info = {
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "order_id": str(order.id),
            "strategy": "scalp",
            "price": entry,
        }

        # Update session state
        session.trades_today += 1
        session.active_positions[ticker] = trade_info

        # Log to database
        log_trade(trade_info)

        logger.info("SCALP %s %s x%d @ $%.2f | SL: $%.2f | TP: $%.2f",
                     action, ticker, qty, entry, stop_loss, take_profit)

        return trade_info

    except Exception as e:
        logger.error("Scalp execution failed: %s", e)
        return None


def check_exits(session: ScalpSession) -> list[dict]:
    """Check active positions against stop-loss and take-profit. Returns list of closed trades."""
    client = _get_client()
    if client is None:
        return []

    closed = []
    try:
        positions = client.get_all_positions()
        pos_by_ticker = {p.symbol: p for p in positions}

        for ticker, trade in list(session.active_positions.items()):
            pos = pos_by_ticker.get(ticker)
            if pos is None:
                # Position was closed externally
                del session.active_positions[ticker]
                continue

            current_price = float(pos.current_price)
            entry = trade["entry"]
            stop_loss = trade["stop_loss"]
            take_profit = trade["take_profit"]

            should_close = False
            exit_reason = ""

            if trade["action"] == "BUY":
                if current_price <= stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price >= take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            else:  # SELL / short
                if current_price >= stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price <= take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"

            if should_close:
                try:
                    client.close_position(ticker)
                    pnl = (current_price - entry) * int(pos.qty)
                    if trade["action"] == "SELL":
                        pnl = -pnl

                    session.daily_pnl += pnl
                    del session.active_positions[ticker]

                    exit_info = {
                        "ticker": ticker,
                        "action": "CLOSE",
                        "reason": exit_reason,
                        "entry": entry,
                        "exit_price": current_price,
                        "pnl": round(pnl, 2),
                        "strategy": "scalp",
                        "price": current_price,
                        "qty": int(pos.qty),
                    }
                    log_trade(exit_info)
                    closed.append(exit_info)

                    logger.info("SCALP EXIT %s %s | entry=$%.2f exit=$%.2f | P&L: $%.2f",
                                exit_reason, ticker, entry, current_price, pnl)
                except Exception as e:
                    logger.error("Failed to close position %s: %s", ticker, e)

    except Exception as e:
        logger.error("Exit check failed: %s", e)

    return closed


def get_session_summary(session: ScalpSession) -> dict:
    """Get a summary of the current scalp session. Returns a status dict."""
    return {
        "date": session.date,
        "trades_today": session.trades_today,
        "daily_pnl": round(session.daily_pnl, 2),
        "active_positions": len(session.active_positions),
        "max_trades": SCALP_MAX_TRADES_PER_DAY,
        "max_daily_loss": SCALP_MAX_DAILY_LOSS,
        "can_trade": session.can_trade()[0],
        "status_reason": session.can_trade()[1],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Scalper Demo ===\n")

    session = ScalpSession()
    summary = get_session_summary(session)

    print(f"Date:              {summary['date']}")
    print(f"Trades today:      {summary['trades_today']} / {summary['max_trades']}")
    print(f"Daily P&L:         ${summary['daily_pnl']:.2f}")
    print(f"Active positions:  {summary['active_positions']}")
    print(f"Can trade:         {summary['can_trade']} ({summary['status_reason']})")

    # Test with a mock signal (no actual trade)
    print("\nTesting WAIT signal (should do nothing)...")
    result = execute_scalp({"action": "WAIT"}, session)
    print(f"Result: {result} (None = no trade, as expected)")
