"""Execute paper trades via Alpaca based on signal output.

All trade execution is gated behind config.PAPER_TRADING — if that flag
is ever False, no orders will be submitted. This is the safety net that
prevents accidental real-money trades.
"""

import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING

logger = logging.getLogger(__name__)

# Buy signals that trigger a purchase order
_BUY_SIGNALS = {"BUY", "STRONG_BUY"}

# Sell signals that trigger a sell order
_SELL_SIGNALS = {"SELL", "STRONG_SELL"}

# Default number of shares per trade — kept small for paper testing.
# A smarter approach (position sizing based on portfolio %) comes in later phases.
_DEFAULT_QTY = 1


def _get_client() -> TradingClient | None:
    """Create an Alpaca TradingClient. Returns None if keys are missing."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API keys not set — check your .env file")
        return None

    try:
        # paper=True forces the paper trading endpoint regardless of key type
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except Exception as e:
        logger.error("Failed to create Alpaca client: %s", e)
        return None


def _has_position(client: TradingClient, ticker: str) -> bool:
    """Check whether we currently hold shares of a ticker."""
    try:
        positions = client.get_all_positions()
        return any(p.symbol == ticker for p in positions)
    except Exception as e:
        logger.error("Failed to check positions: %s", e)
        return False


def execute_signal(signal_dict: dict) -> dict | None:
    """Submit a paper trade based on a signal dict. Returns order info or None."""
    # SAFETY GATE — never trade with real money unless this flag is manually changed
    if not PAPER_TRADING:
        logger.critical("PAPER_TRADING is False — refusing to execute. "
                        "Set PAPER_TRADING=True in config.py before trading.")
        return None

    ticker = signal_dict.get("ticker", "")
    signal = signal_dict.get("signal", "HOLD")

    # HOLD means do nothing — no order needed
    if signal == "HOLD":
        logger.info("Signal is HOLD for %s — no action taken", ticker)
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        if signal in _BUY_SIGNALS and not _has_position(client, ticker):
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=_DEFAULT_QTY,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,  # Cancel if not filled by close
            )
            order = client.submit_order(order_request)
            logger.info("BUY order submitted: %s x%d", ticker, _DEFAULT_QTY)
            return _order_to_dict(order, "BUY")

        elif signal in _SELL_SIGNALS and _has_position(client, ticker):
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=_DEFAULT_QTY,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(order_request)
            logger.info("SELL order submitted: %s x%d", ticker, _DEFAULT_QTY)
            return _order_to_dict(order, "SELL")

        else:
            # Signal says buy but we already hold, or sell but nothing to sell
            logger.info("No action for %s — signal=%s, position_held=%s",
                        ticker, signal, signal in _SELL_SIGNALS)
            return None

    except Exception as e:
        logger.error("Order execution failed for %s: %s", ticker, e)
        return None


def _order_to_dict(order, action: str) -> dict:
    """Convert an Alpaca order object to a simple dict for logging."""
    return {
        "order_id": str(order.id),
        "ticker": order.symbol,
        "action": action,
        "qty": str(order.qty),
        "status": str(order.status),
        "submitted_at": str(order.submitted_at),
    }


def get_portfolio_status() -> dict | None:
    """Get current positions, cash, and total equity from Alpaca. Returns a status dict or None."""
    client = _get_client()
    if client is None:
        return None

    try:
        account = client.get_account()
        positions = client.get_all_positions()

        position_list = [
            {
                "ticker": p.symbol,
                "qty": str(p.qty),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl),
                "current_price": str(p.current_price),
            }
            for p in positions
        ]

        return {
            "cash": str(account.cash),
            "equity": str(account.equity),
            "buying_power": str(account.buying_power),
            "positions": position_list,
        }

    except Exception as e:
        logger.error("Failed to get portfolio status: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Paper Trader Demo ===\n")

    status = get_portfolio_status()
    if status:
        print(f"Cash:         ${float(status['cash']):,.2f}")
        print(f"Equity:       ${float(status['equity']):,.2f}")
        print(f"Buying Power: ${float(status['buying_power']):,.2f}")
        print(f"Positions:    {len(status['positions'])}")
        for p in status["positions"]:
            print(f"  {p['ticker']}: {p['qty']} shares @ ${float(p['current_price']):,.2f} "
                  f"(P&L: ${float(p['unrealized_pl']):,.2f})")
    else:
        print("Could not connect to Alpaca — check your API keys in .env")

    # Demo a HOLD signal (safe no-op)
    print("\nTesting HOLD signal (should do nothing)...")
    result = execute_signal({"ticker": "AAPL", "signal": "HOLD"})
    print(f"Result: {result} (None = no order placed, as expected)")
