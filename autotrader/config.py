"""Central configuration for the AutoTrader application.

All API keys, thresholds, and tunable constants live here.
Nothing is hardcoded in other modules — they import from this file.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from os.environ (local .env) or Streamlit secrets (cloud)."""
    value = os.getenv(key, "")
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(key, default)
        except Exception:
            value = default
    return value


# --- API Keys (loaded from .env or Streamlit secrets, never hardcoded) ---
ALPACA_API_KEY: str = _get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str = _get_secret("ALPACA_SECRET_KEY")
FINNHUB_API_KEY: str = _get_secret("FINNHUB_API_KEY")

# --- Watchlist ---
# Swing watchlist — large-caps with strong trends and heavy news coverage
WATCHLIST: list[str] = ["NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "JPM", "XOM", "SPY"]

# --- Signal Thresholds ---
# Composite scores above BUY_THRESHOLD trigger a buy signal;
# scores below SELL_THRESHOLD trigger a sell signal.
# The gap between them creates a neutral "hold" zone to avoid over-trading.
# STRONG thresholds add conviction levels for position sizing in later phases.
SIGNAL_STRONG_BUY_THRESHOLD: float = 0.80
SIGNAL_BUY_THRESHOLD: float = 0.65
SIGNAL_SELL_THRESHOLD: float = 0.35
SIGNAL_STRONG_SELL_THRESHOLD: float = 0.20

# --- Signal Weighting ---
# How much each analysis component contributes to the composite score.
# Must add up to 1.0. Heavier technical weight because price action is
# more reliable on short timeframes than headline sentiment.
WEIGHT_TECHNICAL: float = 0.60
WEIGHT_SENTIMENT: float = 0.40

# --- Safety Flag ---
# Must be True until manually changed. Gates all trade execution
# so we never accidentally trade with real money.
PAPER_TRADING: bool = True

# --- Data Fetching ---
# How many calendar days of price history to pull for technical analysis.
# 90 days gives enough data for indicators like 50-day EMA to stabilize.
HISTORY_DAYS: int = 90

# Max news headlines to fetch per ticker per API call.
# Keeps API usage reasonable and prevents one noisy day from skewing sentiment.
MAX_HEADLINES: int = 20

# --- Technical Indicator Periods ---
# Standard periods used across the industry. Changing these affects
# how responsive each indicator is to recent price action.
RSI_PERIOD: int = 14          # 14-day RSI is the most widely used period
EMA_SHORT: int = 20           # Short-term trend — reacts faster to price changes
EMA_LONG: int = 50            # Long-term trend — smoother, confirms direction
BB_PERIOD: int = 20           # Bollinger Band lookback — matches the short EMA
BB_STD: float = 2.0           # 2 standard deviations captures ~95% of price action

# --- ML Prediction ---
# 1 year of daily data for training gives ~252 trading days — enough for
# XGBoost to learn seasonal and momentum patterns without overfitting.
TRAINING_DAYS: int = 365

# Directory where trained models are saved (one .pkl file per ticker)
MODEL_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

# If the model's top-class probability is below this threshold,
# the prediction is "FLAT" — the model isn't confident enough to call a direction.
FLAT_CONFIDENCE_THRESHOLD: float = 0.6

# --- Backtesting ---
# Starting capital for paper backtests
BACKTEST_CASH: float = 10_000.00

# Commission per trade — 0.1% is realistic for a retail broker
BACKTEST_COMMISSION: float = 0.001

# Fraction of portfolio value to risk per trade.
# 10% keeps position sizes conservative while still showing meaningful results.
BACKTEST_POSITION_PCT: float = 0.10

# =============================================================
# SCALPER (Day Trading) Configuration
# =============================================================

# --- Scalp Watchlist ---
# High-volume, liquid stocks that move enough intraday to scalp.
# Can overlap with WATCHLIST or be entirely different tickers.
# Scalp watchlist — high volume + high intraday volatility for quick trades
SCALP_WATCHLIST: list[str] = ["TSLA", "NVDA", "AMD", "SPY", "QQQ", "SMCI"]

# --- Intraday Data ---
# Bar interval for intraday analysis. "5Min" balances noise vs responsiveness.
# Options: "1Min", "5Min", "15Min", "30Min", "1Hour"
SCALP_BAR_INTERVAL: str = "5Min"

# How many bars of intraday history to fetch for indicator calculation.
# 100 bars of 5-min data = ~8 hours (covers a full trading day).
SCALP_LOOKBACK_BARS: int = 100

# --- Scalp Indicators ---
SCALP_ATR_PERIOD: int = 14         # ATR period — measures typical bar-to-bar movement
SCALP_BB_PERIOD: int = 20          # Bollinger Band period for intraday bands
SCALP_BB_STD: float = 2.0          # Standard deviations for intraday bands

# --- Entry / Exit Rules ---
# Stop-loss = entry price ± (ATR × this multiplier).
# 1.0 means your stop is one ATR away — tight enough to limit damage,
# wide enough to avoid getting stopped out by normal noise.
SCALP_STOP_LOSS_ATR: float = 1.0

# Take-profit = entry price ± (ATR × this multiplier).
# 1.5:1 reward-to-risk means you only need to win ~40% of trades to profit.
SCALP_TAKE_PROFIT_ATR: float = 1.5

# --- Risk Management ---
# Max % of portfolio to risk on any single scalp trade.
# 1% is the professional standard — small enough that even a string of
# losses won't blow up the account.
SCALP_RISK_PER_TRADE_PCT: float = 0.01

# Maximum number of scalp trades per day. Prevents overtrading on
# choppy days where signals fire constantly but none follow through.
SCALP_MAX_TRADES_PER_DAY: int = 10

# If the scalper loses this much in a single day, stop trading.
# Circuit breaker that prevents emotional revenge trading.
SCALP_MAX_DAILY_LOSS: float = 200.00

# Maximum number of positions the scalper can hold at once.
SCALP_MAX_CONCURRENT: int = 2

# --- Goal Tracking ---
# Weekly dollar target for scalp strategy (e.g., grind $150/week)
GOAL_SCALP_WEEKLY: float = 150.00

# Weekly dollar target for swing strategy
GOAL_SWING_WEEKLY: float = 200.00

# Long-term cumulative target (e.g., grow paper account by $5k)
GOAL_LONG_TERM: float = 5_000.00


if __name__ == "__main__":
    print("=== AutoTrader Config ===")
    print(f"Watchlist: {WATCHLIST}")
    print(f"Buy threshold: {SIGNAL_BUY_THRESHOLD}")
    print(f"Sell threshold: {SIGNAL_SELL_THRESHOLD}")
    print(f"Weights: technical={WEIGHT_TECHNICAL}, sentiment={WEIGHT_SENTIMENT}")
    print(f"Paper trading: {PAPER_TRADING}")
    print(f"History days: {HISTORY_DAYS}")
    print(f"Max headlines: {MAX_HEADLINES}")
    print(f"Alpaca key loaded: {'yes' if ALPACA_API_KEY else 'NO — set in .env'}")
    print(f"Finnhub key loaded: {'yes' if FINNHUB_API_KEY else 'NO — set in .env'}")
