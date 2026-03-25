"""Streamlit dashboard for AutoTrader — live view of signals, portfolio, and performance.

Run with: streamlit run dashboard/app.py  (from the autotrader/ directory)

Two roles:
  admin  — full access, can change settings and trigger actions
  viewer — read-only, can see all data but cannot modify anything
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import config
from dashboard.auth import login_gate, logout, is_admin, get_username
from analysis.signals import compute_composite_signal
from data.market import get_historical_bars
from data.news import get_news_headlines
from analysis.sentiment import sentiment_pipeline, LABEL_SCORES
from trading.paper_trader import get_portfolio_status
from tracking.database import (
    get_recent_signals, get_recent_trades, get_trades_by_strategy,
    get_goal_progress,
)
from tracking.performance import generate_report, compute_metrics


st.set_page_config(page_title="AutoTrader Dashboard", layout="wide")

# --- Authentication ---
if not login_gate():
    st.stop()

# --- Sidebar ---
st.sidebar.markdown(f"Logged in as **{get_username()}** ({st.session_state.get('role', '')})")
if st.sidebar.button("Log out"):
    logout()

# Only show Settings page to admins
pages = ["Signal Dashboard", "Day Trading", "Portfolio", "Performance", "Goals"]
if is_admin():
    pages.append("Settings")

page = st.sidebar.radio("Navigate", pages)

st.title("AutoTrader Dashboard")


def _gauge_chart(score: float, title: str) -> go.Figure:
    """Create a Plotly gauge chart for a 0-1 score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, config.SIGNAL_SELL_THRESHOLD], "color": "#ff4444"},
                {"range": [config.SIGNAL_SELL_THRESHOLD, config.SIGNAL_BUY_THRESHOLD], "color": "#ffdd44"},
                {"range": [config.SIGNAL_BUY_THRESHOLD, 1], "color": "#44bb44"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
    return fig


# ==========================================
# Page 1: Signal Dashboard (Swing Trading)
# ==========================================
if page == "Signal Dashboard":
    ticker = st.selectbox("Select Ticker", config.WATCHLIST)

    # Only admins can trigger signal computation (it calls external APIs)
    if is_admin():
        if st.button("Compute Signal", type="primary"):
            with st.spinner(f"Analyzing {ticker}..."):
                signal = compute_composite_signal(ticker)
            st.session_state["last_signal"] = signal
    else:
        st.info("Read-only mode. Ask an admin to compute fresh signals.")

    signal = st.session_state.get("last_signal")
    if signal:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(_gauge_chart(signal["technical_score"], "Technical"),
                           use_container_width=True)
        with col2:
            st.plotly_chart(_gauge_chart(signal["sentiment_score"], "Sentiment"),
                           use_container_width=True)
        with col3:
            st.plotly_chart(_gauge_chart(signal["composite_score"], "Composite"),
                           use_container_width=True)

        signal_label = signal["signal"]
        color = {"STRONG_BUY": "green", "BUY": "lightgreen", "HOLD": "orange",
                 "SELL": "salmon", "STRONG_SELL": "red"}.get(signal_label, "gray")
        st.markdown(f"### Signal: :{color}[{signal_label}]")
        st.caption(f"Computed at {signal['timestamp']}")

    # Headlines are read-only — safe for everyone
    st.subheader("Recent Headlines")
    headlines = get_news_headlines(ticker)
    if headlines and sentiment_pipeline is not None:
        results = sentiment_pipeline(headlines[:10], truncation=True)
        headline_data = []
        for h, r in zip(headlines[:10], results):
            label = r["label"].lower()
            headline_data.append({
                "Headline": h,
                "Sentiment": label.capitalize(),
                "Score": LABEL_SCORES.get(label, 0.5),
            })
        st.dataframe(pd.DataFrame(headline_data), use_container_width=True, hide_index=True)
    elif headlines:
        for h in headlines[:10]:
            st.write(f"- {h}")
    else:
        st.info("No recent headlines found.")


# ==========================================
# Page 2: Day Trading (Scalper) — read-only
# ==========================================
elif page == "Day Trading":
    st.subheader("Scalp Trading Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Trades/Day", config.SCALP_MAX_TRADES_PER_DAY)
    col2.metric("Risk Per Trade", f"{config.SCALP_RISK_PER_TRADE_PCT:.0%}")
    col3.metric("Stop Loss", f"{config.SCALP_STOP_LOSS_ATR}x ATR")
    col4.metric("Take Profit", f"{config.SCALP_TAKE_PROFIT_ATR}x ATR")

    st.subheader("Recent Scalp Trades")
    scalp_trades = get_trades_by_strategy("scalp", limit=20)
    if scalp_trades:
        st.dataframe(pd.DataFrame(scalp_trades), use_container_width=True, hide_index=True)
    else:
        st.info("No scalp trades yet. The scalper runs during market hours.")

    st.subheader("Scalp Watchlist")
    for ticker in config.SCALP_WATCHLIST:
        df = get_historical_bars(ticker, days=5)
        if not df.empty:
            st.line_chart(df["close"], height=150)
            st.caption(ticker)

    st.subheader("Scalp Performance")
    scalp_metrics = compute_metrics(scalp_trades)
    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{scalp_metrics['win_rate']}%")
    col2.metric("Total Scalps", scalp_metrics["total_trades"])
    col3.metric("W / L", f"{scalp_metrics.get('wins', 0)} / {scalp_metrics.get('losses', 0)}")


# ==========================================
# Page 3: Portfolio — read-only
# ==========================================
elif page == "Portfolio":
    st.subheader("Alpaca Paper Account")

    status = get_portfolio_status()
    if status:
        col1, col2, col3 = st.columns(3)
        col1.metric("Cash", f"${float(status['cash']):,.2f}")
        col2.metric("Equity", f"${float(status['equity']):,.2f}")
        col3.metric("Buying Power", f"${float(status['buying_power']):,.2f}")

        st.subheader("Open Positions")
        if status["positions"]:
            pos_data = []
            for p in status["positions"]:
                pos_data.append({
                    "Ticker": p["ticker"],
                    "Qty": p["qty"],
                    "Price": f"${float(p['current_price']):,.2f}",
                    "Value": f"${float(p['market_value']):,.2f}",
                    "P&L": f"${float(p['unrealized_pl']):,.2f}",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")
    else:
        st.error("Could not connect to Alpaca — check API keys in .env")

    st.subheader("Recent Swing Trades")
    swing_trades = get_trades_by_strategy("swing", limit=10)
    if swing_trades:
        st.dataframe(pd.DataFrame(swing_trades), use_container_width=True, hide_index=True)
    else:
        st.info("No swing trades yet.")

    st.subheader("Recent Scalp Trades")
    scalp_trades = get_trades_by_strategy("scalp", limit=10)
    if scalp_trades:
        st.dataframe(pd.DataFrame(scalp_trades), use_container_width=True, hide_index=True)
    else:
        st.info("No scalp trades yet.")


# ==========================================
# Page 4: Performance — read-only
# ==========================================
elif page == "Performance":
    st.subheader("Performance Report")

    trades = get_recent_trades(limit=100)
    metrics = compute_metrics(trades)

    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{metrics['win_rate']}%")
    col2.metric("Total Trades", metrics["total_trades"])
    col3.metric("Wins / Losses", f"{metrics.get('wins', 0)} / {metrics.get('losses', 0)}")

    st.subheader("Watchlist Price Charts")
    for ticker in config.WATCHLIST:
        df = get_historical_bars(ticker)
        if not df.empty:
            st.line_chart(df["close"], height=200)
            st.caption(ticker)

    st.subheader("Recent Signals")
    signals = get_recent_signals(limit=20)
    if signals:
        st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)
    else:
        st.info("No signals recorded yet.")

    with st.expander("Full Performance Report"):
        report = generate_report()
        st.code(report)


# ==========================================
# Page 5: Goals — read-only
# ==========================================
elif page == "Goals":
    st.subheader("Trading Goals")

    st.markdown("### Weekly Targets")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scalp Weekly Goal", f"${config.GOAL_SCALP_WEEKLY:,.2f}")
        scalp_progress = get_goal_progress("scalp", "weekly")
        if scalp_progress:
            latest = scalp_progress[0]
            pct = (latest["actual"] / latest["target"] * 100) if latest["target"] > 0 else 0
            st.progress(min(pct / 100, 1.0), text=f"${latest['actual']:,.2f} / ${latest['target']:,.2f}")
        else:
            st.progress(0.0, text="No data yet")

    with col2:
        st.metric("Swing Weekly Goal", f"${config.GOAL_SWING_WEEKLY:,.2f}")
        swing_progress = get_goal_progress("swing", "weekly")
        if swing_progress:
            latest = swing_progress[0]
            pct = (latest["actual"] / latest["target"] * 100) if latest["target"] > 0 else 0
            st.progress(min(pct / 100, 1.0), text=f"${latest['actual']:,.2f} / ${latest['target']:,.2f}")
        else:
            st.progress(0.0, text="No data yet")

    st.markdown("### Long-Term Target")
    st.metric("Overall Goal", f"${config.GOAL_LONG_TERM:,.2f}")
    overall_progress = get_goal_progress("overall", "long_term")
    if overall_progress:
        latest = overall_progress[0]
        pct = (latest["actual"] / latest["target"] * 100) if latest["target"] > 0 else 0
        st.progress(min(pct / 100, 1.0), text=f"${latest['actual']:,.2f} / ${latest['target']:,.2f}")
    else:
        st.progress(0.0, text="No data yet — goals update as trades accumulate")


# ==========================================
# Page 6: Settings — ADMIN ONLY
# ==========================================
elif page == "Settings":
    if not is_admin():
        st.error("Access denied. Admin privileges required.")
        st.stop()

    st.subheader("Configuration (Admin Only)")
    st.caption("Changes are applied for the current session. Edit config.py to make them permanent.")

    st.markdown("### Watchlists")
    new_watchlist = st.text_input("Swing Watchlist (comma-separated)", ", ".join(config.WATCHLIST))
    new_scalp_watchlist = st.text_input("Scalp Watchlist (comma-separated)", ", ".join(config.SCALP_WATCHLIST))

    st.markdown("### Signal Thresholds")
    col1, col2 = st.columns(2)
    new_buy = col1.slider("Buy Threshold", 0.5, 0.95, config.SIGNAL_BUY_THRESHOLD, 0.05)
    new_sell = col2.slider("Sell Threshold", 0.05, 0.5, config.SIGNAL_SELL_THRESHOLD, 0.05)

    st.markdown("### Signal Weights")
    new_tech_weight = st.slider("Technical Weight", 0.0, 1.0, config.WEIGHT_TECHNICAL, 0.05)
    st.caption(f"Sentiment Weight: {1.0 - new_tech_weight:.2f} (auto-calculated)")

    st.markdown("### Scalper Risk Management")
    col1, col2 = st.columns(2)
    new_risk_pct = col1.slider("Risk Per Trade %", 0.005, 0.05, config.SCALP_RISK_PER_TRADE_PCT, 0.005,
                                format="%.3f")
    new_max_trades = col2.number_input("Max Trades/Day", 1, 50, config.SCALP_MAX_TRADES_PER_DAY)
    new_max_loss = st.number_input("Max Daily Loss ($)", 50.0, 1000.0, config.SCALP_MAX_DAILY_LOSS, 25.0)

    st.markdown("### Goal Targets")
    col1, col2, col3 = st.columns(3)
    new_scalp_goal = col1.number_input("Scalp Weekly ($)", 0.0, 5000.0, config.GOAL_SCALP_WEEKLY, 25.0)
    new_swing_goal = col2.number_input("Swing Weekly ($)", 0.0, 5000.0, config.GOAL_SWING_WEEKLY, 25.0)
    new_long_goal = col3.number_input("Long-Term ($)", 0.0, 50000.0, config.GOAL_LONG_TERM, 500.0)

    if st.button("Apply Settings", type="primary"):
        config.WATCHLIST = [t.strip() for t in new_watchlist.split(",") if t.strip()]
        config.SCALP_WATCHLIST = [t.strip() for t in new_scalp_watchlist.split(",") if t.strip()]
        config.SIGNAL_BUY_THRESHOLD = new_buy
        config.SIGNAL_SELL_THRESHOLD = new_sell
        config.WEIGHT_TECHNICAL = new_tech_weight
        config.WEIGHT_SENTIMENT = round(1.0 - new_tech_weight, 2)
        config.SCALP_RISK_PER_TRADE_PCT = new_risk_pct
        config.SCALP_MAX_TRADES_PER_DAY = new_max_trades
        config.SCALP_MAX_DAILY_LOSS = new_max_loss
        config.GOAL_SCALP_WEEKLY = new_scalp_goal
        config.GOAL_SWING_WEEKLY = new_swing_goal
        config.GOAL_LONG_TERM = new_long_goal
        st.success("Settings applied for this session!")

    st.markdown("---")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    st.caption(f"Config file: `{config_path}`")
    st.caption("To make changes permanent, edit config.py directly.")
