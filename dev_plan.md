# AutoTrader – Full Development Plan
> Python-based autonomous stock analysis & paper trading platform for educational use.
> Prioritize readability, modularity, and simplicity over cleverness.

---

## Project Structure

```
autotrader/
├── main.py                  # Entry point — runs the full pipeline on a schedule
├── config.py                # All API keys, settings, thresholds in one place
├── requirements.txt
│
├── data/
│   ├── market.py            # Fetch live + historical price data (Alpaca + yfinance)
│   └── news.py              # Fetch financial news headlines (Finnhub)
│
├── analysis/
│   ├── technical.py         # RSI, MACD, Bollinger Bands, EMA via pandas-ta
│   ├── sentiment.py         # FinBERT sentiment scoring on news headlines
│   └── signals.py           # Combine technical + sentiment into a single score
│
├── prediction/
│   └── model.py             # XGBoost price-direction predictor (up/down/flat)
│
├── backtesting/
│   └── backtest.py          # Backtrader strategy that uses our signal engine
│
├── trading/
│   └── paper_trader.py      # Submit paper trades via Alpaca based on signals
│
├── tracking/
│   ├── database.py          # SQLite — log all signals, predictions, trades
│   ├── performance.py       # Win rate, P&L, Sharpe ratio via quantstats
│   └── report.py            # Daily summary report — market moves, signals, trades
│
└── dashboard/
    └── app.py               # Streamlit dashboard — live view of signals + trades
```

---

## Phase 1 — Foundation & Data Pipeline

**Goal:** Get clean price data and news flowing reliably before building anything on top.

### Task 1.1 — Project Setup
- Create the folder structure above
- Create `requirements.txt`:
  ```
  alpaca-py
  yfinance
  pandas
  pandas-ta
  finnhub-python
  transformers
  torch
  xgboost
  scikit-learn
  backtrader
  quantstats
  streamlit
  plotly
  apscheduler
  python-dotenv
  SQLAlchemy
  ```
- Create `.env` file for secrets (never commit this):
  ```
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
  FINNHUB_API_KEY=your_key
  ```
- Create `config.py` that loads from `.env` and defines:
  - `WATCHLIST` — list of ticker symbols to track, e.g. `["AAPL", "MSFT", "NVDA", "SPY"]`
  - `SIGNAL_BUY_THRESHOLD = 0.65`
  - `SIGNAL_SELL_THRESHOLD = 0.35`
  - `PAPER_TRADING = True`

### Task 1.2 — `data/market.py`
- Function `get_historical_bars(ticker, days=90)` — returns a pandas DataFrame with columns: `open, high, low, close, volume`
  - Use `yfinance` as primary source for simplicity
  - Add a fallback comment noting Alpaca can replace this for production
- Function `get_current_price(ticker)` — returns a single float, the latest price
- Keep all functions simple: one job each, return a DataFrame or float, raise clear errors

### Task 1.3 — `data/news.py`
- Function `get_news_headlines(ticker, days=1)` — returns a list of headline strings for the ticker from the last N days
  - Use Finnhub's free company news endpoint
  - Limit to 20 headlines max per call
  - Return plain strings only — sentiment scoring happens separately

---

## Phase 2 — Analysis Engine

**Goal:** Turn raw price data and news into a clean numeric signal between 0 and 1.

### Task 2.1 — `analysis/technical.py`
- Function `compute_indicators(df)` — takes a price DataFrame, returns same DataFrame with new columns added:
  - `rsi` — Relative Strength Index (14-period)
  - `macd`, `macd_signal` — MACD line and signal line
  - `bb_upper`, `bb_lower` — Bollinger Band upper and lower bounds
  - `ema_20`, `ema_50` — Exponential Moving Averages
  - Use `pandas-ta` for all calculations: `df.ta.rsi()`, `df.ta.macd()`, etc.
- Function `technical_score(df)` — takes an indicator DataFrame, returns a float 0.0–1.0
  - Score 0 = strong sell signals, 1 = strong buy signals
  - Logic example:
    - RSI < 30 → bullish +0.2, RSI > 70 → bearish -0.2
    - Price above EMA_50 → bullish +0.15
    - MACD > Signal → bullish +0.15
    - Price near BB lower band → bullish +0.1
  - Clamp final result between 0.0 and 1.0
  - Add comments explaining each rule — this is the "your rules" section

### Task 2.2 — `analysis/sentiment.py`
- Load FinBERT once at module level (not inside the function — it's slow to load):
  ```python
  from transformers import pipeline
  sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
  ```
- Function `score_headlines(headlines: list[str])` → returns a float 0.0–1.0
  - Run each headline through FinBERT
  - Map labels: `positive → 1.0`, `neutral → 0.5`, `negative → 0.0`
  - Return weighted average of all headline scores
  - If headlines list is empty, return 0.5 (neutral)

### Task 2.3 — `analysis/signals.py`
- Function `compute_composite_signal(ticker)` → returns a dict:
  ```python
  {
    "ticker": "AAPL",
    "technical_score": 0.72,
    "sentiment_score": 0.61,
    "composite_score": 0.68,   # weighted average
    "signal": "BUY",           # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    "timestamp": "2026-03-24T09:35:00"
  }
  ```
- Weights (defined in `config.py`, easy to tweak):
  - `WEIGHT_TECHNICAL = 0.60`
  - `WEIGHT_SENTIMENT = 0.40`
- Signal thresholds from `config.py`

---

## Phase 3 — ML Prediction Layer

**Goal:** Add a simple machine learning model that predicts next-day price direction.

### Task 3.1 — `prediction/model.py`
- Use XGBoost — it's fast, interpretable, and works well on tabular data
- Function `build_features(df)` — takes indicator DataFrame, returns feature matrix:
  - Features: `rsi, macd, macd_signal, bb_pct, ema_ratio (ema_20/ema_50), volume_change_pct`
  - Target: `1` if next day close > today close, `0` otherwise (added via `.shift(-1)`)
- Function `train_model(ticker)` — trains an XGBoost classifier on 1 year of daily data, saves model to `models/{ticker}_model.pkl`
- Function `predict_direction(ticker)` → returns `"UP"`, `"DOWN"`, or `"FLAT"` with a confidence float
- Train models once daily (scheduled in `main.py`)
- Keep the model simple — no deep learning yet, this is Phase 3

---

## Phase 4 — Backtesting

**Goal:** Prove the signal engine works on historical data before trading anything.

### Task 4.1 — `backtesting/backtest.py`
- Build a Backtrader strategy class `SignalStrategy` that:
  - On each daily bar: calls `compute_composite_signal()` for the current ticker
  - BUY if signal >= BUY_THRESHOLD and not already in position
  - SELL if signal <= SELL_THRESHOLD and in position
  - Uses a simple fixed position size: 10% of portfolio per trade
- Function `run_backtest(ticker, start_date, end_date)`:
  - Sets up Backtrader cerebro with $10,000 starting cash
  - Adds commission of 0.001 (0.1%) per trade
  - Runs and prints: final portfolio value, return %, number of trades
- Function `run_all_backtests()` — runs backtest for each ticker in WATCHLIST, prints comparison table

---

## Phase 5 — Paper Trading

**Goal:** Run the system live with fake money to validate real-time performance.

### Task 5.1 — `trading/paper_trader.py`
- Use Alpaca's paper trading endpoint (set `paper=True` in client)
- Function `execute_signal(signal_dict)`:
  - If signal is `BUY` or `STRONG_BUY` and no open position: submit market buy order
  - If signal is `SELL` or `STRONG_SELL` and position exists: submit market sell order
  - If `HOLD`: do nothing
  - All orders: `time_in_force = "day"` (cancelled if not filled by close)
- Function `get_portfolio_status()` → returns current positions, cash, and total equity
- Gate all real execution behind `if config.PAPER_TRADING:` — never remove this check

---

## Phase 6 — Tracking & Logging

**Goal:** Log everything so you can compare your predictions vs the AI's vs reality.

### Task 6.1 — `tracking/database.py`
- Use SQLite + SQLAlchemy (simple, no server needed)
- Three tables:
  1. `signals` — every signal computed: ticker, scores, composite, signal label, timestamp
  2. `trades` — every paper trade: ticker, action, price, quantity, timestamp
  3. `manual_predictions` — your own picks (insert manually or via a simple CLI input prompt): ticker, direction, confidence, date
- Functions: `log_signal(signal_dict)`, `log_trade(trade_dict)`, `log_manual_prediction(pred_dict)`

### Task 6.2 — `tracking/performance.py`
- Function `generate_report()` — uses `quantstats` to print:
  - Win rate, total return, Sharpe ratio, max drawdown
  - Comparison: your manual picks vs algorithm picks vs buy-and-hold SPY
- Run this weekly or on demand

---

## Phase 7 — Dashboard

**Goal:** A simple visual interface to see what's happening in real time.

### Task 7.1 — `dashboard/app.py`
- Build with Streamlit — minimal code, runs with `streamlit run dashboard/app.py`
- Page 1: **Signal Dashboard**
  - Dropdown to select ticker from WATCHLIST
  - Show current composite score as a gauge chart (Plotly)
  - Show technical score, sentiment score, and latest signal label
  - Table of last 10 news headlines with their FinBERT sentiment labels
- Page 2: **Portfolio**
  - Current positions from Alpaca paper account
  - Recent trades table from SQLite
- Page 3: **Performance**
  - Equity curve chart
  - Win rate, Sharpe ratio, return vs SPY
  - Your predictions vs algorithm predictions table

---

## Phase 8 — Automation (Scheduler)

**Goal:** The whole pipeline runs automatically every morning without you touching it.

### Task 8.1 — `main.py`
- Use `APScheduler` to schedule jobs:
  ```python
  # Every weekday at 9:00 AM CT (before market open)
  scheduler.add_job(morning_pipeline, 'cron', day_of_week='mon-fri', hour=9, minute=0)

  # Every weekday at 12:00 PM CT (midday refresh)
  scheduler.add_job(midday_pipeline, 'cron', day_of_week='mon-fri', hour=12, minute=0)

  # Every weekday at 4:30 PM CT (after close)
  scheduler.add_job(evening_pipeline, 'cron', day_of_week='mon-fri', hour=16, minute=30)
  ```
- `morning_pipeline()`:
  1. Fetch latest news for all tickers
  2. Compute all signals
  3. Retrain ML models if it's Monday
  4. Execute paper trades based on signals
  5. Log everything to database
- `midday_pipeline()`:
  1. Refresh news headlines for all tickers (catch breaking news since open)
  2. Recompute signals with updated data
  3. Update ML predictions with intraday context
  4. Log updated signals to database (no trades — avoids overtrading)
- `evening_pipeline()`:
  1. Log end-of-day prices
  2. Evaluate today's signals vs actual price movement
  3. Update performance metrics
  4. Generate and save daily report

### Task 8.2 — Daily Report (`tracking/report.py`)
- Function `generate_daily_report()` — produces a text summary including:
  - **Market overview**: SPY direction, overall market move for the day
  - **Watchlist signals**: each ticker's current score, signal label, and ML prediction
  - **Trade activity**: any paper trades executed today (ticker, action, price, quantity)
  - **Portfolio snapshot**: current positions, cash balance, total equity
  - **Prediction accuracy**: yesterday's signals vs what actually happened
- Output as a formatted text file saved to `reports/YYYY-MM-DD.txt`
- Also print to console so the scheduler logs capture it
- Called automatically by `evening_pipeline()` after all metrics are updated

---

## Implementation Notes for Claude Code

- **Keep functions small** — one job per function, max ~30 lines
- **Type hints everywhere** — `def get_news_headlines(ticker: str, days: int = 1) -> list[str]:`
- **Docstrings on every function** — one line describing what it does and what it returns
- **All config in `config.py`** — no magic numbers scattered through code
- **Fail gracefully** — wrap all API calls in try/except, log errors, return None or empty list rather than crashing
- **Comments on business logic** — explain *why* a rule exists, not just what it does
- **No global state** — pass data explicitly between functions
- **Test each module independently** — each file should have a simple `if __name__ == "__main__":` block that demos it works
