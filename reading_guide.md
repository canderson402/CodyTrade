# AutoTrader – Plain English Reading Guide
*What are we building, what are we tracking, and why?*

---

## 🧠 What Is This App?

- A **bot that watches stocks** for you every morning
- It reads price data + today's news, then decides: Buy, Sell, or Hold
- It trades with **fake money first** (called "paper trading") so nothing is at risk
- It also logs **your own predictions** so you can see if you or the AI is smarter over time
- Think of it as: **a research lab for testing trading ideas, not a get-rich scheme**

---

## 📦 The Tech Stack (Plain English)

| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **Python** | The programming language | Simple, has every library we need |
| **Alpaca** | A free brokerage with an API | Paper trade + get live prices |
| **yfinance** | Pulls historical stock data | Free, easy, good for learning |
| **Finnhub** | News headlines for any stock | Free API, easy to use |
| **pandas-ta** | Math library for stock indicators | Calculates RSI, MACD, etc. for us |
| **FinBERT** | An AI trained to read financial news | Tells us if a headline is good or bad news |
| **XGBoost** | A simple ML prediction model | Guesses if tomorrow's price goes up or down |
| **Backtrader** | Simulates trades on old data | Tests if our strategy would have made money historically |
| **SQLite** | A tiny local database | Saves all our signals and trades to a file |
| **Streamlit** | Turns Python into a web app | Our visual dashboard, no web dev needed |
| **APScheduler** | A Python alarm clock | Runs our pipeline automatically every morning |

> 🔑 **Keywords to ask about:** `API`, `paper trading`, `DataFrame`, `backtesting`, `scheduler`

---

## 📊 What Are We Tracking & Why?

### Price Indicators (the "Technical" score)
These are math formulas applied to price history. Traders have used these for decades.

- **RSI** (Relative Strength Index)
  - Measures if a stock is *overbought* (probably due for a drop) or *oversold* (might bounce back)
  - Score 0–100. Below 30 = potentially good buy. Above 70 = might be overpriced.
  - 🔑 Ask about: `momentum indicators`, `mean reversion`

- **MACD** (Moving Average Convergence Divergence)
  - Compares two moving averages to spot when a trend is changing direction
  - When the MACD line crosses above the signal line = potential uptrend starting
  - 🔑 Ask about: `moving averages`, `trend following`

- **Bollinger Bands**
  - A price "envelope" that shows normal vs extreme price movement
  - Price touching the lower band = might be a buying opportunity
  - 🔑 Ask about: `volatility`, `standard deviation`

- **EMA** (Exponential Moving Average)
  - A smoothed average of recent prices that reacts faster than a simple average
  - Price above EMA = generally in an uptrend
  - 🔑 Ask about: `moving average crossover`

### News Sentiment (the "Sentiment" score)
- We pull today's headlines for each stock from Finnhub
- **FinBERT** reads each headline and scores it: Positive / Neutral / Negative
- We average those scores into one number for the day
- Example: "Apple crushes earnings" = positive. "SEC investigates Tesla" = negative.
- 🔑 Ask about: `NLP`, `BERT`, `sentiment analysis`, `transformer models`

### ML Prediction (the "AI" layer)
- **XGBoost** is trained on 1 year of historical indicator data
- It learns patterns like "when RSI is at X and MACD does Y, price usually goes up next day"
- It outputs: UP, DOWN, or FLAT with a confidence percentage
- This isn't magic — it's pattern matching on past data. It will be wrong sometimes.
- 🔑 Ask about: `supervised learning`, `classification models`, `feature engineering`

---

## 🎯 How a Signal Gets Made

Every morning for each stock in your watchlist:

```
1. Fetch last 90 days of price data
        ↓
2. Calculate RSI, MACD, Bollinger Bands, EMA
        ↓
3. Compute Technical Score (0.0 to 1.0)
        ↓
4. Fetch today's news headlines (up to 20)
        ↓
5. Run FinBERT on each headline
        ↓
6. Compute Sentiment Score (0.0 to 1.0)
        ↓
7. Weighted average → Composite Score
   (60% Technical + 40% Sentiment)
        ↓
8. Score > 0.65 → BUY
   Score < 0.35 → SELL
   In between   → HOLD
        ↓
9. Log it. Execute paper trade. Done.
```

---

## 🧪 Testing Yourself vs the Algorithm

- We save a table called `manual_predictions` in the database
- Every day you can enter: *"I think AAPL goes UP tomorrow, confidence 70%"*
- At end of week, the performance report shows:
  - **Your picks:** win rate, average gain
  - **Algorithm picks:** win rate, average gain
  - **Just holding SPY:** for comparison (the "did anything beat the market?" check)
- This is the most educational part — you'll quickly see where intuition beats rules and where it doesn't
- 🔑 Ask about: `alpha`, `benchmark comparison`, `Sharpe ratio`

---

## 🗓️ What Happens Every Day (Automated)

**9:00 AM (before market opens)**
- Fetch news & prices
- Compute all signals
- Execute paper trades

**4:30 PM (after market closes)**
- Compare today's signals to what actually happened
- Update performance log

**Every Monday morning**
- Retrain the XGBoost model on the latest data

---

## 🚦 The Two Modes

| Mode | What It Does | Risk |
|------|-------------|------|
| **Paper Trading** | Fake money, real prices | Zero — it's a simulation |
| **Live Trading** | Real money, real execution | Only go here after months of paper validation |

We stay in Paper Trading mode until the system has proven itself over many weeks.

---

## 💡 The Big Picture Goal

You're building a system where:
- **You define the rules** (which indicators matter, what thresholds to use)
- **The AI scores the news** (FinBERT reads faster than you ever could)
- **The ML model spots patterns** (XGBoost finds non-obvious correlations)
- **You validate everything** against history before trusting it with real decisions

This is exactly how professional quant funds work — just at a much smaller, learnable scale.
