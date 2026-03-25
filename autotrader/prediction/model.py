"""XGBoost price-direction predictor.

Trains a simple classifier on technical indicator features to predict whether
tomorrow's closing price will be higher (UP), lower (DOWN), or uncertain (FLAT).

This is pattern matching on historical data — not a crystal ball. It will be
wrong sometimes, which is why we validate with backtesting before trusting it.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING_DAYS, MODEL_DIR, FLAT_CONFIDENCE_THRESHOLD
from data.market import get_historical_bars
from analysis.technical import compute_indicators

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Build feature matrix and target from an indicator DataFrame. Returns (X, y) where y is None if no target is needed."""
    if df.empty:
        return pd.DataFrame(), None

    features = pd.DataFrame(index=df.index)

    # RSI — raw momentum oscillator value
    features["rsi"] = df["rsi"]

    # MACD and signal — trend momentum and its smoothed version
    features["macd"] = df["macd"]
    features["macd_signal"] = df["macd_signal"]

    # Bollinger Band %B — where price sits within the volatility envelope
    # 0.0 = at lower band, 1.0 = at upper band
    bb_range = df["bb_upper"] - df["bb_lower"]
    features["bb_pct"] = np.where(
        bb_range > 0,
        (df["close"] - df["bb_lower"]) / bb_range,
        0.5,  # If bands collapse (zero range), default to midpoint
    )

    # EMA ratio — short-term trend relative to long-term trend
    # > 1.0 means short EMA is above long EMA (bullish crossover zone)
    features["ema_ratio"] = np.where(
        df["ema_50"] > 0,
        df["ema_20"] / df["ema_50"],
        1.0,
    )

    # Volume change % — spikes in volume often precede big price moves
    features["volume_change_pct"] = df["volume"].pct_change()

    # Target: 1 if next day's close is higher than today's, 0 otherwise
    # shift(-1) looks one day into the future
    target = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows with NaN values (from indicators warming up and the last row's target)
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    return features, target


def train_model(ticker: str) -> XGBClassifier | None:
    """Train an XGBoost classifier on 1 year of daily data and save to disk. Returns the trained model."""
    try:
        # Fetch 1 year of data — more history gives the model more patterns to learn
        df = get_historical_bars(ticker, days=TRAINING_DAYS)
        if df.empty:
            logger.error("No data to train model for %s", ticker)
            return None

        df = compute_indicators(df)
        X, y = build_features(df)

        if X.empty or y is None or len(X) < 30:
            logger.error("Not enough training data for %s (%d rows)", ticker, len(X))
            return None

        # XGBoost with conservative hyperparameters to avoid overfitting on
        # only ~250 trading days of data. Shallow trees + low learning rate.
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)

        # Save trained model to models/ directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)

        return model

    except Exception as e:
        logger.error("Failed to train model for %s: %s", ticker, e)
        return None


def _load_model(ticker: str) -> XGBClassifier | None:
    """Load a previously trained model from disk. Returns None if not found."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        logger.warning("No saved model for %s — run train_model() first", ticker)
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error("Failed to load model for %s: %s", ticker, e)
        return None


def predict_direction(ticker: str) -> tuple[str, float]:
    """Predict next-day price direction. Returns (direction, confidence) where direction is 'UP', 'DOWN', or 'FLAT'."""
    model = _load_model(ticker)
    if model is None:
        # No model available — can't make a prediction
        return ("FLAT", 0.0)

    try:
        # Get latest data and build features for the most recent day
        df = get_historical_bars(ticker)
        if df.empty:
            return ("FLAT", 0.0)

        df = compute_indicators(df)
        X, _ = build_features(df)

        if X.empty:
            return ("FLAT", 0.0)

        # Predict on the latest row only
        latest_features = X.iloc[[-1]]
        probabilities = model.predict_proba(latest_features)[0]
        confidence = float(max(probabilities))

        # If the model isn't confident enough, call it FLAT rather than
        # making a weak directional bet
        if confidence < FLAT_CONFIDENCE_THRESHOLD:
            return ("FLAT", confidence)

        predicted_class = int(model.predict(latest_features)[0])
        direction = "UP" if predicted_class == 1 else "DOWN"
        return (direction, confidence)

    except Exception as e:
        logger.error("Prediction failed for %s: %s", ticker, e)
        return ("FLAT", 0.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ML Prediction Demo ===\n")

    ticker = "AAPL"
    print(f"Training XGBoost model for {ticker} (1 year of data)...")
    model = train_model(ticker)

    if model is not None:
        print(f"Model trained successfully.\n")

        # Show feature importances — which indicators matter most
        feature_names = ["rsi", "macd", "macd_signal", "bb_pct",
                         "ema_ratio", "volume_change_pct"]
        importances = model.feature_importances_
        print("Feature importances:")
        for name, imp in sorted(zip(feature_names, importances),
                                key=lambda x: x[1], reverse=True):
            print(f"  {name:>20s}: {imp:.3f}")

        print(f"\nPredicting next-day direction for {ticker}...")
        direction, confidence = predict_direction(ticker)
        print(f"  Direction:  {direction}")
        print(f"  Confidence: {confidence:.2f}")
    else:
        print("Model training failed.")
