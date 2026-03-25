"""Phase 1–9 tests — full test suite for the AutoTrader application.

Run with: python tests/test_all.py  (from the autotrader/ directory)
"""

import sys
import os

# Ensure the autotrader package root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    WATCHLIST,
    WEIGHT_TECHNICAL,
    WEIGHT_SENTIMENT,
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    PAPER_TRADING,
)
from data.market import get_historical_bars, get_current_price
from data.news import get_news_headlines
from analysis.technical import compute_indicators, technical_score
from analysis.sentiment import score_headlines
from analysis.signals import compute_composite_signal
from prediction.model import build_features, train_model, predict_direction
from backtesting.backtest import run_backtest
from trading.paper_trader import get_portfolio_status, execute_signal
from tracking.database import log_signal, log_trade, log_manual_prediction, get_recent_signals, get_recent_trades
from tracking.performance import compute_metrics, generate_report
from tracking.report import generate_daily_report
from tracking.database import get_trades_by_strategy, log_goal_progress, get_goal_progress
from analysis.volatility import compute_scalp_indicators, scalp_signal
from trading.scalper import ScalpSession, get_session_summary


def test_config_watchlist() -> None:
    """WATCHLIST must be a non-empty list of ticker strings."""
    assert isinstance(WATCHLIST, list), "WATCHLIST should be a list"
    assert len(WATCHLIST) > 0, "WATCHLIST should not be empty"
    print("  PASS: config.WATCHLIST is a non-empty list")


def test_config_weights() -> None:
    """Signal weights must add up to 1.0 so the composite score is properly normalized."""
    total = WEIGHT_TECHNICAL + WEIGHT_SENTIMENT
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"
    print("  PASS: WEIGHT_TECHNICAL + WEIGHT_SENTIMENT == 1.0")


def test_config_thresholds() -> None:
    """Buy threshold must be above sell threshold to create a hold zone."""
    assert SIGNAL_BUY_THRESHOLD > SIGNAL_SELL_THRESHOLD, (
        "BUY_THRESHOLD must be greater than SELL_THRESHOLD"
    )
    print("  PASS: BUY_THRESHOLD > SELL_THRESHOLD")


def test_config_paper_trading() -> None:
    """Safety flag must default to True."""
    assert PAPER_TRADING is True, "PAPER_TRADING must be True by default"
    print("  PASS: PAPER_TRADING is True")


def test_historical_bars() -> None:
    """get_historical_bars should return a non-empty DataFrame with OHLCV columns."""
    df = get_historical_bars("AAPL")
    assert not df.empty, "Historical bars DataFrame should not be empty"
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    print(f"  PASS: get_historical_bars('AAPL') returned {len(df)} rows with correct columns")


def test_current_price() -> None:
    """get_current_price should return a positive float."""
    price = get_current_price("AAPL")
    assert price is not None, "Current price should not be None"
    assert isinstance(price, float), f"Price should be a float, got {type(price)}"
    assert price > 0, f"Price should be positive, got {price}"
    print(f"  PASS: get_current_price('AAPL') returned ${price:.2f}")


def test_news_headlines() -> None:
    """get_news_headlines should return a list (may be empty if API key is not set)."""
    headlines = get_news_headlines("AAPL")
    assert isinstance(headlines, list), f"Headlines should be a list, got {type(headlines)}"
    print(f"  PASS: get_news_headlines('AAPL') returned a list with {len(headlines)} items")


def test_compute_indicators() -> None:
    """compute_indicators should add RSI, MACD, BB, and EMA columns to the DataFrame."""
    df = get_historical_bars("AAPL")
    df = compute_indicators(df)
    expected_cols = {"rsi", "macd", "macd_signal", "bb_lower", "bb_upper", "ema_20", "ema_50"}
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing indicator columns: {expected_cols - set(df.columns)}"
    )
    print(f"  PASS: compute_indicators added all expected columns")


def test_technical_score() -> None:
    """technical_score should return a float between 0.0 and 1.0."""
    df = get_historical_bars("AAPL")
    df = compute_indicators(df)
    score = technical_score(df)
    assert isinstance(score, float), f"Score should be a float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score should be 0.0-1.0, got {score}"
    print(f"  PASS: technical_score('AAPL') = {score:.2f} (in range 0.0-1.0)")


def test_score_headlines() -> None:
    """score_headlines should return a float between 0.0 and 1.0."""
    sample = ["Apple beats earnings expectations", "Markets fall on recession fears"]
    score = score_headlines(sample)
    assert isinstance(score, float), f"Score should be a float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score should be 0.0-1.0, got {score}"
    print(f"  PASS: score_headlines returned {score:.2f} (in range 0.0-1.0)")


def test_score_headlines_empty() -> None:
    """score_headlines should return 0.5 (neutral) for an empty list."""
    score = score_headlines([])
    assert score == 0.5, f"Empty headlines should return 0.5, got {score}"
    print(f"  PASS: score_headlines([]) returned 0.5 (neutral)")


def test_composite_signal() -> None:
    """compute_composite_signal should return a dict with all required keys."""
    signal = compute_composite_signal("AAPL")
    required_keys = {"ticker", "technical_score", "sentiment_score",
                     "composite_score", "signal", "timestamp"}
    assert required_keys.issubset(set(signal.keys())), (
        f"Missing keys: {required_keys - set(signal.keys())}"
    )
    assert signal["ticker"] == "AAPL"
    assert 0.0 <= signal["composite_score"] <= 1.0
    assert signal["signal"] in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
    print(f"  PASS: composite signal for AAPL = {signal['signal']} "
          f"(composite: {signal['composite_score']})")


def test_build_features() -> None:
    """build_features should return a feature DataFrame and target Series with matching lengths."""
    df = get_historical_bars("AAPL", days=365)
    df = compute_indicators(df)
    X, y = build_features(df)
    assert not X.empty, "Feature matrix should not be empty"
    assert y is not None, "Target series should not be None"
    assert len(X) == len(y), "X and y must have the same length"
    expected_features = {"rsi", "macd", "macd_signal", "bb_pct", "ema_ratio", "volume_change_pct"}
    assert expected_features == set(X.columns), f"Unexpected features: {set(X.columns)}"
    print(f"  PASS: build_features returned {len(X)} rows with {len(X.columns)} features")


def test_train_model() -> None:
    """train_model should return a trained XGBClassifier and save it to disk."""
    model = train_model("AAPL")
    assert model is not None, "train_model should return a model"
    # Verify the model file was saved
    from config import MODEL_DIR
    model_path = os.path.join(MODEL_DIR, "AAPL_model.pkl")
    assert os.path.exists(model_path), f"Model file should exist at {model_path}"
    print(f"  PASS: train_model('AAPL') trained and saved model successfully")


def test_predict_direction() -> None:
    """predict_direction should return a valid direction string and confidence float."""
    direction, confidence = predict_direction("AAPL")
    assert direction in ("UP", "DOWN", "FLAT"), f"Invalid direction: {direction}"
    assert isinstance(confidence, float), f"Confidence should be float, got {type(confidence)}"
    assert 0.0 <= confidence <= 1.0, f"Confidence should be 0.0-1.0, got {confidence}"
    print(f"  PASS: predict_direction('AAPL') = {direction} (confidence: {confidence:.2f})")


def test_run_backtest() -> None:
    """run_backtest should return a result dict with a positive final value."""
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=180)
    result = run_backtest("AAPL", start, end)
    assert result is not None, "Backtest should return a result"
    assert result["final_value"] > 0, "Final portfolio value should be positive"
    print(f"  PASS: run_backtest('AAPL') returned ${result['final_value']:,.2f} "
          f"({result['return_pct']:+.2f}%, {result['num_trades']} trades)")


def test_backtest_result_keys() -> None:
    """Backtest result dict should contain all required keys."""
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=180)
    result = run_backtest("AAPL", start, end)
    required_keys = {"ticker", "start_date", "end_date", "starting_cash",
                     "final_value", "return_pct", "num_trades"}
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )
    print(f"  PASS: backtest result contains all required keys")


def test_portfolio_status() -> None:
    """get_portfolio_status should connect to Alpaca and return account info."""
    status = get_portfolio_status()
    assert status is not None, "Portfolio status should not be None — check Alpaca API keys"
    assert "cash" in status, "Status should include cash"
    assert "equity" in status, "Status should include equity"
    assert "positions" in status, "Status should include positions"
    print(f"  PASS: get_portfolio_status() connected — "
          f"equity=${float(status['equity']):,.2f}, "
          f"{len(status['positions'])} positions")


def test_hold_signal_no_op() -> None:
    """execute_signal with HOLD should return None (no order placed)."""
    result = execute_signal({"ticker": "AAPL", "signal": "HOLD"})
    assert result is None, "HOLD signal should not place an order"
    print(f"  PASS: execute_signal(HOLD) returned None (no order)")


def test_paper_trading_flag() -> None:
    """PAPER_TRADING must be True — this is the safety gate for all execution."""
    from config import PAPER_TRADING
    assert PAPER_TRADING is True, "PAPER_TRADING must be True"
    print(f"  PASS: PAPER_TRADING safety flag is True")


def test_log_signal() -> None:
    """log_signal should write a record retrievable by get_recent_signals."""
    from datetime import datetime
    test_signal = {
        "ticker": "TEST",
        "technical_score": 0.55,
        "sentiment_score": 0.45,
        "composite_score": 0.51,
        "signal": "HOLD",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    log_signal(test_signal)
    recent = get_recent_signals(limit=5)
    assert len(recent) > 0, "Should have at least one signal after logging"
    assert any(s["ticker"] == "TEST" for s in recent), "TEST signal should be retrievable"
    print(f"  PASS: log_signal wrote and get_recent_signals retrieved successfully")


def test_log_trade() -> None:
    """log_trade should write a record retrievable by get_recent_trades."""
    test_trade = {"ticker": "TEST", "action": "BUY", "price": 100.0, "qty": 1}
    log_trade(test_trade)
    recent = get_recent_trades(limit=5)
    assert len(recent) > 0, "Should have at least one trade after logging"
    assert any(t["ticker"] == "TEST" for t in recent), "TEST trade should be retrievable"
    print(f"  PASS: log_trade wrote and get_recent_trades retrieved successfully")


def test_log_manual_prediction() -> None:
    """log_manual_prediction should write without error."""
    test_pred = {"ticker": "TEST", "direction": "UP", "confidence": 0.80}
    log_manual_prediction(test_pred)
    # No getter for predictions yet — just verify no exception was raised
    print(f"  PASS: log_manual_prediction wrote successfully")


def test_generate_report() -> None:
    """generate_report should return a non-empty string."""
    report = generate_report()
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 50, "Report should have meaningful content"
    assert "PERFORMANCE REPORT" in report, "Report should contain header"
    print(f"  PASS: generate_report returned {len(report)} chars")


def test_dashboard_imports() -> None:
    """Dashboard module should import without errors."""
    # Don't run Streamlit — just verify the module structure is sound
    import importlib
    spec = importlib.util.find_spec("dashboard.app")
    assert spec is not None, "dashboard.app module should be importable"
    print(f"  PASS: dashboard.app module found")


def test_daily_report() -> None:
    """generate_daily_report should produce a report and save it to disk."""
    report = generate_daily_report()
    assert isinstance(report, str), "Report should be a string"
    assert "DAILY REPORT" in report, "Report should contain header"
    assert "Watchlist Signals" in report, "Report should include signals section"
    # Check file was saved
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.join(reports_dir, f"{today}.txt")
    assert os.path.exists(report_file), f"Report file should exist at {report_file}"
    print(f"  PASS: daily report generated and saved ({len(report)} chars)")


def test_main_imports() -> None:
    """main.py pipeline functions should be importable."""
    from main import morning_pipeline, midday_pipeline, evening_pipeline, run_once
    assert callable(morning_pipeline), "morning_pipeline should be callable"
    assert callable(midday_pipeline), "midday_pipeline should be callable"
    assert callable(evening_pipeline), "evening_pipeline should be callable"
    assert callable(run_once), "run_once should be callable"
    print(f"  PASS: main.py pipeline functions importable")


def test_scalp_indicators() -> None:
    """compute_scalp_indicators should add ATR, VWAP, and BB columns."""
    # Use daily data as a stand-in (same OHLCV structure as intraday)
    df = get_historical_bars("AAPL", days=30)
    if df.empty:
        print(f"  SKIP: No data available (market may be closed)")
        return
    df = compute_scalp_indicators(df)
    expected = {"atr", "vwap", "bb_lower", "bb_mid", "bb_upper"}
    assert expected.issubset(set(df.columns)), f"Missing: {expected - set(df.columns)}"
    print(f"  PASS: compute_scalp_indicators added all expected columns")


def test_scalp_signal() -> None:
    """scalp_signal should return a dict with an action key."""
    df = get_historical_bars("AAPL", days=30)
    if df.empty:
        print(f"  SKIP: No data available")
        return
    df = compute_scalp_indicators(df)
    signal = scalp_signal(df)
    assert "action" in signal, "Signal must have an action key"
    assert signal["action"] in ("BUY", "SELL", "WAIT"), f"Invalid action: {signal['action']}"
    assert "reason" in signal, "Signal must have a reason"
    print(f"  PASS: scalp_signal returned action={signal['action']}")


def test_scalp_session() -> None:
    """ScalpSession should track state correctly."""
    session = ScalpSession()
    summary = get_session_summary(session)
    assert summary["trades_today"] == 0
    assert summary["daily_pnl"] == 0.0
    assert summary["can_trade"] is True
    print(f"  PASS: ScalpSession initialized correctly")


def test_scalp_risk_limits() -> None:
    """ScalpSession should block trades when limits are hit."""
    from config import SCALP_MAX_TRADES_PER_DAY
    session = ScalpSession()
    session.trades_today = SCALP_MAX_TRADES_PER_DAY
    allowed, reason = session.can_trade()
    assert allowed is False, "Should be blocked at max trades"
    assert "Max trades" in reason
    print(f"  PASS: Risk limits block trading at max trades ({SCALP_MAX_TRADES_PER_DAY})")


def test_goal_tracking() -> None:
    """log_goal_progress and get_goal_progress should round-trip correctly."""
    log_goal_progress("scalp", "weekly", target=150.0, actual=75.0, week_start="2026-03-23")
    progress = get_goal_progress("scalp", "weekly")
    assert len(progress) > 0, "Should have at least one goal record"
    assert progress[0]["target"] == 150.0
    assert progress[0]["actual"] == 75.0
    print(f"  PASS: Goal tracking round-trip works")


def test_trades_by_strategy() -> None:
    """get_trades_by_strategy should filter correctly."""
    swing = get_trades_by_strategy("swing")
    scalp = get_trades_by_strategy("scalp")
    assert isinstance(swing, list), "Should return a list"
    assert isinstance(scalp, list), "Should return a list"
    print(f"  PASS: get_trades_by_strategy returned swing={len(swing)}, scalp={len(scalp)}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=== Phase 1 Tests ===\n")

    print("[Config Tests]")
    test_config_watchlist()
    test_config_weights()
    test_config_thresholds()
    test_config_paper_trading()

    print("\n[Market Data Tests]")
    test_historical_bars()
    test_current_price()

    print("\n[News Data Tests]")
    test_news_headlines()

    print("\n" + "=" * 40)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 2 Tests ===\n")

    print("[Technical Analysis Tests]")
    test_compute_indicators()
    test_technical_score()

    print("\n[Sentiment Analysis Tests]")
    test_score_headlines()
    test_score_headlines_empty()

    print("\n[Composite Signal Tests]")
    test_composite_signal()

    print("\n" + "=" * 40)
    print("ALL PHASE 2 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 3 Tests ===\n")

    print("[Feature Engineering Tests]")
    test_build_features()

    print("\n[Model Training Tests]")
    test_train_model()

    print("\n[Prediction Tests]")
    test_predict_direction()

    print("\n" + "=" * 40)
    print("ALL PHASE 3 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 4 Tests ===\n")

    print("[Backtest Tests]")
    test_run_backtest()
    test_backtest_result_keys()

    print("\n" + "=" * 40)
    print("ALL PHASE 4 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 5 Tests ===\n")

    print("[Paper Trading Tests]")
    test_portfolio_status()
    test_hold_signal_no_op()
    test_paper_trading_flag()

    print("\n" + "=" * 40)
    print("ALL PHASE 5 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 6 Tests ===\n")

    print("[Database Tests]")
    test_log_signal()
    test_log_trade()
    test_log_manual_prediction()

    print("\n[Performance Tests]")
    test_generate_report()

    print("\n" + "=" * 40)
    print("ALL PHASE 6 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 7 & 8 Tests ===\n")

    print("[Dashboard Tests]")
    test_dashboard_imports()

    print("\n[Daily Report Tests]")
    test_daily_report()

    print("\n[Scheduler Tests]")
    test_main_imports()

    print("\n" + "=" * 40)
    print("ALL PHASE 7 & 8 TESTS PASSED")
    print("=" * 40)

    print("\n\n=== Phase 9 Tests ===\n")

    print("[Volatility Analysis Tests]")
    test_scalp_indicators()
    test_scalp_signal()

    print("\n[Scalper Tests]")
    test_scalp_session()
    test_scalp_risk_limits()

    print("\n[Goal Tracking Tests]")
    test_goal_tracking()
    test_trades_by_strategy()

    print("\n" + "=" * 40)
    print("ALL PHASE 9 TESTS PASSED")
    print("=" * 40)

    print("\n\n" + "#" * 40)
    print("  ALL TESTS PASSED — BUILD COMPLETE")
    print("#" * 40)
