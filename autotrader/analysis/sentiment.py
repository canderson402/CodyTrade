"""Score financial news headlines using FinBERT sentiment analysis.

FinBERT is a BERT model fine-tuned on financial text. It classifies headlines
as positive, negative, or neutral — we map those to a 0.0–1.0 score.

The model is loaded once at module level to avoid reloading on every call
(loading takes several seconds, inference is fast).
"""

import logging

logger = logging.getLogger(__name__)

# Load FinBERT once at import time — it's slow to initialize but fast to run.
# Wrapped in try/except so the rest of the app doesn't crash if
# transformers/torch aren't installed or the model can't download.
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
    )
    logger.info("FinBERT model loaded successfully")
except Exception as e:
    sentiment_pipeline = None
    logger.warning("Could not load FinBERT model: %s", e)


# Map FinBERT labels to numeric scores.
# Positive news → 1.0 (bullish), negative → 0.0 (bearish), neutral → 0.5 (no signal).
LABEL_SCORES: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}


def score_headlines(headlines: list[str]) -> float:
    """Score a list of headlines from 0.0 (bearish) to 1.0 (bullish). Returns 0.5 if empty or model unavailable."""
    if not headlines:
        # No news is neutral news — don't let missing data bias the signal
        return 0.5

    if sentiment_pipeline is None:
        logger.error("FinBERT not available — returning neutral score")
        return 0.5

    try:
        results = sentiment_pipeline(headlines, truncation=True)

        scores: list[float] = []
        for result in results:
            label = result["label"].lower()
            score = LABEL_SCORES.get(label, 0.5)
            scores.append(score)

        # Simple average — each headline gets equal weight.
        # A more sophisticated approach could weight by recency or source quality.
        return sum(scores) / len(scores)

    except Exception as e:
        logger.error("FinBERT inference failed: %s", e)
        return 0.5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Sentiment Analysis Demo ===\n")

    sample_headlines = [
        "Apple reports record quarterly revenue, beating analyst expectations",
        "SEC launches investigation into tech company accounting practices",
        "Markets close flat amid mixed economic data",
        "NVIDIA stock surges on strong AI chip demand",
        "Federal Reserve signals potential rate cuts later this year",
    ]

    print("Sample headlines:")
    for i, h in enumerate(sample_headlines, 1):
        print(f"  {i}. {h}")

    if sentiment_pipeline is not None:
        print("\nScoring each headline individually:")
        for h in sample_headlines:
            result = sentiment_pipeline(h, truncation=True)[0]
            print(f"  [{result['label']:>8s} {result['score']:.2f}] {h[:60]}...")

        overall = score_headlines(sample_headlines)
        print(f"\nOverall sentiment score: {overall:.2f}")
    else:
        print("\nFinBERT model not loaded — install transformers and torch.")
