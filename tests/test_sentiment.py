"""Tests for sentiment.py: V4 news sentiment signal (stage 7)."""

import pandas as pd
import pytest

from atlas import sentiment

CFG = {
    "general": {"random_seed": 42},
    "sentiment": {
        "lag_days": 1,
        "max_view_magnitude": 0.01,
        "n_topics": 2,
        "request_timeout_seconds": 10,
        "user_agent": "ATLAS-test-bot/0.1",
    },
}


def test_preprocess_lowercases_removes_stopwords_and_lemmatizes():
    tokens = sentiment.preprocess("The Markets are Rallying strongly!")
    assert "the" not in tokens
    assert "are" not in tokens
    assert "market" in tokens
    assert "rally" in tokens


def test_score_sentiment_distinguishes_positive_and_negative():
    positive = sentiment.score_sentiment(
        "Stocks rally as earnings beat expectations, investors cheer"
    )
    negative = sentiment.score_sentiment(
        "Markets crash amid recession fears and mass layoffs"
    )
    assert positive > 0
    assert negative < 0


def test_tag_bucket_matches_expected_keywords():
    fixed_income = sentiment.tag_bucket("Treasury yields climb as Fed holds")
    commodity = sentiment.tag_bucket("Gold prices surge amid oil rally")
    real_estate = sentiment.tag_bucket("Mortgage rates weigh on housing")
    equity = sentiment.tag_bucket("S&P 500 and Nasdaq stocks rally")

    assert fixed_income == "fixed_income"
    assert commodity == "commodity"
    assert real_estate == "real_estate"
    assert equity == "equity"


def test_tag_bucket_none_when_no_keywords_match():
    headline = "Local bakery wins award for best croissant"
    assert sentiment.tag_bucket(headline) is None


def test_aggregate_sentiment_applies_lag_and_weekly_grouping():
    headlines = pd.DataFrame(
        {
            "title": [
                "Stocks rally on strong earnings",
                "Nasdaq surges as tech stocks jump",
                "Treasury yields fall as bonds rally",
            ],
            "published": [
                "Fri, 02 Jan 2026 12:00:00 -0000",
                "Fri, 02 Jan 2026 15:00:00 -0000",
                "Fri, 02 Jan 2026 09:00:00 -0000",
            ],
        }
    )
    weekly = sentiment.aggregate_sentiment(headlines, CFG)

    assert "equity" in weekly.columns
    assert "fixed_income" in weekly.columns
    # Lag of 1 day moves Fri Jan 2 -> Sat Jan 3, a different ISO week.
    expected_week = (
        pd.Timestamp("2026-01-02") + pd.Timedelta(days=1)
    ).to_period("W")
    assert expected_week in weekly.index


def test_aggregate_sentiment_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["title", "published"])
    result = sentiment.aggregate_sentiment(empty, CFG)
    assert result.empty


def test_aggregate_sentiment_drops_unclassified_and_undated_rows():
    headlines = pd.DataFrame(
        {
            "title": [
                "Local bakery wins award",  # no bucket match
                "Stocks rally on earnings",  # no published date
                "Bond yields rise sharply",
            ],
            "published": [
                "Fri, 02 Jan 2026 12:00:00 -0000",
                None,
                "Fri, 02 Jan 2026 12:00:00 -0000",
            ],
        }
    )
    weekly = sentiment.aggregate_sentiment(headlines, CFG)
    # Only the bond-yields row survives both filters.
    assert weekly.notna().to_numpy().sum() == 1
    assert "fixed_income" in weekly.columns


def test_sentiment_view_broadcasts_bucket_score_to_assets():
    class_bucket = pd.Series(
        {"ACWI": "equity", "AGG": "fixed_income", "BIL": "cash"}
    )
    bucket_scores = pd.Series({"equity": 0.5, "fixed_income": -0.5})

    view = sentiment.sentiment_view(bucket_scores, class_bucket, CFG)

    max_magnitude = CFG["sentiment"]["max_view_magnitude"]
    assert view["ACWI"] == pytest.approx(0.5 * max_magnitude)
    assert view["AGG"] == pytest.approx(-0.5 * max_magnitude)
    # No sentiment for "cash" -> zero, not NaN.
    assert view["BIL"] == 0.0


def test_build_lda_model_returns_expected_number_of_topics():
    documents = [
        ["stock", "market", "rally", "earnings"],
        ["bond", "yield", "treasury", "fed"],
        ["stock", "earnings", "rally", "buy"],
        ["bond", "treasury", "yield", "rate"],
    ]
    model, dictionary, corpus = sentiment.build_lda_model(documents, CFG)
    assert model.num_topics == CFG["sentiment"]["n_topics"]
    assert len(corpus) == len(documents)


def test_fetch_rss_returns_empty_on_robots_disallow(monkeypatch):
    monkeypatch.setattr(sentiment, "_robots_allowed", lambda url, ua: False)
    result = sentiment._fetch_rss("https://example.com/rss", CFG)
    assert result.empty
    assert list(result.columns) == [
        "title", "description", "link", "published", "source",
    ]


def test_scrape_headlines_skips_failed_sources_gracefully(monkeypatch):
    monkeypatch.setattr(sentiment, "_robots_allowed", lambda url, ua: False)
    result = sentiment.scrape_headlines(
        ["https://example.com/a", "https://example.com/b"], CFG
    )
    assert result.empty
