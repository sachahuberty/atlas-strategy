"""Tests for buckets.py: bucket-level allocation building blocks
(stage 11, Tier 2)."""

import pandas as pd
import pytest

from atlas import buckets

CLASS_BUCKET = pd.Series(
    {
        "A": "equity",
        "B": "equity",
        "C": "fixed_income",
        "D": "cash",
        "E": "real_estate",
    }
)


def test_bucket_returns_is_equal_weighted_mean_within_bucket():
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.0, 0.03],
            "B": [0.03, 0.00, 0.01, 0.02, -0.01],
            "C": [0.001] * 5,
            "D": [0.0001] * 5,
        },
        index=dates,
    )
    result = buckets.bucket_returns(
        returns, CLASS_BUCKET, ["equity", "fixed_income", "cash"]
    )
    expected_equity = (returns["A"] + returns["B"]) / 2
    pd.testing.assert_series_equal(
        result["equity"], expected_equity, check_names=False
    )
    pd.testing.assert_series_equal(
        result["fixed_income"], returns["C"], check_names=False
    )


def test_bucket_returns_omits_a_bucket_with_no_members_present():
    dates = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
    result = buckets.bucket_returns(returns, CLASS_BUCKET, ["equity", "cash"])
    assert "cash" not in result.columns
    assert "equity" in result.columns


def test_expand_bucket_weights_splits_equally_within_bucket():
    bucket_weights = pd.Series({"equity": 0.6, "fixed_income": 0.4})
    result = buckets.expand_bucket_weights(bucket_weights, CLASS_BUCKET)

    assert result["A"] == pytest.approx(0.3)
    assert result["B"] == pytest.approx(0.3)
    assert result["C"] == pytest.approx(0.4)
    assert result.sum() == pytest.approx(1.0)


def test_expand_bucket_weights_zero_for_bucket_not_present():
    # "cash" and "real_estate" are not keys in bucket_weights at all
    # (e.g. a bucket excluded from the bucket-level universe entirely).
    bucket_weights = pd.Series({"equity": 1.0})
    result = buckets.expand_bucket_weights(bucket_weights, CLASS_BUCKET)

    assert result["D"] == 0.0
    assert result["E"] == 0.0
    assert result.sum() == pytest.approx(1.0)


def test_expand_bucket_weights_covers_every_ticker_in_class_bucket():
    bucket_weights = pd.Series({"equity": 0.5, "cash": 0.5})
    result = buckets.expand_bucket_weights(bucket_weights, CLASS_BUCKET)

    assert set(result.index) == set(CLASS_BUCKET.index)
    assert not result.isna().any()
