"""Shape/consistency tests for views.py (stage 8)."""

import numpy as np
import pandas as pd
import pytest

from atlas import views

TICKERS = ["SPY", "QQQ", "AGG", "GLD"]
CLASS_BUCKET = pd.Series(
    {
        "SPY": "equity",
        "QQQ": "equity",
        "AGG": "fixed_income",
        "GLD": "commodity",
    }
)
PRIOR = pd.Series({"SPY": 0.06, "QQQ": 0.07, "AGG": 0.02, "GLD": 0.03})
COV = pd.DataFrame(np.eye(4) * 0.04, index=TICKERS, columns=TICKERS)

CFG = {
    "modules": {
        "regime_view": True,
        "meanreversion_view": True,
        "technical_view": True,
        "sentiment_view": True,
        "momentum_view": True,
    },
    "black_litterman": {
        "tau": 0.05,
        "regime_view_magnitude": 0.03,
        "view_confidence": {
            "regime": 1.0,
            "meanreversion": 0.8,
            "technical": 0.5,
            "sentiment": 0.3,
            "momentum": 0.8,
        },
    },
}


def test_regime_view_risk_on_targets_equity_bucket():
    view_set = views.regime_view("risk_on", CLASS_BUCKET, PRIOR, CFG)
    assert view_set.labels == ["V1_regime_equity"]
    equity_prior = (PRIOR["SPY"] + PRIOR["QQQ"]) / 2
    assert view_set.q_rows[0] == pytest.approx(
        equity_prior + CFG["black_litterman"]["regime_view_magnitude"]
    )
    np.testing.assert_allclose(view_set.p_rows[0], [0.5, 0.5, 0.0, 0.0])


def test_regime_view_risk_off_targets_fixed_income_and_commodity():
    view_set = views.regime_view("risk_off", CLASS_BUCKET, PRIOR, CFG)
    assert set(view_set.labels) == {
        "V1_regime_fixed_income", "V1_regime_commodity",
    }
    magnitude = CFG["black_litterman"]["regime_view_magnitude"]
    for label, q in zip(view_set.labels, view_set.q_rows):
        bucket = label.replace("V1_regime_", "")
        ticker = CLASS_BUCKET[CLASS_BUCKET == bucket].index[0]
        assert q == pytest.approx(PRIOR[ticker] + magnitude)


def test_regime_view_neutral_is_empty():
    view_set = views.regime_view("neutral", CLASS_BUCKET, PRIOR, CFG)
    assert len(view_set) == 0


def test_regime_view_disabled_module_is_empty():
    cfg = {**CFG, "modules": {**CFG["modules"], "regime_view": False}}
    view_set = views.regime_view("risk_on", CLASS_BUCKET, PRIOR, cfg)
    assert len(view_set) == 0


def test_meanreversion_views_skips_zero_and_uses_prior_plus_value():
    view_series = pd.Series(
        {"SPY": 0.0, "QQQ": 0.01, "AGG": 0.0, "GLD": -0.005}
    )
    view_set = views.meanreversion_views(view_series, PRIOR, CFG)

    assert view_set.labels == ["V2_meanrev_QQQ", "V2_meanrev_GLD"]
    assert view_set.q_rows[0] == pytest.approx(PRIOR["QQQ"] + 0.01)
    assert view_set.q_rows[1] == pytest.approx(PRIOR["GLD"] - 0.005)
    assert all(c == CFG["black_litterman"]["view_confidence"]["meanreversion"]
               for c in view_set.confidence)


def test_meanreversion_views_disabled_module_is_empty():
    cfg = {**CFG, "modules": {**CFG["modules"], "meanreversion_view": False}}
    view_series = pd.Series({"SPY": 0.01, "QQQ": 0.0, "AGG": 0.0, "GLD": 0.0})
    view_set = views.meanreversion_views(view_series, PRIOR, cfg)
    assert len(view_set) == 0


def test_technical_and_sentiment_views_use_correct_confidence_keys():
    view_series = pd.Series({"SPY": 0.01, "QQQ": 0.0, "AGG": 0.0, "GLD": 0.0})
    tech = views.technical_views(view_series, PRIOR, CFG)
    sent = views.sentiment_views(view_series, PRIOR, CFG)
    confidence = CFG["black_litterman"]["view_confidence"]
    assert tech.confidence == [confidence["technical"]]
    assert sent.confidence == [confidence["sentiment"]]


def test_momentum_views_uses_correct_confidence_key_and_label():
    view_series = pd.Series(
        {"SPY": 0.02, "QQQ": 0.0, "AGG": 0.0, "GLD": -0.01}
    )
    mom = views.momentum_views(view_series, PRIOR, CFG)
    confidence = CFG["black_litterman"]["view_confidence"]
    assert mom.labels == ["V5_momentum_SPY", "V5_momentum_GLD"]
    assert mom.confidence == [confidence["momentum"], confidence["momentum"]]
    assert mom.q_rows[0] == pytest.approx(PRIOR["SPY"] + 0.02)
    assert mom.q_rows[1] == pytest.approx(PRIOR["GLD"] - 0.01)


def test_momentum_views_disabled_module_is_empty():
    cfg = {**CFG, "modules": {**CFG["modules"], "momentum_view": False}}
    view_series = pd.Series({"SPY": 0.02, "QQQ": 0.0, "AGG": 0.0, "GLD": 0.0})
    view_set = views.momentum_views(view_series, PRIOR, cfg)
    assert len(view_set) == 0


def test_assemble_stacks_multiple_view_sets():
    risk_on = views.regime_view("risk_on", CLASS_BUCKET, PRIOR, CFG)
    meanrev = views.meanreversion_views(
        pd.Series({"SPY": 0.0, "QQQ": 0.01, "AGG": 0.0, "GLD": -0.005}),
        PRIOR, CFG,
    )
    P, Q, Omega, labels = views.assemble(
        [risk_on, meanrev], pd.Index(TICKERS), COV, CFG
    )

    assert P.shape == (3, 4)
    assert len(Q) == 3
    assert Omega.shape == (3, 3)
    assert labels == ["V1_regime_equity", "V2_meanrev_QQQ", "V2_meanrev_GLD"]


def test_assemble_empty_when_no_views():
    empty = views.ViewSet()
    P, Q, Omega, labels = views.assemble([empty], pd.Index(TICKERS), COV, CFG)
    assert P is None
    assert Q is None
    assert Omega is None
    assert labels == []


def test_assemble_omega_smaller_for_higher_confidence():
    # Same view row/magnitude, only confidence differs -> higher
    # confidence must give a strictly smaller Omega entry.
    high_conf = views.ViewSet(
        p_rows=[np.array([1.0, 0.0, 0.0, 0.0])],
        q_rows=[0.1],
        confidence=[1.0],
        labels=["high"],
    )
    low_conf = views.ViewSet(
        p_rows=[np.array([1.0, 0.0, 0.0, 0.0])],
        q_rows=[0.1],
        confidence=[0.1],
        labels=["low"],
    )
    ticker_index = pd.Index(TICKERS)
    _, _, omega_high, _ = views.assemble([high_conf], ticker_index, COV, CFG)
    _, _, omega_low, _ = views.assemble([low_conf], ticker_index, COV, CFG)

    assert omega_high[0, 0] < omega_low[0, 0]
