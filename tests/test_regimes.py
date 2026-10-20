"""Profile-based mapping and HMM/K-Means tests for regimes.py (stage 3)."""

import numpy as np
import pandas as pd
import pytest

from atlas import regimes

POSTURE_CFG = {
    "postures": {
        "risk_on": {
            "profile": {"vol": "low", "mean_return": "high"},
            "allocation_method": "max_sharpe",
        },
        "neutral": {
            "profile": {"vol": "medium", "mean_return": "medium"},
            "allocation_method": "risk_parity",
        },
        "risk_off": {
            "profile": {"vol": "high", "mean_return": "low"},
            "allocation_method": "gmv_defensive",
            "defensive_class_tilt": {"fixed_income": 0.10, "commodity": 0.10},
        },
    }
}


def test_map_states_to_postures_uses_profile_not_index():
    # State ids deliberately out of order vs. their risk profile, to
    # prove the mapping reads vol/return, not the raw index.
    profiles = pd.DataFrame(
        {
            "mean_return": {0: -0.20, 1: 0.15, 2: 0.02},
            "vol": {0: 0.30, 1: 0.08, 2: 0.15},
        }
    )
    mapping = regimes.map_states_to_postures(profiles, POSTURE_CFG)
    assert mapping[0] == "risk_off"
    assert mapping[1] == "risk_on"
    assert mapping[2] == "neutral"


def test_map_states_to_postures_degrades_with_fewer_states():
    # ANALYSIS_V4 Sec 2.3a: only 2 of 3 configured postures' states
    # actually appeared (e.g. a quiet window where one regime never
    # won the Viterbi path) -- must not raise, must still preserve
    # relative risk ordering, AND the high-vol/low-return state must
    # actually reach risk_off, not just rank "more risk-off than the
    # other one." Pre-fix, a 0..(n-1) empirical scale maxed out at
    # +-1 with 2 states, tying risk_off's target (+2) with neutral's
    # (0); dict order silently picked neutral every time.
    profiles = pd.DataFrame(
        {"mean_return": {0: 0.15, 1: -0.20}, "vol": {0: 0.08, 1: 0.30}}
    )
    mapping = regimes.map_states_to_postures(profiles, POSTURE_CFG)
    assert set(mapping.keys()) == {0, 1}
    assert mapping[0] == "risk_on"
    assert mapping[1] == "risk_off"


def test_map_states_to_postures_three_states_unchanged():
    # 2/(n-1) == 1 at n=3, so this must reproduce the pre-fix result
    # exactly: the rescaling is a no-op when every configured posture's
    # state is actually visited.
    profiles = pd.DataFrame(
        {
            "mean_return": {0: -0.20, 1: 0.15, 2: 0.02},
            "vol": {0: 0.30, 1: 0.08, 2: 0.15},
        }
    )
    mapping = regimes.map_states_to_postures(profiles, POSTURE_CFG)
    assert mapping == {0: "risk_off", 1: "risk_on", 2: "neutral"}


def test_map_states_to_postures_single_state_is_neutral():
    # A single visited state carries no rank information -- must land
    # on neutral exactly (the midpoint target score, 0), not on
    # whichever posture happens to be first in insertion order.
    profiles = pd.DataFrame({"mean_return": {0: 0.30}, "vol": {0: 0.30}})
    mapping = regimes.map_states_to_postures(profiles, POSTURE_CFG)
    assert mapping == {0: "neutral"}


def test_state_profiles_known_values():
    states = pd.Series([0, 0, 1, 1], index=pd.RangeIndex(4))
    returns = pd.Series([0.01, 0.02, -0.01, -0.02], index=pd.RangeIndex(4))
    profiles = regimes.state_profiles(states, returns)

    state0 = returns.iloc[:2]
    state1 = returns.iloc[2:]
    assert profiles.loc[0, "mean_return"] == pytest.approx(
        state0.mean() * 252
    )
    assert profiles.loc[0, "vol"] == pytest.approx(
        state0.std(ddof=1) * np.sqrt(252)
    )
    assert profiles.loc[1, "mean_return"] == pytest.approx(
        state1.mean() * 252
    )


def test_market_regime_flags_a_clearly_volatile_negative_block():
    # Two-block synthetic market: calm/up, then volatile/down.
    n = 1400
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(1)
    values = []
    for i in range(n):
        if (i // 100) % 2 == 0:
            values.append(rng.normal(0.0006, 0.006))
        else:
            values.append(rng.normal(-0.0004, 0.02))
    market_returns = pd.Series(values, index=dates)

    cfg = {
        "general": {"random_seed": 42},
        "regimes": {
            "hmm": {
                "n_states": 3,
                "vol_window_days": 21,
                "lookback_days": 1260,
            },
        },
    }
    result = regimes.market_regime(market_returns, cfg, POSTURE_CFG)

    # The series ends in a volatile/negative block -> risk_off.
    assert result["current_posture"] == "risk_off"
    assert set(result["posture_series"].unique()) <= {
        "risk_on", "neutral", "risk_off",
    }


def _two_block_market_returns(n=1400, seed=1) -> pd.Series:
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    values = []
    for i in range(n):
        if (i // 100) % 2 == 0:
            values.append(rng.normal(0.0006, 0.006))
        else:
            values.append(rng.normal(-0.0004, 0.02))
    return pd.Series(values, index=dates)


def test_hmm_bic_curve_returns_series_indexed_by_n_states():
    market_returns = _two_block_market_returns()
    cfg = {
        "general": {"random_seed": 42},
        "regimes": {
            "hmm": {"n_states": 3, "vol_window_days": 21, "lookback_days": 1260}
        },
    }
    bic = regimes.hmm_bic_curve(market_returns, cfg, range(2, 5))
    assert list(bic.index) == [2, 3, 4]
    assert bic.notna().all()
    assert (np.isfinite(bic)).all()


def test_hmm_bic_curve_is_deterministic_given_a_fixed_seed():
    market_returns = _two_block_market_returns()
    cfg = {
        "general": {"random_seed": 42},
        "regimes": {
            "hmm": {"n_states": 3, "vol_window_days": 21, "lookback_days": 1260}
        },
    }
    first = regimes.hmm_bic_curve(market_returns, cfg, range(2, 4))
    second = regimes.hmm_bic_curve(market_returns, cfg, range(2, 4))
    pd.testing.assert_series_equal(first, second)


def _synthetic_macro_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    growth = pd.Series(100.0, index=dates)
    inflation = pd.Series(100.0, index=dates)
    # Engineer exact 12-month-lagged pct_change values at index 12-15
    # (see label mapping test below for the resulting quadrant math).
    growth.iloc[[12, 13, 14, 15]] = [110.0, 108.0, 95.0, 93.0]
    inflation.iloc[[12, 13, 14, 15]] = [99.0, 106.0, 105.0, 98.0]
    return pd.DataFrame({"GROWTH": growth, "INFLATION": inflation})


def test_label_macro_clusters_matches_business_cycle_quadrants():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    macro_df = _synthetic_macro_df(dates)
    cfg = {
        "regimes": {
            "kmeans_macro": {
                "growth_indicator": "GROWTH",
                "inflation_indicator": "INFLATION",
            }
        }
    }
    # Four fabricated clusters, one per quadrant, at the four engineered
    # dates: (growth+, infl-) Recovery, (growth+, infl+) Expansion,
    # (growth-, infl+) Slowdown, (growth-, infl-) Contraction.
    labels = pd.Series(
        [0, 1, 2, 3], index=[dates[12], dates[13], dates[14], dates[15]]
    )

    mapping = regimes.label_macro_clusters(macro_df, labels, cfg)

    assert mapping[0] == "Recovery"
    assert mapping[1] == "Expansion"
    assert mapping[2] == "Slowdown"
    assert mapping[3] == "Contraction"
