"""Tests for technicals.py: V3 technical/positioning signal (stage 6)."""

import numpy as np
import pandas as pd
import pytest

from atlas import technicals

CFG = {
    "general": {"random_seed": 42},
    "technicals": {
        "pivot_order": 5,
        "sr_cluster_k": 2,
        "zone_width_bps": 100,
        "max_view_magnitude": 0.02,
        "min_history_days": 60,
        "phase_fraction": 0.5,
    },
}


def _oscillating_prices(n=400, seed=9) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = 100 + 5 * np.sin(2 * np.pi * np.arange(n) / 40) + rng.normal(
        0, 0.3, n
    )
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(x, index=dates)


def test_has_enough_history():
    short = pd.DataFrame({"A": np.zeros(10)})
    long_ = pd.DataFrame({"A": np.zeros(200)})
    assert not technicals.has_enough_history(short, CFG)
    assert technicals.has_enough_history(long_, CFG)


def test_find_pivots_detects_a_known_peak_and_trough():
    values = [1, 2, 3, 5, 3, 2, 1, 0, 1, 2, 3]
    series = pd.Series(values, dtype=float)
    pivots = technicals.find_pivots(series, order=3)
    assert pivots.loc[3, "pivot_high"]
    assert pivots.loc[7, "pivot_low"]
    assert not pivots.loc[5, "pivot_high"]
    assert not pivots.loc[5, "pivot_low"]


def test_sr_zones_finds_two_clear_levels():
    prices = _oscillating_prices()
    zones = technicals.sr_zones(prices, CFG)
    assert len(zones) == 2
    # The true oscillation levels are ~95 and ~105.
    assert zones[0] == pytest.approx(95.0, abs=1.0)
    assert zones[1] == pytest.approx(105.0, abs=1.0)


def test_sr_zones_empty_when_too_few_pivots():
    # Strictly monotonic: no interior point is a local max or min, so
    # there are no pivots at all to cluster.
    monotonic = pd.Series(np.arange(30, dtype=float))
    zones = technicals.sr_zones(monotonic, CFG)
    assert len(zones) == 0


def test_level_proximity_signal_support_and_resistance_signs():
    zones = np.array([95.0, 105.0])
    # Sitting just above support -> positive view, not phased.
    support = technicals.level_proximity_signal(95.05, zones, CFG)
    assert support["role"] == "support"
    assert support["view"] > 0
    assert not support["phase_entry"]

    # Sitting just below resistance -> negative view, phased entry.
    resistance = technicals.level_proximity_signal(104.95, zones, CFG)
    assert resistance["role"] == "resistance"
    assert resistance["view"] < 0
    assert resistance["phase_entry"]


def test_level_proximity_signal_zero_when_far_from_any_zone():
    zones = np.array([95.0, 105.0])
    mid = technicals.level_proximity_signal(100.0, zones, CFG)
    assert mid["role"] is None
    assert mid["view"] == 0.0


def test_level_proximity_signal_view_bounded_by_max_magnitude():
    zones = np.array([100.0])
    # Sitting exactly at the zone: proximity = 1.0, so |view| should
    # equal max_view_magnitude exactly.
    at_zone = technicals.level_proximity_signal(100.0, zones, CFG)
    assert abs(at_zone["view"]) == pytest.approx(
        CFG["technicals"]["max_view_magnitude"]
    )


def test_event_study_high_hit_rate_for_a_clean_oscillator():
    prices = _oscillating_prices().to_frame("A")
    event = technicals.event_study(prices, CFG, horizon_days=10)
    assert event.loc["A", "n_events"] > 0
    assert event.loc["A", "hit_rate"] > 0.7


def test_technical_view_matches_technical_signal_view_column():
    prices = _oscillating_prices().to_frame("A")
    view = technicals.technical_view(prices, CFG)
    diag = technicals.technical_signal(prices, CFG)
    assert view["A"] == pytest.approx(diag.loc["A", "view"])


def _synthetic_option_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [95, 100, 105, 95, 100, 105],
            "openInterest": [500, 800, 3000, 4000, 1200, 300],
            "option_type": [
                "call", "call", "call", "put", "put", "put",
            ],
        }
    )


def test_call_put_walls_picks_max_oi_strike():
    chain = _synthetic_option_chain()
    walls = technicals.call_put_walls(chain)
    assert walls["call_wall"] == 105
    assert walls["put_wall"] == 95


def test_oi_notional_known_value():
    chain = _synthetic_option_chain()
    spot = 100.0
    expected = chain["openInterest"].sum() * 100 * spot
    assert technicals.oi_notional(chain, spot) == pytest.approx(expected)


def test_gamma_proxy_sign_reflects_put_heavy_positioning():
    # Near-spot strikes (within 5%) are put-heavy -> negative proxy.
    chain = _synthetic_option_chain()
    proxy = technicals.gamma_proxy(chain, spot_price=100.0)
    assert -1.0 <= proxy <= 1.0
    assert proxy < 0
