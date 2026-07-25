"""Technical levels & options positioning (L2 view V3 + execution
timing, S12).

Stage 6. Price part (backtestable): pivot highs/lows (vectorized via a
centered rolling max/min, no manual row loops) -> K-Means
support/resistance zones (same clustering pattern as universe.py's
correlation de-dup) -> a bounded level-proximity view + an
event-study sanity check.

Options part (LIVE ONLY): yfinance has no historical option chains
(confirmed in data.py's stage-1 docstring), so OI notional, call/put
walls, and the gamma proxy cannot be backtested -- per
PROJECT_STRUCTURE.md section 7, the backtest uses only the price-level
component of V3, while positioning is a live-snapshot diagnostic.

Public API:
    has_enough_history(prices, cfg) -> bool
    find_pivots(prices, order) -> pd.DataFrame   # pivot_high/pivot_low bools
    sr_zones(prices, cfg) -> np.ndarray           # per-asset zone levels
    level_proximity_signal(current_price, zones, cfg) -> dict
    technical_signal(prices, cfg, as_of) -> pd.DataFrame  # diagnostic table
    technical_view(prices, cfg) -> pd.Series      # per-asset view, V3
    event_study(prices, cfg, horizon_days) -> pd.DataFrame
    call_put_walls(option_chain) -> dict          # live only
    oi_notional(option_chain, spot_price) -> float  # live only
    gamma_proxy(option_chain, spot_price) -> float  # live only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def has_enough_history(prices: pd.DataFrame, cfg: dict) -> bool:
    """Whether there's enough trailing data for a meaningful pivot
    count to cluster into zones."""
    tcfg = cfg["technicals"]
    min_obs = max(tcfg["min_history_days"], tcfg["pivot_order"] * 10)
    return len(prices.dropna(how="all")) >= min_obs


def find_pivots(prices: pd.Series, order: int) -> pd.DataFrame:
    """Local pivot highs/lows (S12): pivot_high[i] is True when
    prices[i] equals the max within +/- `order` days, pivot_low[i]
    when it equals the min. A centered rolling window, not a manual
    row loop."""
    window = 2 * order + 1
    rolling_max = prices.rolling(window, center=True).max()
    rolling_min = prices.rolling(window, center=True).min()
    return pd.DataFrame(
        {
            "pivot_high": prices == rolling_max,
            "pivot_low": prices == rolling_min,
        }
    )


def sr_zones(prices: pd.Series, cfg: dict) -> np.ndarray:
    """K-Means cluster centers of pivot price levels (S6/S12): the
    per-asset support/resistance zone levels, sorted ascending. Empty
    array if there aren't enough pivots to form `sr_cluster_k` clusters."""
    tcfg = cfg["technicals"]
    pivots = find_pivots(prices, tcfg["pivot_order"])
    is_pivot = pivots["pivot_high"] | pivots["pivot_low"]
    levels = prices[is_pivot].dropna().to_numpy()
    k = tcfg["sr_cluster_k"]
    if len(levels) < k:
        return np.array([])
    seed = cfg["general"]["random_seed"]
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    model.fit(levels.reshape(-1, 1))
    return np.sort(model.cluster_centers_.flatten())


def level_proximity_signal(
    current_price: float, zones: np.ndarray, cfg: dict
) -> dict:
    """Per-asset level-proximity view (S12): is current price sitting
    at/above a zone (support -- expect a bounce, positive view) or
    at/below one (resistance -- expect rejection, negative view)?
    Bounded by `max_view_magnitude`; zero and role=None if price isn't
    within `zone_width_bps` of any zone."""
    tcfg = cfg["technicals"]
    if len(zones) == 0 or current_price <= 0:
        return {
            "zone": np.nan,
            "distance_bps": np.nan,
            "role": None,
            "view": 0.0,
            "phase_entry": False,
        }

    width = current_price * tcfg["zone_width_bps"] / 1e4
    distances = current_price - zones
    nearest_idx = np.argmin(np.abs(distances))
    distance = distances[nearest_idx]

    if abs(distance) > width:
        return {
            "zone": zones[nearest_idx],
            "distance_bps": distance / current_price * 1e4,
            "role": None,
            "view": 0.0,
            "phase_entry": False,
        }

    is_support = distance >= 0
    proximity = 1.0 - abs(distance) / width
    direction = 1.0 if is_support else -1.0
    return {
        "zone": zones[nearest_idx],
        "distance_bps": distance / current_price * 1e4,
        "role": "support" if is_support else "resistance",
        "view": direction * proximity * tcfg["max_view_magnitude"],
        # Per PROJECT_STRUCTURE.md's execution-timing rule: entries at
        # resistance may be phased. Simplified to flag the whole
        # asset, not just increasing trades (see strategy.py).
        "phase_entry": not is_support,
    }


def technical_signal(
    prices: pd.DataFrame, cfg: dict, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Per-asset technical diagnostic table as of the last (or given)
    date: nearest zone, distance, role, the resulting bounded view,
    and the execution-timing flag (S12)."""
    if as_of is not None:
        prices = prices.loc[:as_of]

    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        zones = sr_zones(series, cfg)
        current_price = float(series.iloc[-1])
        signal = level_proximity_signal(current_price, zones, cfg)
        rows.append(
            {"ticker": ticker, "current_price": current_price, **signal}
        )
    return pd.DataFrame(rows).set_index("ticker")


def technical_view(prices: pd.DataFrame, cfg: dict) -> pd.Series:
    """Per-asset V3 view (S12): thin wrapper around technical_signal
    for callers that just want the view vector."""
    return technical_signal(prices, cfg)["view"]


def event_study(
    prices: pd.DataFrame, cfg: dict, horizon_days: int = 10
) -> pd.DataFrame:
    """Sanity check (S12): among all in-sample dates where price sat
    within `zone_width_bps` of a K-Means S/R zone, did it move in the
    expected direction (bounce off support, reject off resistance)
    over the next `horizon_days`?"""
    tcfg = cfg["technicals"]
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        zones = sr_zones(series, cfg)
        if len(zones) == 0:
            rows.append({"ticker": ticker, "n_events": 0, "hit_rate": np.nan})
            continue

        values = series.to_numpy()
        width = values * tcfg["zone_width_bps"] / 1e4
        diffs = values.reshape(-1, 1) - zones.reshape(1, -1)
        nearest_idx = np.argmin(np.abs(diffs), axis=1)
        nearest_zone = zones[nearest_idx]
        distance = values - nearest_zone
        near = np.abs(distance) <= width
        expected_direction = np.where(distance >= 0, 1.0, -1.0)

        future_price = series.shift(-horizon_days).to_numpy()
        future_change = future_price - values
        valid = near & ~np.isnan(future_change)
        n = int(valid.sum())
        if n == 0:
            rows.append({"ticker": ticker, "n_events": 0, "hit_rate": np.nan})
            continue
        hits = np.sign(future_change[valid]) == expected_direction[valid]
        rows.append(
            {"ticker": ticker, "n_events": n, "hit_rate": float(hits.mean())}
        )
    return pd.DataFrame(rows).set_index("ticker")


def call_put_walls(option_chain: pd.DataFrame) -> dict:
    """Live-only (S12): the strike with the largest call open interest
    (a common resistance level) and the largest put open interest
    (a common support level)."""
    calls = option_chain[option_chain["option_type"] == "call"]
    puts = option_chain[option_chain["option_type"] == "put"]
    call_wall = (
        calls.groupby("strike")["openInterest"].sum().idxmax()
        if len(calls) > 0
        else np.nan
    )
    put_wall = (
        puts.groupby("strike")["openInterest"].sum().idxmax()
        if len(puts) > 0
        else np.nan
    )
    return {"call_wall": call_wall, "put_wall": put_wall}


def oi_notional(option_chain: pd.DataFrame, spot_price: float) -> float:
    """Live-only (S12): total open-interest notional (contracts * 100
    shares * spot price), a simple positioning-size proxy."""
    return float(option_chain["openInterest"].sum() * 100 * spot_price)


def gamma_proxy(option_chain: pd.DataFrame, spot_price: float) -> float:
    """Live-only (S12): a coarse dealer-positioning proxy = the
    call/put open-interest imbalance within 5% of spot, in [-1, 1].
    yfinance's chain has no per-contract greeks, so this is a
    simplification of a true gamma-exposure (GEX) calculation, not a
    substitute for one."""
    near = option_chain[
        (option_chain["strike"] - spot_price).abs() / spot_price < 0.05
    ]
    call_oi = near.loc[near["option_type"] == "call", "openInterest"].sum()
    put_oi = near.loc[near["option_type"] == "put", "openInterest"].sum()
    total = call_oi + put_oi
    if total == 0:
        return 0.0
    return float((call_oi - put_oi) / total)
