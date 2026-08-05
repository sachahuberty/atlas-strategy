"""Momentum signal (L2 view V5): 12-1 cross-sectional rank + absolute
trend-vs-cash gate (S4 factor investing / smart beta).

Stage 11 (ANALYSIS_V2.md Sec 5/6 action 1). The one course-covered
signal family this project never implemented, and the natural
complement to a Permanent Portfolio: V2 (mean-reversion) and V3
(technical) both measured negative or ~zero, and notebook 05's own
event study found some tickers' extreme deviations reliably
CONTINUED rather than reverted (a momentum signal wearing a
mean-reversion disguise). Dual momentum (Antonacci): a relative leg
(within each asset-class bucket, rank tickers by their own 12-1
return and favor the top half) and an absolute leg (a bucket only
gets a relative-momentum view at all if its own trailing return beats
the risk-free rate over the same horizon; otherwise every member of
that bucket gets a flat negative view instead).

12-1 momentum: cumulative return from `lookback_days` (252, ~12
months) ago to `skip_days` (21, ~1 month) ago -- the most recent
month is excluded from the lookback window, the standard convention
to avoid the well-documented short-term reversal effect overlapping
with the momentum window.

Public API:
    has_enough_history(prices, cfg) -> bool
    cross_sectional_score(prices, class_bucket, cfg) -> pd.Series
        # percentile rank in [0, 1], within each bucket
    absolute_trend(prices, class_bucket, rf_series, cfg) -> pd.Series
        # bool, indexed by bucket
    momentum_view(prices, class_bucket, rf_series, cfg) -> pd.Series
        # per-asset view, V5
    event_study(prices, cfg, horizon_days) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import buckets


def has_enough_history(prices: pd.DataFrame, cfg: dict) -> bool:
    """Whether there's enough trailing data for a meaningful 12-1
    score and trend read: the longer of the momentum lookback or the
    trend lookback, plus the skip window and a small margin."""
    mcfg = cfg["momentum"]
    min_obs = (
        max(mcfg["lookback_days"], mcfg["trend_lookback_days"])
        + mcfg["skip_days"]
        + 5
    )
    return len(prices.dropna(how="all")) >= min_obs


def cross_sectional_score(
    prices: pd.DataFrame, class_bucket: pd.Series, cfg: dict
) -> pd.Series:
    """12-1 cross-sectional momentum, ranked as a percentile WITHIN
    each asset-class bucket (not across the whole universe -- bucket
    sizes and volatility differ enormously, so a global rank would
    just reproduce "most volatile trend," not real relative momentum
    within a comparable set). Returns a percentile rank in [0, 1] per
    ticker (1.0 = the best performer in its own bucket); NaN for a
    ticker with too little history or no assigned bucket.
    """
    mcfg = cfg["momentum"]
    lookback = mcfg["lookback_days"]
    skip = mcfg["skip_days"]
    if len(prices) < lookback + 1:
        return pd.Series(np.nan, index=prices.columns)

    p_lookback = prices.iloc[-lookback]
    p_skip = prices.iloc[-(skip + 1)] if skip > 0 else prices.iloc[-1]
    raw_score = p_skip / p_lookback - 1.0

    bucket_of = class_bucket.reindex(raw_score.index)
    return raw_score.groupby(bucket_of).rank(pct=True)


def absolute_trend(
    prices: pd.DataFrame,
    class_bucket: pd.Series,
    rf_series: pd.Series | None,
    cfg: dict,
) -> pd.Series:
    """Per-bucket absolute (time-series) trend gate: a bucket is "in
    uptrend" if its own trailing `trend_lookback_days` return
    (equal-weighted across its member tickers, via `buckets.
    bucket_returns`) exceeds the prevailing risk-free rate compounded
    over the same horizon -- compared against cash, not zero, the
    Antonacci dual-momentum convention, so a bucket merely drifting
    sideways doesn't count as trending. `rf_series`, if given, is
    looked up via `.asof` at the last date in `prices` (never a future
    value); defaults to rf=0.0 if omitted, matching `strategy.
    black_litterman_strategy`'s convention. Returns a bool Series
    indexed by bucket name; empty if there isn't enough history yet.
    """
    mcfg = cfg["momentum"]
    lookback = mcfg["trend_lookback_days"]
    if len(prices) < lookback + 1:
        return pd.Series(dtype=bool)

    active_buckets = sorted(class_bucket.unique())
    daily_returns = prices.pct_change()
    bucket_daily = buckets.bucket_returns(
        daily_returns, class_bucket, active_buckets
    )
    trailing = bucket_daily.tail(lookback)
    bucket_cum_return = (1.0 + trailing).prod() - 1.0

    rf = 0.0
    if rf_series is not None:
        looked_up = rf_series.asof(prices.index[-1])
        if pd.notna(looked_up):
            rf = float(looked_up)
    rf_cum_return = (1.0 + rf) ** (lookback / 252) - 1.0

    return bucket_cum_return > rf_cum_return


def momentum_view(
    prices: pd.DataFrame,
    class_bucket: pd.Series,
    rf_series: pd.Series | None,
    cfg: dict,
) -> pd.Series:
    """Per-asset V5 view (dual momentum, S4): positive for a ticker in
    the top `top_fraction` of its own bucket's 12-1 cross-sectional
    rank, provided that bucket also clears the absolute-trend-vs-cash
    gate; graded by how far above the top_fraction cutoff the
    ticker's percentile rank sits (rank=1.0, the single best performer
    in its bucket, gets the full `max_view_magnitude`). Every member
    of a bucket that FAILS the absolute-trend gate gets a flat
    negative view instead, regardless of its own relative rank --
    absolute momentum is a whole-asset-class defensive signal, not a
    relative one. A bottom-half ticker in a bucket that IS trending
    gets zero (no view either way). Bounded by
    `momentum.max_view_magnitude`.
    """
    mcfg = cfg["momentum"]
    tickers = prices.columns
    view = pd.Series(0.0, index=tickers)

    trending = absolute_trend(prices, class_bucket, rf_series, cfg)
    if trending.empty:
        return view
    rank = cross_sectional_score(prices, class_bucket, cfg)

    bucket_of = class_bucket.reindex(tickers)
    is_trending = bucket_of.map(trending)

    not_trending = is_trending.eq(False)
    view[not_trending] = -mcfg["max_view_magnitude"]

    top_fraction = mcfg["top_fraction"]
    cutoff = 1.0 - top_fraction
    top_half = is_trending.eq(True) & (rank >= cutoff)
    scale = ((rank - cutoff) / top_fraction).clip(lower=0.0, upper=1.0)
    view[top_half] = (scale * mcfg["max_view_magnitude"])[top_half]

    return view


def event_study(
    prices: pd.DataFrame, cfg: dict, horizon_days: int = 20
) -> pd.DataFrame:
    """Sanity check (S10/S12-style, per-asset, bucket-agnostic): does
    a trailing 12-1 momentum score predict CONTINUATION (same-sign
    forward return) rather than reversion over the next
    `horizon_days`? Vectorized across the whole history via shift, one
    ticker's loop only for the final aggregation -- same convention as
    `meanreversion.event_study`/`technicals.event_study`.
    """
    mcfg = cfg["momentum"]
    lookback = mcfg["lookback_days"]
    skip = mcfg["skip_days"]

    log_price = np.log(prices)
    trailing_score = log_price.shift(skip) - log_price.shift(lookback)
    future_change = log_price.shift(-horizon_days) - log_price

    rows = []
    for ticker in prices.columns:
        score = trailing_score[ticker]
        change = future_change[ticker]
        valid = score.notna() & change.notna() & (score != 0)
        n = int(valid.sum())
        if n == 0:
            rows.append({"ticker": ticker, "n_events": 0, "hit_rate": np.nan})
            continue
        continuation = np.sign(change[valid]) == np.sign(score[valid])
        rows.append(
            {
                "ticker": ticker,
                "n_events": n,
                "hit_rate": float(continuation.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
