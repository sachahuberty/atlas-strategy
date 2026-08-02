"""Bucket-level allocation (L3 alternative to asset-level BL, S11 Tier 2).

DIAGNOSTIC.md's most robust finding: with ~22 assets that are really
4-5 uncorrelated bets (asset-class buckets), estimating a 22x22
covariance and 22 expected returns is mostly estimating noise.
Collapsing to bucket-level estimation makes both problems tractable,
and V1 (regime posture) is already an asset-class-level view -- the
only view that operates at the right granularity, and the only one
with a positive marginal OOS contribution (stage 11's ablation). This
module provides the two building blocks that let the SAME Black-
Litterman machinery in allocation.py/views.py run on bucket-level
"assets" instead of individual tickers:

- `bucket_returns` turns per-asset returns into per-bucket returns
  (equal-weighted average within each bucket), for covariance/prior
  estimation.
- `expand_bucket_weights` turns the resulting bucket-level weight
  vector back into per-asset weights (equal split within each
  bucket), for backtest.run's per-asset accounting.

See strategy.bucket_black_litterman_strategy for the actual strategy
built from these two functions -- kept alongside, not instead of, the
asset-level path (strategy.black_litterman_strategy), so both can be
compared on the same OOS window.

Public API:
    bucket_returns(returns, class_bucket, buckets) -> pd.DataFrame
    expand_bucket_weights(bucket_weights, class_bucket) -> pd.Series
"""

from __future__ import annotations

import pandas as pd


def bucket_returns(
    returns: pd.DataFrame, class_bucket: pd.Series, buckets: list[str]
) -> pd.DataFrame:
    """Equal-weighted average daily return per bucket (S11 Tier 2):
    one column per bucket in `buckets`, each the mean of that
    bucket's member tickers' returns. A bucket with no members
    present in `returns` is silently omitted from the result."""
    columns = {}
    for bucket in buckets:
        members = class_bucket[class_bucket == bucket].index
        members = [m for m in members if m in returns.columns]
        if not members:
            continue
        columns[bucket] = returns[members].mean(axis=1)
    return pd.DataFrame(columns)


def expand_bucket_weights(
    bucket_weights: pd.Series, class_bucket: pd.Series
) -> pd.Series:
    """Split each bucket's weight equally across its member tickers
    (S11 Tier 2), returning a full per-asset weight Series indexed
    over every ticker in `class_bucket`. Tickers whose bucket isn't a
    key in `bucket_weights` (e.g. a bucket excluded from the bucket-
    level universe because its equilibrium weight was zero) get 0."""
    weights = pd.Series(0.0, index=class_bucket.index)
    for bucket, weight in bucket_weights.items():
        members = class_bucket[class_bucket == bucket].index
        if len(members) == 0:
            continue
        weights.loc[members] = weight / len(members)
    return weights
