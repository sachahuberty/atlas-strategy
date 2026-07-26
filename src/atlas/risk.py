"""Risk overlays (L4): historical/sensitivity/reverse stress tests (S7).

The anomaly override (stage 4, `strategy.with_anomaly_override`) and
regime-conditional caps (stage 3, `allocation.apply_defensive_tilt`)
already live where they were built, wired directly into strategy.py's
weekly pipeline -- this module is the historical/sensitivity/reverse
stress-test half of stage 10, the part not yet built anywhere.

All three functions take a single, static weight vector representing
"the current book" (e.g. the frozen strategy's most recent target
weights) and ask what would happen to THAT book under a shock,
without rebalancing -- these are one-off diagnostic reports, not
per-week backtest components, so unlike regimes.py/meanreversion.py/
technicals.py/sentiment.py there is no `has_enough_history` guard:
the caller supplies appropriately-ranged data directly.

Public API:
    historical_stress(weights, returns, scenarios) -> pd.DataFrame
    sensitivity_stress(weights, class_bucket, shocks) -> pd.DataFrame
    reverse_stress(weights, returns, window_days) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import max_drawdown

TRADING_DAYS_PER_YEAR = 252


def historical_stress(
    weights: pd.Series,
    returns: pd.DataFrame,
    scenarios: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Buy-and-hold the current book's weights through each historical
    scenario window (S7): "if this shock repeated, starting from
    today's weights and never rebalancing, what happens?" One row per
    scenario: realized date range, cumulative return, max drawdown,
    worst single day, and annualized vol during the window."""
    w = weights.reindex(returns.columns).fillna(0.0)
    rows = {}
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end]
        port_returns = window.mul(w, axis=1).sum(axis=1)
        rows[name] = {
            "start": window.index.min() if len(window) else pd.NaT,
            "end": window.index.max() if len(window) else pd.NaT,
            "n_days": len(window),
            "cumulative_return": float((1.0 + port_returns).prod() - 1.0),
            "max_drawdown": max_drawdown(port_returns),
            "worst_day": (
                float(port_returns.min()) if len(port_returns) else np.nan
            ),
            "ann_vol": (
                float(
                    port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                )
                if len(port_returns) > 1
                else np.nan
            ),
        }
    return pd.DataFrame(rows).T


def sensitivity_stress(
    weights: pd.Series,
    class_bucket: pd.Series,
    shocks: dict[str, list[float]],
) -> pd.DataFrame:
    """One-shot P&L of the current book under hypothetical per-asset-
    class return shocks (S7), e.g. "equity -20%": every ticker in that
    bucket takes the same percentage shock, weighted by the book's
    actual per-ticker weight. Not a historical replay (see
    `historical_stress`) -- a direct sensitivity table."""
    w = weights.reindex(class_bucket.index).fillna(0.0)
    rows = []
    for bucket, magnitudes in shocks.items():
        bucket_weight = w[class_bucket == bucket].sum()
        for shock in magnitudes:
            rows.append(
                {
                    "bucket": bucket,
                    "shock": shock,
                    "bucket_weight": bucket_weight,
                    "portfolio_impact": bucket_weight * shock,
                }
            )
    return pd.DataFrame(rows)


def reverse_stress(
    weights: pd.Series, returns: pd.DataFrame, window_days: int
) -> dict:
    """"What kills this strategy?" (S7): scan the current book's
    buy-and-hold return series (same weights, no rebalancing) for its
    single worst day and its worst `window_days`-day rolling window
    within the given return history, and report which tickers drove
    each, ranked by weight * realized return (marginal contribution).
    """
    w = weights.reindex(returns.columns).fillna(0.0)
    port_returns = returns.mul(w, axis=1).sum(axis=1)

    worst_day = port_returns.idxmin()
    worst_day_contributors = (returns.loc[worst_day] * w).sort_values()

    rolling_cum = (
        (1.0 + port_returns)
        .rolling(window_days)
        .apply(lambda x: x.prod() - 1.0, raw=True)
    )
    worst_window_end = rolling_cum.idxmin()
    end_pos = returns.index.get_loc(worst_window_end)
    start_pos = max(0, end_pos - window_days + 1)
    worst_window_start = returns.index[start_pos]
    window_returns = returns.loc[worst_window_start:worst_window_end]
    worst_window_contributors = (
        window_returns.mul(w, axis=1).sum().sort_values()
    )

    return {
        "worst_day": worst_day,
        "worst_day_return": float(port_returns.loc[worst_day]),
        "worst_day_contributors": worst_day_contributors,
        "worst_window_start": worst_window_start,
        "worst_window_end": worst_window_end,
        "worst_window_return": float(rolling_cum.loc[worst_window_end]),
        "worst_window_contributors": worst_window_contributors,
    }
