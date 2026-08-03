"""Mean-reversion signal (L2 view V2): ADF, OU half-life, z-scores (S10).

Stage 5. Signal = detrended log-price z-score, gated by ADF
stationarity, a max-half-life filter, and a volatility filter.
Time-series math (rolling mean/std, shift/diff) is vectorized across
the whole DataFrame; only the per-asset ADF/OU calls loop over the
(small) ticker list, since statsmodels' adfuller has no batched form.

ADF runs on the RAW log-price over a longer `adf_lookback_days`
window, not on the rolling-detrended series used for the z-score.
Subtracting a trailing rolling mean from a series induces spurious
apparent stationarity (confirmed empirically: a true random walk's
rolling-detrended residual rejected the unit-root null at p=0.002,
a false positive), and a short window gives ADF too little power
regardless. Testing the untouched log-price over a longer horizon
avoids both problems.

`exit_z` (config) is position-level hysteresis for a stateful trading
system; this module is a stateless weekly view generator (no held
position to exit from), so it isn't used here -- it becomes relevant
once views feed an actual position-tracking layer.

Stage 11 fix: `view` (in `reversion_signal`) used to be `beta *
detrended` -- the OU regression's raw expected NEXT-DAY drift, a few
tens of basis points. `views._per_asset_views` adds this straight to
the ANNUALIZED equilibrium prior, so the view entered Black-Litterman
a full order of magnitude (or more) below `regime_view_magnitude`
(0.03) and `technicals.max_view_magnitude` (0.02) -- effectively
noise, not a real signal, confirmed by the stage-11 ablation showing
V2's marginal OOS Sharpe contribution near zero. Fixed by scaling the
daily drift by the (capped) reversion half-life in days, so it
represents the expected cumulative move over the reversion horizon
instead of one day -- landing in the same ballpark as the other view
families.

Stage 11 Tier 3: per-asset hit-rate gating. `event_study` scores each
asset's own historical hit rate; DIAGNOSTIC.md Sec 2.1/6 action 6
observed that this rate varies a lot by asset (some names meaningfully
above chance, most indistinguishable from it, a couple below) and
proposed only trading V2 on the names that clear a minimum bar,
instead of applying one uniform rule regardless of whether an asset
has ever shown a real reversion edge. `hit_rate_eligible_tickers`
derives that per-asset eligibility set (IS-only, by construction:
callers pass an IS-only price slice); `reversion_signal`/
`mean_reversion_view` accept an optional `eligible` index to mask
`tradeable` down to that set, so an asset that fails every other gate
check gets zeroed exactly as before, and one that passes them all but
was never a real reversion play (per its own historical hit rate) is
now zeroed too.

Public API:
    detrend_log_price(prices, lookback) -> pd.DataFrame
    zscore(series, lookback) -> pd.Series | pd.DataFrame
    adf_pvalue(series) -> float
    ou_half_life(series) -> float
    volatility_filter(returns, cfg) -> pd.Series        # bool mask
    reversion_signal(prices, cfg, as_of, eligible) -> pd.DataFrame
    mean_reversion_view(prices, cfg, eligible) -> pd.Series  # V2 view
    event_study(prices, cfg, horizon_days) -> pd.DataFrame
    hit_rate_eligible_tickers(prices, cfg, horizon_days) -> pd.Index
    has_enough_history(prices, cfg) -> bool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def has_enough_history(prices: pd.DataFrame, cfg: dict) -> bool:
    """Whether there's enough trailing data for a meaningful signal:
    the longer of the rolling window or the ADF lookback, plus ADF's
    own minimum sample size."""
    mcfg = cfg["meanreversion"]
    min_obs = max(mcfg["lookback_days"], mcfg["adf_lookback_days"]) + 20
    return len(prices.dropna(how="all")) >= min_obs


def detrend_log_price(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Log-price minus its own rolling mean (S10)."""
    log_price = np.log(prices)
    return log_price - log_price.rolling(lookback).mean()


def zscore(series: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Rolling z-score: (x - rolling_mean) / rolling_std (S10)."""
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std()
    return (series - rolling_mean) / rolling_std


def adf_pvalue(series: pd.Series) -> float:
    """ADF test p-value for one series' stationarity (S10). Low
    p-value = reject the unit-root null = the deviation is
    mean-reverting, not just drifting."""
    clean = series.dropna()
    if len(clean) < 20 or clean.std() == 0:
        return np.nan
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except ValueError:
        # adfuller raises on degenerate input it doesn't recognize as
        # constant up front (e.g. near-zero variation after lag
        # selection); treat the same as "can't tell," not a crash.
        return np.nan


def _ou_beta(series: pd.Series) -> float:
    """Closed-form OLS slope of Delta x_t on x_{t-1} (the OU
    discretization): beta = Cov(x_lag, dx) / Var(x_lag)."""
    x = series.dropna()
    if len(x) < 10:
        return np.nan
    x_lag = x.shift(1)
    dx = x - x_lag
    valid = x_lag.notna()
    x_lag, dx = x_lag[valid], dx[valid]
    var_x = ((x_lag - x_lag.mean()) ** 2).sum()
    if var_x == 0:
        return np.nan
    cov_x_dx = ((x_lag - x_lag.mean()) * (dx - dx.mean())).sum()
    return float(cov_x_dx / var_x)


def ou_half_life(series: pd.Series) -> float:
    """OU half-life in periods: ln(2) / -beta (S10). inf when beta >=
    0 (the series isn't mean-reverting at all over this window)."""
    beta = _ou_beta(series)
    if np.isnan(beta):
        return np.nan
    if beta >= 0:
        return np.inf
    return float(np.log(2) / -beta)


def volatility_filter(returns: pd.DataFrame, cfg: dict) -> pd.Series:
    """Per-asset bool mask: True if current rolling vol is at or below
    its OWN trailing percentile cutoff (S10) -- mean-reversion signals
    are unreliable during abnormal volatility spikes."""
    mcfg = cfg["meanreversion"]
    window = mcfg["lookback_days"]
    percentile = mcfg["vol_filter_percentile"]
    rolling_vol = returns.rolling(window).std()
    current = rolling_vol.iloc[-1]
    cutoff = rolling_vol.quantile(percentile / 100.0)
    return current <= cutoff


def reversion_signal(
    prices: pd.DataFrame,
    cfg: dict,
    as_of: pd.Timestamp | None = None,
    eligible: pd.Index | None = None,
) -> pd.DataFrame:
    """Per-asset mean-reversion diagnostic table as of the last (or
    given) date: z, ADF p-value, half-life, vol-filter pass, and the
    resulting view (S10). `view` is the OU regression's fitted
    expected next-day drift (beta * current deviation), scaled by the
    (capped) half-life in days to express it as an expected return
    over the reversion horizon rather than a raw one-day drift --
    comparable in magnitude to the other (annual) view families, not
    two orders of magnitude below them (stage 11 fix). Zero unless
    |z| > entry_z, ADF, half-life, and vol-filter all pass.

    `eligible`, if given (stage 11 Tier 3), further restricts
    `tradeable` to tickers in that index -- see
    `hit_rate_eligible_tickers`. Defaults to None (no restriction,
    every ticker eligible), the pre-Tier-3 behavior.
    """
    mcfg = cfg["meanreversion"]
    window = mcfg["lookback_days"]
    if as_of is not None:
        prices = prices.loc[:as_of]

    log_price = np.log(prices)
    z = zscore(log_price, window)
    detrended = detrend_log_price(prices, window)
    returns = prices.pct_change()

    recent_detrended = detrended.tail(window)
    recent_log_price = log_price.tail(mcfg["adf_lookback_days"])
    rows = []
    for ticker in prices.columns:
        rows.append(
            {
                "ticker": ticker,
                "beta": _ou_beta(recent_detrended[ticker]),
                "half_life": ou_half_life(recent_detrended[ticker]),
                # ADF on the raw log-price, not the rolling-detrended
                # series -- see module docstring.
                "adf_pvalue": adf_pvalue(recent_log_price[ticker]),
            }
        )
    diag = pd.DataFrame(rows).set_index("ticker")
    diag["z"] = z.iloc[-1]
    diag["detrended"] = detrended.iloc[-1]

    if mcfg["vol_filter"]:
        diag["vol_ok"] = volatility_filter(returns, cfg)
    else:
        diag["vol_ok"] = True

    diag["tradeable"] = (
        (diag["z"].abs() > mcfg["entry_z"])
        & (diag["adf_pvalue"] < mcfg["adf_pvalue_threshold"])
        & (diag["half_life"] <= mcfg["max_half_life_days"])
        & diag["vol_ok"]
    ).fillna(False)
    if eligible is not None:
        diag["tradeable"] &= diag.index.isin(eligible)

    # beta * detrended alone is the expected NEXT-DAY drift; scaling by
    # the (capped) half-life turns it into an expected return over the
    # reversion horizon, comparable in magnitude to the other (annual)
    # view families instead of ~two orders of magnitude below them.
    # The cap rarely binds in practice: `tradeable` already requires
    # half_life <= max_half_life_days (60 by default), well under 252.
    horizon = diag["half_life"].clip(upper=252)
    diag["view"] = (diag["beta"] * diag["detrended"] * horizon).where(
        diag["tradeable"], 0.0
    )
    return diag


def mean_reversion_view(
    prices: pd.DataFrame, cfg: dict, eligible: pd.Index | None = None
) -> pd.Series:
    """Per-asset V2 view (S10): thin wrapper around reversion_signal
    for callers that just want the final view vector. `eligible` is
    passed straight through -- see `reversion_signal`."""
    return reversion_signal(prices, cfg, eligible=eligible)["view"]


def event_study(
    prices: pd.DataFrame, cfg: dict, horizon_days: int = 20
) -> pd.DataFrame:
    """Sanity check (S10): among all in-sample dates where |z| >
    entry_z, did price actually move toward the mean over the next
    `horizon_days`? The forward-looking comparison is vectorized
    across time (shift + sign); only the per-asset aggregation loops."""
    mcfg = cfg["meanreversion"]
    window = mcfg["lookback_days"]
    log_price = np.log(prices)
    z = zscore(log_price, window)

    future_change = log_price.shift(-horizon_days) - log_price
    expected_direction = -np.sign(z)
    reverted = np.sign(future_change) == expected_direction
    extreme = z.abs() > mcfg["entry_z"]

    rows = []
    for ticker in prices.columns:
        mask = extreme[ticker] & future_change[ticker].notna()
        n = int(mask.sum())
        hit_rate = float(reverted[ticker][mask].mean()) if n > 0 else np.nan
        rows.append({"ticker": ticker, "n_events": n, "hit_rate": hit_rate})
    return pd.DataFrame(rows).set_index("ticker")


def hit_rate_eligible_tickers(
    prices: pd.DataFrame, cfg: dict, horizon_days: int = 20
) -> pd.Index:
    """Tickers whose own `event_study` hit rate clears
    `meanreversion.min_hit_rate` (stage 11 Tier 3, DIAGNOSTIC.md
    Sec 2.1/6 action 6): a uniform mean-reversion rule applied
    indiscriminately across the universe has no clean edge (stage 5's
    own event study, reconfirmed by the stage-11 ablation's near-zero
    marginal Sharpe for V2), but some individual assets do show a
    real, above-chance reversion tendency. Restricting V2 to only
    those names, rather than every ticker that happens to be
    `tradeable` this week, is meant to isolate that edge.

    Intended to be computed ONCE on an in-sample-only price slice and
    frozen for the OOS run (same convention as `meanreversion.
    lookback_days`/`entry_z`'s grid search), not recomputed inside the
    weekly `reversion_signal` call -- pass the result via
    `reversion_signal`'s/`mean_reversion_view`'s `eligible` parameter.
    """
    min_hit_rate = cfg["meanreversion"]["min_hit_rate"]
    hit_rates = event_study(prices, cfg, horizon_days=horizon_days)
    return hit_rates.index[hit_rates["hit_rate"] >= min_hit_rate]
