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

Public API:
    detrend_log_price(prices, lookback) -> pd.DataFrame
    zscore(series, lookback) -> pd.Series | pd.DataFrame
    adf_pvalue(series) -> float
    ou_half_life(series) -> float
    volatility_filter(returns, cfg) -> pd.Series        # bool mask
    reversion_signal(prices, cfg, as_of) -> pd.DataFrame  # diagnostic table
    mean_reversion_view(prices, cfg) -> pd.Series       # per-asset view, V2
    event_study(prices, cfg, horizon_days) -> pd.DataFrame
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
    prices: pd.DataFrame, cfg: dict, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Per-asset mean-reversion diagnostic table as of the last (or
    given) date: z, ADF p-value, half-life, vol-filter pass, and the
    resulting view (S10). `view` is the OU regression's own fitted
    expected next-period drift (beta * current deviation), zero
    unless |z| > entry_z, ADF, half-life, and vol-filter all pass.
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

    diag["view"] = (diag["beta"] * diag["detrended"]).where(
        diag["tradeable"], 0.0
    )
    return diag


def mean_reversion_view(prices: pd.DataFrame, cfg: dict) -> pd.Series:
    """Per-asset V2 view (S10): thin wrapper around reversion_signal
    for callers that just want the final view vector."""
    return reversion_signal(prices, cfg)["view"]


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
