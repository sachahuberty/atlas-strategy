"""Universe screening: rule-based global asset selection (L0).

Stage 1. Screens per config: min history, liquidity, missing-data cap,
correlation de-duplication via K-Means clustering (S6), asset-class
bucketing. Output universes are versioned:
data/processed/universe_YYYYMMDD.csv. Re-screened per walk-forward
fold using as-of data only (no survivorship peeking).

Public API:
    stat_ratios(prices) -> pd.DataFrame
        # 1/3/5Y return, vol, Sharpe (S6)
    factor_correlations(returns, factors) -> pd.DataFrame
    screen(candidates, prices, cfg, as_of) -> pd.DataFrame
        # ticker, class_bucket
    save_universe(df, as_of) / load_universe(as_of)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .metrics import ann_return, ann_vol, sharpe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRADING_DAYS_PER_YEAR = 252

# Built-in candidate pool used when config.universe.candidate_pool_file
# is null: broad, liquid, globally diversified ETFs across asset classes.
BUILTIN_CANDIDATE_POOL: dict[str, str] = {
    "SPY": "equity",
    "VTI": "equity",
    "ACWI": "equity",
    "VEA": "equity",
    "VWO": "equity",
    "QQQ": "equity",
    "IWM": "equity",
    "EFA": "equity",
    "AGG": "fixed_income",
    "BND": "fixed_income",
    "TLT": "fixed_income",
    "IEF": "fixed_income",
    "SHY": "fixed_income",
    "BSV": "fixed_income",
    "LQD": "fixed_income",
    "HYG": "fixed_income",
    "TIP": "fixed_income",
    "GLD": "commodity",
    "SLV": "commodity",
    "DBC": "commodity",
    "USO": "commodity",
    "VNQ": "real_estate",
    "BIL": "cash",
}

# Factor proxies for correlation screening vs equity/FI/cash (S6).
FACTOR_PROXIES: dict[str, str] = {
    "equity": "ACWI",
    "fixed_income": "AGG",
    "cash": "BIL",
}


def stat_ratios(
    prices: pd.DataFrame, lookback_years: tuple[int, ...] = (1, 3, 5)
) -> pd.DataFrame:
    """Trailing annualized return, vol, and Sharpe per lookback
    window, per asset (S6)."""
    returns = prices.pct_change().dropna(how="all")
    rows = []
    for ticker in prices.columns:
        r = returns[ticker].dropna()
        row = {"ticker": ticker}
        for years in lookback_years:
            window = r.tail(years * TRADING_DAYS_PER_YEAR)
            min_len = years * TRADING_DAYS_PER_YEAR * 0.5
            enough_history = len(window) >= min_len
            row[f"return_{years}y"] = (
                ann_return(window) if enough_history else np.nan
            )
            row[f"vol_{years}y"] = (
                ann_vol(window) if enough_history else np.nan
            )
            row[f"sharpe_{years}y"] = (
                sharpe(window) if enough_history else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def factor_correlations(
    returns: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """Correlation of each asset's returns against each factor
    return series (S6)."""
    combined = returns.join(factors, how="inner", rsuffix="_factor")
    corr = combined.corr()
    return corr.loc[returns.columns, factors.columns]


def _history_years(prices: pd.Series) -> float:
    valid = prices.dropna()
    if valid.empty:
        return 0.0
    return (valid.index[-1] - valid.index[0]).days / 365.25


def screen(
    candidates: dict[str, str],
    prices: pd.DataFrame,
    cfg: dict,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Apply history/missing-data screens, then de-dup correlated
    assets per asset-class bucket via K-Means clustering on their
    correlation profile (S6).

    `prices` is truncated to `as_of` internally so callers can pass
    the full cached history without introducing lookahead.
    """
    ucfg = cfg["universe"]
    prices = prices.loc[:as_of]

    rows = []
    for ticker, bucket in candidates.items():
        if ticker not in prices.columns:
            continue
        series = prices[ticker]
        first_valid = series.first_valid_index()
        if first_valid is not None:
            missing_ratio = series.loc[first_valid:].isna().mean()
        else:
            missing_ratio = 1.0
        if _history_years(series) < ucfg["min_history_years"]:
            continue
        if missing_ratio > ucfg["max_missing_ratio"]:
            continue
        rows.append({"ticker": ticker, "class_bucket": bucket})

    screened = pd.DataFrame(rows).set_index("ticker")
    if screened.empty:
        return screened

    screened = screened.join(stat_ratios(prices[screened.index]))

    seed = cfg["general"]["random_seed"]
    kept = []
    for _, group in screened.groupby("class_bucket"):
        if len(group) <= 1:
            kept.append(group)
            continue

        returns = prices[group.index].pct_change().dropna()
        corr = returns.corr()
        if corr.isna().any().any():
            kept.append(group)
            continue

        n_clusters = min(ucfg["n_asset_clusters"], len(group))
        # Cluster on each asset's correlation profile (co-movement
        # with peers in the bucket); keep the best-Sharpe rep per
        # cluster.
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = model.fit_predict(corr.values)
        clustered = group.assign(cluster=labels)
        representatives = (
            clustered.reset_index()
            .sort_values("sharpe_3y", ascending=False, na_position="last")
            .groupby("cluster")
            .head(1)
            .set_index("ticker")
            .drop(columns="cluster")
        )
        kept.append(representatives)

    universe = pd.concat(kept).sort_index()
    if len(universe) > ucfg["max_universe_size"]:
        universe = (
            universe.sort_values(
                "sharpe_3y", ascending=False, na_position="last"
            )
            .head(ucfg["max_universe_size"])
            .sort_index()
        )
    return universe


def save_universe(df: pd.DataFrame, as_of: pd.Timestamp) -> Path:
    """Persist a versioned universe snapshot to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = as_of.strftime("%Y%m%d")
    path = PROCESSED_DIR / f"universe_{stamp}.csv"
    df.to_csv(path)
    return path


def load_universe(as_of: pd.Timestamp) -> pd.DataFrame:
    """Load a previously saved universe snapshot."""
    stamp = as_of.strftime("%Y%m%d")
    path = PROCESSED_DIR / f"universe_{stamp}.csv"
    return pd.read_csv(path, index_col="ticker")
