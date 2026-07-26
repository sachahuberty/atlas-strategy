"""Monte Carlo / GBM simulation (L4 forward risk, S8).

Stage 10. Single-asset (portfolio-level) GBM: the strategy's own
empirically estimated annualized drift/vol become the process
parameters, simulated forward as a fan chart of a total-return index
starting at 1.0 -- not a per-asset multi-factor model (course
convention: "GBM Monte Carlo on strategy returns").

Public API:
    business_days(start, n_days) -> pd.DatetimeIndex
    gbm_paths(mu, sigma, dates, n_paths, seed) -> np.ndarray
    scenario_summary(paths, horizons_years) -> pd.DataFrame
    recovery_analysis(paths, peak) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def business_days(start: pd.Timestamp, n_days: int) -> pd.DatetimeIndex:
    """`n_days` trading days strictly after `start` (S8): calendar
    labels for the simulated horizon, e.g. a fan-chart x-axis."""
    return pd.bdate_range(start=start, periods=n_days + 1)[1:]


def gbm_paths(
    mu: float,
    sigma: float,
    dates: pd.DatetimeIndex,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Simulate `n_paths` GBM total-return-index paths over `dates`
    (S8), given annualized drift `mu` and vol `sigma`. Deterministic
    for a given `seed`. Returns an `(n_paths, len(dates) + 1)` array;
    column 0 is 1.0 (today, before `dates[0]`)."""
    rng = np.random.default_rng(seed)
    n_days = len(dates)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    shocks = rng.standard_normal((n_paths, n_days))
    log_returns = drift + diffusion * shocks
    log_paths = np.cumsum(log_returns, axis=1)
    return np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def scenario_summary(
    paths: np.ndarray,
    horizons_years: list[int],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Terminal-value distribution at each horizon (S8): mean/median
    growth, 5th/95th percentile, and P(positive) = fraction of paths
    above breakeven (1.0) by that horizon."""
    n_days = paths.shape[1] - 1
    rows = []
    for years in horizons_years:
        day = min(years * periods_per_year, n_days)
        terminal = paths[:, day]
        rows.append(
            {
                "horizon_years": years,
                "mean_growth": float(terminal.mean()),
                "median_growth": float(np.median(terminal)),
                "p05_growth": float(np.percentile(terminal, 5)),
                "p95_growth": float(np.percentile(terminal, 95)),
                "prob_positive": float((terminal > 1.0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("horizon_years")


def recovery_analysis(paths: np.ndarray, peak: float = 1.0) -> pd.DataFrame:
    """Per-path days to reach `peak`, a growth factor relative to
    today (S8): e.g. `peak=1.05` if the book currently sits 4.8%
    below its prior all-time high and needs +5% to recover it.
    Defaults to breakeven (1.0, no current drawdown to recover from).
    Paths already at or above `peak` recover immediately (0 days);
    paths that never reach it within the simulated horizon are
    censored (`recovered` False, `recovery_days` NaN)."""
    rows = []
    for path in paths:
        reached = path >= peak
        if reached.any():
            rows.append(
                {"recovered": True, "recovery_days": int(np.argmax(reached))}
            )
        else:
            rows.append({"recovered": False, "recovery_days": np.nan})
    return pd.DataFrame(rows)
