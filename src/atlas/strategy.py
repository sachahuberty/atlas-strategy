"""Weekly decision pipeline (glues L0-L5). See PROJECT_STRUCTURE.md
section 5.

Stage 3 (this file): V1 regime view only. The HMM posture (from
regimes.py) selects which classical allocation book (from
allocation.py) to run each Friday -- "posture switching," the
project's first full strategy version. Later stages (mean-reversion,
technicals, sentiment, anomaly override, Black-Litterman) will layer
in more views without changing this entry point's contract: a
`backtest.run`-compatible `strategy_fn(as_of, window) -> pd.Series`.

Public API (stage 3):
    regime_switching_strategy(class_bucket, cfg, posture_cfg,
                               market_ticker) -> strategy_fn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import allocation, regimes


def regime_switching_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
) -> Callable[[pd.Timestamp, pd.DataFrame], pd.Series]:
    """Build a `backtest.run`-compatible strategy_fn implementing V1:
    each Friday, detect the HMM posture from trailing market data and
    run that posture's configured allocation book (S4/S7)."""
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]

    def strategy_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        market_returns = window[market_ticker]
        if not regimes.has_enough_history(market_returns, cfg):
            # Not enough trailing data yet for a meaningful HMM read
            # (e.g. the first few weeks of a backtest): default to
            # neutral rather than force a fit on almost no data.
            posture = "neutral"
        else:
            try:
                regime_result = regimes.market_regime(
                    market_returns, cfg, posture_cfg
                )
                posture = regime_result["current_posture"]
            except (ValueError, np.linalg.LinAlgError):
                # HMM EM can occasionally degenerate on a given week's
                # window (a state collapses to ~zero responsibility,
                # producing NaN parameters) -- a known hmmlearn failure
                # mode on live/evolving data, not a logic bug. Fall
                # back to neutral rather than crash the whole backtest
                # over one bad week's fit.
                posture = "neutral"
        posture_spec = posture_cfg["postures"][posture]
        method = posture_spec["allocation_method"]

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)

        if method == "max_sharpe":
            mu = allocation.mean_returns(recent)
            return allocation.max_sharpe(mu, cov, cfg)
        if method == "risk_parity":
            return allocation.risk_parity(cov, cfg)
        if method == "gmv_defensive":
            weights = allocation.gmv(cov, cfg)
            tilt = posture_spec["defensive_class_tilt"]
            return allocation.apply_defensive_tilt(
                weights, class_bucket, tilt, cfg
            )
        raise ValueError(f"Unknown allocation_method: {method}")

    return strategy_fn
