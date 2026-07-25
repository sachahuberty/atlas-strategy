"""Mean-reversion signal (L2 view V2): ADF, OU half-life, z-scores (S10).

Stage 5.
Signal = detrended log-price z-score, gated by ADF stationarity of the deviation,
half-life filter, and volatility filter. Vectorized, no row loops.

Planned public API:
    adf_pvalue(series) -> float
    ou_half_life(series) -> float
    zscore(series, lookback) -> pd.Series
    reversion_signal(prices, cfg, as_of) -> pd.DataFrame   # per-asset score + gates
"""
