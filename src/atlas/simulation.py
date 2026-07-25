"""Monte Carlo / GBM simulation (L4 forward risk, S8).

Stage 10.

Planned public API:
    business_days(start, end) -> list
    gbm_paths(mu, sigma, dates, n) -> np.ndarray
    scenario_summary(paths, horizons) -> pd.DataFrame   # histograms, P(positive)
    recovery_analysis(paths, peak) -> pd.DataFrame
"""
