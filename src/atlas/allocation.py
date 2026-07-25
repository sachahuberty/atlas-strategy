"""Optimizers (L3): Black-Litterman fusion + all classical books (S1-S5).

Stages 2 (classical) and 8 (BL).
All functions return pd.Series of weights indexed by ticker; weights >= 0, sum = 1.
Constraints (caps, TE budget) from config. Covariance: Ledoit-Wolf shrinkage.

Planned public API:
    equilibrium_returns(market_weights, cov, delta) -> pd.Series
    black_litterman(prior, cov, P, Q, Omega, tau) -> pd.Series   # posterior mu
    max_sharpe(mu, cov, cfg) -> weights          # SLSQP (S2/S3)
    gmv(returns, cfg) -> weights                 # S5
    risk_parity(returns, cfg) -> weights         # S5 risk contributions
    hrp(returns) -> weights                      # S4 dendrogram
    tracking_error_min(returns, benchmark, cfg) -> weights   # S2
    permanent() / sixty_forty() -> weights       # benchmarks
    utility_select(candidates, cfg) -> weights   # U-comparison vs defensive book
"""
