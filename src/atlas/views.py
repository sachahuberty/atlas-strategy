"""View builder: convert signals into Black-Litterman inputs P, Q, Omega (L3).

Stage 8.
Each enabled view family (config.modules) contributes rows to P (picks), Q (expected
excess returns) and Omega (uncertainty). Confidence scalers from config, optionally
scaled by HMM/GMM regime-agreement score.

Planned public API:
    regime_view(posture, buckets, cfg) -> ViewSet          # V1
    meanreversion_views(signal_df, cfg) -> ViewSet         # V2
    technical_views(levels_df, cfg) -> ViewSet             # V3
    sentiment_views(sent_series, cfg) -> ViewSet           # V4
    assemble(view_sets, universe) -> (P, Q, Omega)
"""
