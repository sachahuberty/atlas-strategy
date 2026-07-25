"""Regime detection (L1): HMM primary switch + K-Means macro context (S7/S8).

Stage 3.
Rules: train-only scaler fits; posture assignment by cluster PROFILE, never index
(see config/regime_posture.yaml); HMM decodes weekly, refit per config.

Planned public API:
    fit_hmm(returns, vol, cfg) -> HMMResult          # states, transmat, persistence
    decode_state(model, features_asof) -> str        # posture name
    fit_kmeans_macro(macro_df, cfg) -> KMeansResult  # elbow, PCA coords, heatmap data
    map_states_to_postures(model, features) -> dict  # profile-based mapping
"""
