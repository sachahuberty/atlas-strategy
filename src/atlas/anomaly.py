"""Autoencoder anomaly detection + GMM latent regimes (L1, S8).

Stage 4.
Rules: stationary features only; 5-day sequences; chronological split before scaling;
threshold = percentile of TRAINING reconstruction errors; persist model+scaler per fold.

Planned public API:
    build_features(prices, vix) -> pd.DataFrame
    make_sequences(features, seq_len) -> np.ndarray
    fit_autoencoder(train_seq, cfg) -> AEResult      # early stopping on val loss
    anomaly_score(model, seq_asof) -> float
    is_anomalous(score, threshold) -> bool
    latent_regimes_gmm(latent, n) -> labels          # cross-check vs HMM
"""
