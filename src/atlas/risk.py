"""Risk overlays (L4): anomaly override, regime caps, stress tests (S7/S8).

Stages 4 (override) and 10 (stress).

Planned public API:
    apply_anomaly_override(weights, gmv_weights, flag, blend) -> weights
    apply_regime_caps(weights, posture, cfg) -> weights
    historical_stress(weights, returns, scenarios) -> pd.DataFrame
    sensitivity_stress(weights, shocks) -> pd.DataFrame
    reverse_stress(weights, returns) -> dict     # what kills this book?
"""
