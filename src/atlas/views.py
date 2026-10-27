"""View builder: convert each signal family into Black-Litterman
inputs P, Q, Omega (L3, S4).

Stage 8. Each view family contributes rows to P (which portfolio the
view is about: one-hot for a per-asset view, equal-weight within a
bucket for the regime class view), Q (that portfolio's absolute
expected return: equilibrium prior + tilt), and a confidence scalar
that `assemble` converts to Omega. Honors `config.modules`' on/off
flags -- the same flags the ablation study (stage 11, notebook 11)
will flip -- so a disabled family contributes zero rows, not a
zeroed-out row.

V4 (sentiment) is included for completeness/live use, but is not part
of the stage-8 historical backtest: sentiment is fundamentally
live-only (see sentiment.py), so a caller passes an empty/zero Series
for it there.

Public API:
    ViewSet                                       # p_rows, q_rows,
                                                   # confidence, labels
    regime_view(posture, class_bucket, prior, cfg) -> ViewSet   # V1
    meanreversion_views(view_series, prior, cfg) -> ViewSet     # V2
    technical_views(view_series, prior, cfg) -> ViewSet         # V3
    sentiment_views(view_series, prior, cfg) -> ViewSet         # V4
    momentum_views(view_series, prior, cfg) -> ViewSet          # V5
    assemble(view_sets, tickers, cov, cfg) -> (P, Q, Omega, labels)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_TARGET_BUCKETS = {
    "risk_on": ["equity"],
    "risk_off": ["fixed_income", "commodity"],
    "neutral": [],
}

# Reselection task (ANALYSIS_V4 external-review Phase C): risk_off's
# commodity-bucket target lumps in USO/DBC/SLV alongside GLD, and only
# gold is plausibly defensive (oil went negative in April 2020).
# `black_litterman.regime_targets.risk_off` selects among these three
# BUCKET-LEVEL target lists; "legacy" (the default -- absent config key
# included) is byte-identical to the original hardcoded
# `REGIME_TARGET_BUCKETS["risk_off"]` above. "bonds_gold" additionally
# needs an ASSET-LEVEL (not bucket-averaged) row for gold specifically,
# handled separately in `_risk_off_targets` below via
# `black_litterman.regime_targets.risk_off_assets` -- no ticker is
# hardcoded here; the caller's config supplies the asset list.
_RISK_OFF_BUCKET_TARGETS = {
    "legacy": ["fixed_income", "commodity"],
    "bonds_cash": ["fixed_income", "cash"],
    "bonds_gold": ["fixed_income"],
}


def _risk_off_targets(
    class_bucket: pd.Series, tickers: pd.Index, cfg: dict
) -> list[tuple[pd.Index, str]]:
    """(members, label_suffix) pairs for the risk_off posture's target,
    bucket-level entries plus `bonds_gold`'s extra asset-level row.
    Unknown `regime_targets.risk_off` values fall back to "legacy"."""
    regime_targets = cfg["black_litterman"].get("regime_targets", {})
    variant = regime_targets.get("risk_off", "legacy")
    bucket_names = _RISK_OFF_BUCKET_TARGETS.get(
        variant, _RISK_OFF_BUCKET_TARGETS["legacy"]
    )

    targets = []
    for bucket in bucket_names:
        members = class_bucket[class_bucket == bucket].index
        if len(members) > 0:
            targets.append((members, bucket))

    if variant == "bonds_gold":
        asset_list = regime_targets.get("risk_off_assets", [])
        members = pd.Index([t for t in asset_list if t in tickers])
        if len(members) > 0:
            targets.append((members, "_".join(members)))

    return targets


@dataclass
class ViewSet:
    """One view family's contribution: parallel lists, one entry per
    view row."""

    p_rows: list = field(default_factory=list)
    q_rows: list = field(default_factory=list)
    confidence: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.q_rows)


def regime_view(
    posture: str, class_bucket: pd.Series, prior: pd.Series, cfg: dict
) -> ViewSet:
    """V1 (S4/S7): the HMM posture tilts a class bucket's average
    return. risk_on -> equity; risk_off -> a config-selectable target
    (default: fixed_income + commodity, byte-identical to the original
    hardcoded behavior -- see `_risk_off_targets`); neutral -> no view
    (an empty ViewSet)."""
    view_set = ViewSet()
    if not cfg["modules"]["regime_view"]:
        return view_set

    bcfg = cfg["black_litterman"]
    tickers = prior.index
    if posture == "risk_off":
        targets = _risk_off_targets(class_bucket, tickers, cfg)
    else:
        targets = [
            (class_bucket[class_bucket == bucket].index, bucket)
            for bucket in REGIME_TARGET_BUCKETS.get(posture, [])
        ]

    for members, label_suffix in targets:
        if len(members) == 0:
            continue
        p_row = pd.Series(0.0, index=tickers)
        p_row[members] = 1.0 / len(members)
        target_prior = float(p_row @ prior)
        view_set.p_rows.append(p_row.to_numpy())
        view_set.q_rows.append(target_prior + bcfg["regime_view_magnitude"])
        view_set.confidence.append(bcfg["view_confidence"]["regime"])
        view_set.labels.append(f"V1_regime_{label_suffix}")
    return view_set


def _per_asset_views(
    view_series: pd.Series,
    prior: pd.Series,
    cfg: dict,
    module_key: str,
    confidence_key: str,
    label_prefix: str,
) -> ViewSet:
    view_set = ViewSet()
    if not cfg["modules"][module_key]:
        return view_set

    tickers = prior.index
    aligned = view_series.reindex(tickers).fillna(0.0)
    confidence = cfg["black_litterman"]["view_confidence"][confidence_key]
    for ticker in tickers:
        value = aligned[ticker]
        if value == 0.0:
            continue
        p_row = pd.Series(0.0, index=tickers)
        p_row[ticker] = 1.0
        view_set.p_rows.append(p_row.to_numpy())
        view_set.q_rows.append(float(prior[ticker] + value))
        view_set.confidence.append(confidence)
        view_set.labels.append(f"{label_prefix}_{ticker}")
    return view_set


def meanreversion_views(
    view_series: pd.Series, prior: pd.Series, cfg: dict
) -> ViewSet:
    """V2 (S10): per-asset OU-implied reversion drift as an absolute
    view on that asset's return."""
    return _per_asset_views(
        view_series, prior, cfg, "meanreversion_view", "meanreversion",
        "V2_meanrev",
    )


def technical_views(
    view_series: pd.Series, prior: pd.Series, cfg: dict
) -> ViewSet:
    """V3 (S12): per-asset level-proximity view (price-level component
    only; options positioning is live-only, see technicals.py)."""
    return _per_asset_views(
        view_series, prior, cfg, "technical_view", "technical", "V3_technical"
    )


def sentiment_views(
    view_series: pd.Series, prior: pd.Series, cfg: dict
) -> ViewSet:
    """V4 (S14): per-asset sentiment tilt. Live use only -- pass an
    empty/zero Series for historical backtests (see sentiment.py)."""
    return _per_asset_views(
        view_series, prior, cfg, "sentiment_view", "sentiment", "V4_sentiment"
    )


def momentum_views(
    view_series: pd.Series, prior: pd.Series, cfg: dict
) -> ViewSet:
    """V5 (S4/S11, ANALYSIS_V2.md action 1): per-asset dual-momentum
    view (see momentum.momentum_view) as an absolute view on that
    asset's return, same pattern as V2/V3."""
    return _per_asset_views(
        view_series, prior, cfg, "momentum_view", "momentum", "V5_momentum"
    )


def assemble(
    view_sets: list[ViewSet], tickers: pd.Index, cov: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Stack all view sets into P, Q, Omega (S4). Omega's diagonal is
    each view portfolio's prior variance (tau * Sigma), scaled
    inversely by its confidence: higher confidence -> smaller Omega ->
    more posterior influence. (None, None, None, []) if there are no
    views at all (e.g. neutral posture and no V2/V3 signals fired)."""
    tau = cfg["black_litterman"]["tau"]
    tau_cov = tau * cov.loc[tickers, tickers].to_numpy()

    p_rows, q_rows, confidences, labels = [], [], [], []
    for view_set in view_sets:
        p_rows.extend(view_set.p_rows)
        q_rows.extend(view_set.q_rows)
        confidences.extend(view_set.confidence)
        labels.extend(view_set.labels)

    if not p_rows:
        return None, None, None, []

    P = np.vstack(p_rows)
    Q = np.array(q_rows)
    omega_diag = np.array(
        [
            (P[i] @ tau_cov @ P[i].T) / confidences[i]
            for i in range(len(confidences))
        ]
    )
    return P, Q, np.diag(omega_diag), labels
