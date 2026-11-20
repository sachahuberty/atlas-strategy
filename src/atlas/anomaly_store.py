"""Cross-process persistence for the anomaly autoencoder (external
review, Part A).

Problem this fixes: `strategy.with_anomaly_override` caches the fitted
autoencoder in a plain closure dict that lives only as long as the
Python process. Every GitHub Actions run is a fresh process, so the
live path was silently refitting the autoencoder from scratch every
week instead of following `anomaly.refit_frequency_days` the way the
backtest does (which keeps one long-lived cache across its whole
walk-forward loop). This module gives the live runner a disk-backed
store so a fitted model -- and everything needed to score with it
without refitting -- survives across runs, the same way `live/
state.json` is the portfolio's memory across runs.

Stored under `live/models/anomaly/` (committed, same spirit as
state.json): the Keras model (`model.keras`) and a small JSON manifest
(`manifest.json`) carrying the fitted scaler's parameters, the
threshold, seq_len, fitted_at, and a hash of exactly the config this
model depends on (the `anomaly` section + the random seed -- NOT the
whole config, so an unrelated config edit elsewhere does not force a
refit).

Both files are written atomically (same rename-into-place pattern as
`state.save_state`): the model is saved to a temp `.keras` path first,
then renamed onto `model.keras`; the manifest is written last, after
the model file is already in place, so a crash between the two steps
leaves `load` seeing either nothing usable (manifest missing) or the
old, still-consistent pair -- never a manifest that claims a model
that is not actually on disk in its final form.

Public API:
    DEFAULT_STORE_DIR
    anomaly_config_hash(cfg) -> str
    LoadResult                                  # dataclass
    AnomalyModelStore
        .load(cfg) -> LoadResult
        .save(result, fitted_at, cfg) -> None
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from . import anomaly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_DIR = PROJECT_ROOT / "live" / "models" / "anomaly"

SCHEMA_VERSION = 1
MODEL_FILENAME = "model.keras"
MANIFEST_FILENAME = "manifest.json"


def anomaly_config_hash(cfg: dict) -> str:
    """Hash of exactly what the autoencoder's fit depends on: the
    `anomaly` config section plus the general random seed. Changing
    anything else in `config.yaml` (e.g. the turnover cap) must not
    force a refit of a model that is unaffected by it."""
    blob = json.dumps(
        {
            "anomaly": cfg["anomaly"],
            "random_seed": cfg["general"]["random_seed"],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class LoadResult:
    """Outcome of `AnomalyModelStore.load`.

    `result`/`fitted_at` are set only when a model was actually loaded
    from disk (`reason is None`). Any other `reason` means there is no
    usable stored model and the caller must refit -- never silently;
    `reason` is a short, human-readable string meant to end up in the
    run's diagnostics (`anomaly_model_source` / `refit_reason`).
    """

    result: anomaly.AutoencoderResult | None
    fitted_at: pd.Timestamp | None
    # None | "missing" | "corrupt: ..." | "config_changed: ..."
    reason: str | None


class AnomalyModelStore:
    """Disk-backed store for one fitted autoencoder, under `store_dir`.

    `load` never raises on an ordinary "nothing usable here" outcome
    (missing files, corrupt JSON/model, a config that has moved on) --
    those are reported via `LoadResult.reason` so the caller can refit
    and record why. Only a genuinely unexpected error propagates.
    """

    def __init__(self, store_dir: str | Path = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)

    @property
    def model_path(self) -> Path:
        return self.store_dir / MODEL_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.store_dir / MANIFEST_FILENAME

    def load(self, cfg: dict) -> LoadResult:
        if not self.manifest_path.exists() or not self.model_path.exists():
            return LoadResult(result=None, fitted_at=None, reason="missing")

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            model = keras.models.load_model(self.model_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            return LoadResult(
                result=None, fitted_at=None, reason=f"corrupt: {exc}"
            )

        if manifest.get("schema_version") != SCHEMA_VERSION:
            bad_version = manifest.get("schema_version")
            return LoadResult(
                result=None,
                fitted_at=None,
                reason=f"corrupt: unknown schema_version {bad_version!r}",
            )

        current_hash = anomaly_config_hash(cfg)
        stored_hash = manifest.get("anomaly_config_hash")
        if stored_hash != current_hash:
            return LoadResult(
                result=None,
                fitted_at=None,
                reason=f"config_changed: {stored_hash} -> {current_hash}",
            )

        try:
            encoder = keras.Model(
                inputs=model.input,
                outputs=model.get_layer("latent").output,
            )
            scaler = StandardScaler()
            scaler.mean_ = np.array(manifest["scaler_mean"])
            scaler.scale_ = np.array(manifest["scaler_scale"])
            scaler.var_ = scaler.scale_**2
            scaler.n_features_in_ = len(scaler.mean_)
            scaler.n_samples_seen_ = np.int64(1)
            result = anomaly.AutoencoderResult(
                model=model,
                encoder=encoder,
                scaler=scaler,
                seq_len=int(manifest["seq_len"]),
                threshold=float(manifest["threshold"]),
                train_errors=np.array([]),
            )
        except (KeyError, ValueError) as exc:
            return LoadResult(
                result=None, fitted_at=None, reason=f"corrupt: {exc}"
            )

        fitted_at = pd.Timestamp(manifest["fitted_at"])
        return LoadResult(result=result, fitted_at=fitted_at, reason=None)

    def save(
        self,
        result: anomaly.AutoencoderResult,
        fitted_at: pd.Timestamp,
        cfg: dict,
    ) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)

        tmp_model_path = self.store_dir / ("tmp_" + MODEL_FILENAME)
        result.model.save(tmp_model_path)
        os.replace(tmp_model_path, self.model_path)

        scaler = result.scaler
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fitted_at": pd.Timestamp(fitted_at).isoformat(),
            "seq_len": int(result.seq_len),
            "threshold": float(result.threshold),
            "scaler_mean": np.asarray(scaler.mean_).tolist(),
            "scaler_scale": np.asarray(scaler.scale_).tolist(),
            "n_features": int(len(scaler.mean_)),
            "anomaly_config_hash": anomaly_config_hash(cfg),
        }
        tmp_manifest_path = self.store_dir / ("tmp_" + MANIFEST_FILENAME)
        with open(tmp_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manifest_path, self.manifest_path)
