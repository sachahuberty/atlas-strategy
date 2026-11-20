"""Tests for anomaly_store.py (external review, Part A): the
cross-process persistence layer that lets the live runner's
autoencoder survive between GitHub Actions runs instead of refitting
from scratch every week.
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas import anomaly, anomaly_store

CFG = {
    "general": {"random_seed": 42},
    "anomaly": {
        "sequence_length": 5,
        "latent_dim": 2,
        "hidden_units": 4,
        "threshold_percentile": 99,
        "validation_fraction": 0.2,
        "epochs": 5,
        "patience": 2,
        "batch_size": 16,
        "refit_frequency_days": 126,
        "blend_to_gmv": 1.0,
    },
}


def _features(seed=1, n=250):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (n, 3))
    return anomaly.build_features(
        pd.DataFrame(data, columns=["A", "B", "C"])
    )


# --------------------------------------------------------------------- #
# anomaly_config_hash
# --------------------------------------------------------------------- #
def test_anomaly_config_hash_stable_under_unrelated_change():
    cfg_a = copy.deepcopy(CFG)
    cfg_b = copy.deepcopy(CFG)
    cfg_b["rebalance"] = {"max_weekly_turnover": 0.01}  # unrelated section
    assert anomaly_store.anomaly_config_hash(
        cfg_a
    ) == anomaly_store.anomaly_config_hash(cfg_b)


def test_anomaly_config_hash_changes_with_anomaly_section():
    cfg_a = copy.deepcopy(CFG)
    cfg_b = copy.deepcopy(CFG)
    cfg_b["anomaly"]["hidden_units"] = 8
    assert anomaly_store.anomaly_config_hash(
        cfg_a
    ) != anomaly_store.anomaly_config_hash(cfg_b)


def test_anomaly_config_hash_changes_with_seed():
    cfg_a = copy.deepcopy(CFG)
    cfg_b = copy.deepcopy(CFG)
    cfg_b["general"]["random_seed"] = 7
    assert anomaly_store.anomaly_config_hash(
        cfg_a
    ) != anomaly_store.anomaly_config_hash(cfg_b)


# --------------------------------------------------------------------- #
# save/load round trip
# --------------------------------------------------------------------- #
def test_save_then_load_roundtrip_scores_identically(tmp_path):
    features = _features()
    result = anomaly.fit_autoencoder(features, CFG)
    fitted_at = pd.Timestamp("2026-10-23")

    store = anomaly_store.AnomalyModelStore(tmp_path / "anomaly")
    store.save(result, fitted_at, CFG)

    # Fresh store instance, same directory -- simulates a separate
    # process picking up what an earlier one persisted.
    reloaded_store = anomaly_store.AnomalyModelStore(tmp_path / "anomaly")
    loaded = reloaded_store.load(CFG)

    assert loaded.reason is None
    assert loaded.fitted_at == fitted_at
    assert loaded.result.seq_len == result.seq_len
    assert loaded.result.threshold == pytest.approx(result.threshold)

    original_errors = anomaly.reconstruction_error(result, features)
    reloaded_errors = anomaly.reconstruction_error(loaded.result, features)
    np.testing.assert_allclose(
        reloaded_errors, original_errors, rtol=1e-5, atol=1e-8
    )


def test_save_creates_model_and_manifest_files(tmp_path):
    features = _features()
    result = anomaly.fit_autoencoder(features, CFG)
    store = anomaly_store.AnomalyModelStore(tmp_path / "anomaly")
    store.save(result, pd.Timestamp("2026-10-23"), CFG)

    assert store.model_path.exists()
    assert store.manifest_path.exists()
    # No leftover temp files from the atomic rename.
    assert list(tmp_path.glob("anomaly/tmp_*")) == []


# --------------------------------------------------------------------- #
# load failure paths -- never raise, always report a reason
# --------------------------------------------------------------------- #
def test_load_missing_store_returns_reason_missing(tmp_path):
    store = anomaly_store.AnomalyModelStore(tmp_path / "nothing_here")
    loaded = store.load(CFG)
    assert loaded.result is None
    assert loaded.reason == "missing"


def test_load_corrupt_manifest_returns_reason_corrupt(tmp_path):
    store_dir = tmp_path / "anomaly"
    store_dir.mkdir()
    (store_dir / anomaly_store.MANIFEST_FILENAME).write_text("{not json")
    (store_dir / anomaly_store.MODEL_FILENAME).write_text("placeholder")

    store = anomaly_store.AnomalyModelStore(store_dir)
    loaded = store.load(CFG)
    assert loaded.result is None
    assert loaded.reason.startswith("corrupt")


def test_load_missing_model_file_returns_reason_missing(tmp_path):
    store_dir = tmp_path / "anomaly"
    store_dir.mkdir()
    (store_dir / anomaly_store.MANIFEST_FILENAME).write_text("{}")
    # model.keras deliberately absent

    store = anomaly_store.AnomalyModelStore(store_dir)
    loaded = store.load(CFG)
    assert loaded.result is None
    assert loaded.reason == "missing"


def test_load_config_changed_returns_reason(tmp_path):
    features = _features()
    result = anomaly.fit_autoencoder(features, CFG)
    store = anomaly_store.AnomalyModelStore(tmp_path / "anomaly")
    store.save(result, pd.Timestamp("2026-10-23"), CFG)

    changed_cfg = copy.deepcopy(CFG)
    changed_cfg["anomaly"]["hidden_units"] = 999

    loaded = store.load(changed_cfg)
    assert loaded.result is None
    assert loaded.reason.startswith("config_changed")


def test_load_unknown_schema_version_returns_reason_corrupt(tmp_path):
    features = _features()
    result = anomaly.fit_autoencoder(features, CFG)
    store = anomaly_store.AnomalyModelStore(tmp_path / "anomaly")
    store.save(result, pd.Timestamp("2026-10-23"), CFG)

    manifest_path = store.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    loaded = store.load(CFG)
    assert loaded.result is None
    assert loaded.reason.startswith("corrupt")
