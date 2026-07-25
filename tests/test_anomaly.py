"""Tests for anomaly.py: sequences, autoencoder, GMM latents (stage 4)."""

import numpy as np
import pandas as pd
import pytest

from atlas import anomaly

CFG = {
    "general": {"random_seed": 42},
    "regimes": {"hmm": {"n_states": 3}},
    "anomaly": {
        "sequence_length": 5,
        "latent_dim": 2,
        "hidden_units": 4,
        "threshold_percentile": 99,
        "validation_fraction": 0.2,
        "epochs": 5,
        "patience": 2,
        "batch_size": 16,
    },
}


def test_has_enough_history():
    short = pd.DataFrame(np.zeros((10, 2)))
    long_ = pd.DataFrame(np.zeros((100, 2)))
    assert not anomaly.has_enough_history(short, CFG)
    assert anomaly.has_enough_history(long_, CFG)


def test_build_features_fills_and_drops():
    returns = pd.DataFrame(
        {"A": [0.01, np.nan, 0.02], "B": [np.nan, np.nan, np.nan]}
    )
    features = anomaly.build_features(returns)
    # The all-NaN row is dropped; remaining NaN cells filled with 0.
    assert len(features) == 2
    assert features.isna().sum().sum() == 0


def test_make_sequences_shape_and_content():
    features = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    seq = anomaly.make_sequences(features, seq_len=3)
    assert seq.shape == (8, 3, 2)
    np.testing.assert_array_equal(seq[0], features.iloc[0:3].to_numpy())
    np.testing.assert_array_equal(seq[-1], features.iloc[7:10].to_numpy())


def test_make_sequences_too_short_returns_empty():
    features = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    seq = anomaly.make_sequences(features, seq_len=5)
    assert seq.shape == (0, 5, 2)


def test_fit_autoencoder_flags_injected_anomaly():
    rng = np.random.default_rng(1)
    n = 250
    data = rng.normal(0.0, 0.01, (n, 3))
    data[-10:] *= 8  # injected burst, well inside the validation tail
    returns = pd.DataFrame(data, columns=["A", "B", "C"])
    features = anomaly.build_features(returns)

    result = anomaly.fit_autoencoder(features, CFG)
    errors = anomaly.reconstruction_error(result, features)
    flags = anomaly.is_anomalous(errors, result.threshold)

    assert flags[-5:].any()
    # Threshold is the 99th pct of TRAINING errors, so roughly 1% of
    # the calm early period should flag, not wholesale.
    assert flags[: n - 20].mean() < 0.10


def test_encode_and_latent_gmm_agreement():
    rng = np.random.default_rng(2)
    n = 200
    data = rng.normal(0.0, 0.01, (n, 3))
    features = anomaly.build_features(
        pd.DataFrame(data, columns=["A", "B", "C"])
    )

    result = anomaly.fit_autoencoder(features, CFG)
    latent = anomaly.encode(result, features)
    expected_n_seq = n - CFG["anomaly"]["sequence_length"] + 1
    assert latent.shape == (expected_n_seq, CFG["anomaly"]["latent_dim"])

    gmm_result = anomaly.fit_latent_gmm(latent, CFG)
    assert len(gmm_result.labels) == len(latent)
    self_agreement = anomaly.agreement_score(
        gmm_result.labels, gmm_result.labels
    )
    assert self_agreement == pytest.approx(1.0)
