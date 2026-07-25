"""Autoencoder anomaly detection + GMM latent regimes (L1, S8).

Stage 4. Trains on a trailing window with a chronological
train/validation split (scaler fit on train only), threshold set from
TRAINING reconstruction errors only (never validation/eval errors).
Refitting on the fold schedule (S8) is handled by the caller: for now
that's strategy.py's lightweight time-based cache (refit every
`anomaly.refit_frequency_days`), a stand-in for the real per-fold
walk-forward persistence that lands in stage 9 (models/ directory).

Public API:
    has_enough_history(features, cfg) -> bool
    build_features(returns) -> pd.DataFrame
    make_sequences(features, seq_len) -> np.ndarray
    fit_autoencoder(features, cfg) -> AutoencoderResult
    reconstruction_error(result, features) -> np.ndarray
    encode(result, features) -> np.ndarray
    is_anomalous(errors, threshold) -> np.ndarray[bool]
    fit_latent_gmm(latent, cfg) -> GMMResult
    agreement_score(labels_a, labels_b) -> float
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def has_enough_history(features: pd.DataFrame, cfg: dict) -> bool:
    """Whether there's enough trailing data for a meaningful
    autoencoder fit: at least one sequence window, plus enough rows
    for a non-trivial train/validation split."""
    acfg = cfg["anomaly"]
    min_obs = acfg["sequence_length"] + 30
    return len(features.dropna()) >= min_obs


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily returns are already stationary (S8 rule); this is the
    autoencoder's raw feature matrix, one column per asset."""
    return returns.dropna(how="all").fillna(0.0)


def make_sequences(features: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Overlapping `seq_len`-day windows: shape (n_sequences, seq_len,
    n_features), each ending on its as-of date."""
    values = features.to_numpy()
    n, n_features = values.shape
    if n < seq_len:
        return np.empty((0, seq_len, n_features))
    return np.stack([values[i : i + seq_len] for i in range(n - seq_len + 1)])


@dataclass
class AutoencoderResult:
    model: keras.Model
    encoder: keras.Model
    scaler: StandardScaler
    seq_len: int
    threshold: float
    train_errors: np.ndarray


def _build_model(
    seq_len: int,
    n_features: int,
    latent_dim: int,
    hidden_units: int,
    seed: int,
) -> tuple[keras.Model, keras.Model]:
    tf.keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(seq_len, n_features))
    encoded = keras.layers.LSTM(hidden_units, activation="tanh")(inputs)
    latent = keras.layers.Dense(latent_dim, name="latent")(encoded)
    repeated = keras.layers.RepeatVector(seq_len)(latent)
    decoded = keras.layers.LSTM(
        hidden_units, activation="tanh", return_sequences=True
    )(repeated)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(
        decoded
    )

    autoencoder = keras.Model(inputs, outputs, name="sequential_autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def reconstruction_error(
    result: AutoencoderResult, features: pd.DataFrame
) -> np.ndarray:
    """Per-sequence MSE between `features` and its reconstruction."""
    scaled = result.scaler.transform(features.to_numpy())
    seq = make_sequences(
        pd.DataFrame(scaled, index=features.index), result.seq_len
    )
    if len(seq) == 0:
        return np.array([])
    reconstructed = result.model.predict(seq, verbose=0)
    return np.mean((seq - reconstructed) ** 2, axis=(1, 2))


def encode(result: AutoencoderResult, features: pd.DataFrame) -> np.ndarray:
    """Latent-space representation of `features` (S7/S8)."""
    scaled = result.scaler.transform(features.to_numpy())
    seq = make_sequences(
        pd.DataFrame(scaled, index=features.index), result.seq_len
    )
    if len(seq) == 0:
        return np.empty((0, result.encoder.output_shape[-1]))
    return result.encoder.predict(seq, verbose=0)


def is_anomalous(errors: np.ndarray, threshold: float) -> np.ndarray:
    return errors > threshold


def fit_autoencoder(features: pd.DataFrame, cfg: dict) -> AutoencoderResult:
    """Fit the sequential autoencoder on a trailing window (S8):
    chronological train/validation split, scaler fit on train only,
    early stopping on validation loss, threshold from TRAINING
    reconstruction errors only."""
    acfg = cfg["anomaly"]
    seed = cfg["general"]["random_seed"]
    seq_len = acfg["sequence_length"]

    n_val = max(0, int(len(features) * acfg["validation_fraction"]))
    split = len(features) - n_val
    train_features = features.iloc[:split]
    val_features = features.iloc[split:]

    scaler = StandardScaler().fit(train_features.to_numpy())
    train_scaled = pd.DataFrame(
        scaler.transform(train_features.to_numpy()), index=train_features.index
    )
    train_seq = make_sequences(train_scaled, seq_len)

    val_seq = np.empty((0, seq_len, features.shape[1]))
    if len(val_features) >= seq_len:
        val_scaled = pd.DataFrame(
            scaler.transform(val_features.to_numpy()),
            index=val_features.index,
        )
        val_seq = make_sequences(val_scaled, seq_len)

    model, encoder = _build_model(
        seq_len,
        features.shape[1],
        acfg["latent_dim"],
        acfg["hidden_units"],
        seed,
    )
    callbacks = []
    fit_kwargs = {}
    if len(val_seq) > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=acfg["patience"],
                restore_best_weights=True,
            )
        )
        fit_kwargs["validation_data"] = (val_seq, val_seq)
    model.fit(
        train_seq,
        train_seq,
        epochs=acfg["epochs"],
        batch_size=acfg["batch_size"],
        verbose=0,
        callbacks=callbacks,
        **fit_kwargs,
    )

    result = AutoencoderResult(
        model=model,
        encoder=encoder,
        scaler=scaler,
        seq_len=seq_len,
        threshold=0.0,
        train_errors=np.array([]),
    )
    train_errors = reconstruction_error(result, train_features)
    result.threshold = float(
        np.percentile(train_errors, acfg["threshold_percentile"])
    )
    result.train_errors = train_errors
    return result


@dataclass
class GMMResult:
    model: GaussianMixture
    labels: np.ndarray


def fit_latent_gmm(latent: np.ndarray, cfg: dict) -> GMMResult:
    """GMM clustering in the autoencoder's latent space (S7/S8), for
    an independent regime cross-check against the HMM. Uses the same
    number of components as the HMM for a fair agreement comparison."""
    n_components = cfg["regimes"]["hmm"]["n_states"]
    seed = cfg["general"]["random_seed"]
    model = GaussianMixture(
        n_components=n_components, random_state=seed, n_init=5
    )
    labels = model.fit_predict(latent)
    return GMMResult(model=model, labels=labels)


def agreement_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two label sequences (e.g. HMM
    states vs. GMM latent clusters): 1.0 = perfect agreement (up to
    relabeling), ~0.0 = no better than chance."""
    return float(adjusted_rand_score(labels_a, labels_b))
