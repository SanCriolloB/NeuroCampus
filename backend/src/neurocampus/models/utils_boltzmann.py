# backend/src/neurocampus/models/utils_boltzmann.py
from __future__ import annotations
from typing import Iterator
import numpy as np

__all__ = [
    "EPS",
    "sigmoid",
    "bernoulli_sample",
    "check_numeric_matrix",
    "batch_iter",
    "binarize",
    "energy_rbm",
    "energy_bm",
    "gibbs_step_vh",
]

EPS = 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid numéricamente estable."""
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float64)).astype(np.float64)


def bernoulli_sample(
    p: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Muestreo Bernoulli a partir de probabilidades p."""
    if rng is None:
        rng = np.random.default_rng()
    return (p > rng.random(p.shape)).astype(np.float32)


def check_numeric_matrix(X: np.ndarray, name: str = "X") -> None:
    """Valida que el array sea 2D, finito y sin NaNs."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} debe ser np.ndarray")
    if X.ndim != 2:
        raise ValueError(f"{name} debe ser 2D (shape=(n_samples, n_features))")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contiene inf/NaN")


def batch_iter(
    X: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[np.ndarray]:
    """Iterador de mini-batches con barajado."""
    if rng is None:
        rng = np.random.default_rng()
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield X[j]


def binarize(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binariza features en {0,1} con umbral dado."""
    return (X >= threshold).astype(np.float32)


# ---------------- Gibbs sampling helpers ----------------


def gibbs_step_vh(
    v: np.ndarray,
    W: np.ndarray,
    b_v: np.ndarray,
    b_h: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """
    Un paso de Gibbs v -> h -> v para RBM binaria.

    Parámetros
    ----------
    v : (batch_size, n_visible)
        Estado visible actual (0/1 o probabilidades).
    W : (n_visible, n_hidden)
        Pesos visibles-ocultos.
    b_v : (n_visible,)
        Bias de visibles.
    b_h : (n_hidden,)
        Bias de ocultas.
    rng : np.random.Generator, opcional
        Generador de aleatoriedad (para reproducibilidad).

    Devuelve
    --------
    v_sample : np.ndarray
        Muestra binaria de visibles después del paso Gibbs.
    v_prob : np.ndarray
        Probabilidades de visibles p(v=1 | h).
    h_sample : np.ndarray
        Muestra binaria de ocultas p(h=1 | v).
    h_prob : np.ndarray
        Probabilidades de ocultas p(h=1 | v).
    """
    if rng is None:
        rng = np.random.default_rng()

    # v -> h
    h_lin = v @ W + b_h
    h_prob = sigmoid(h_lin).astype(np.float32)
    h_sample = bernoulli_sample(h_prob, rng)

    # h -> v
    v_lin = h_sample @ W.T + b_v
    v_prob = sigmoid(v_lin).astype(np.float32)
    v_sample = bernoulli_sample(v_prob, rng)

    return v_sample, v_prob, h_sample, h_prob


# ---------------- Energías ----------------


def energy_rbm(
    v: np.ndarray,
    h: np.ndarray,
    W: np.ndarray,
    b_v: np.ndarray,
    b_h: np.ndarray,
) -> float:
    """
    Energía de una RBM binaria:
        E(v,h) = - v^T W h - b_v^T v - b_h^T h
    """
    v = v.astype(np.float64, copy=False)
    h = h.astype(np.float64, copy=False)
    W = W.astype(np.float64, copy=False)
    b_v = b_v.astype(np.float64, copy=False)
    b_h = b_h.astype(np.float64, copy=False)
    term = -(v @ W @ h) - (b_v @ v) - (b_h @ h)
    return float(term)


def energy_bm(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    """
    Energía de una Máquina de Boltzmann (visible+oculta apilados) con matriz simétrica:
        E(x) = - 0.5 * x^T W x - b^T x
    Asume diag(W)=0.
    """
    x = x.astype(np.float64, copy=False)
    W = W.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    W_sym = 0.5 * (W + W.T)
    np.fill_diagonal(W_sym, 0.0)
    term = -0.5 * (x @ W_sym @ x) - (b @ x)
    return float(term)
