# backend/src/neurocampus/models/utils_boltzmann.py
import numpy as np
from typing import Tuple

EPS = 1e-8

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid numéricamente estable."""
    # clip para evitar overflow en exp(-x)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def bernoulli_sample(p: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
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

def batch_iter(X: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    """Iterador de mini-batches con barajado."""
    if rng is None:
        rng = np.random.default_rng()
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j]

def binarize(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binariza features en {0,1} con umbral dado."""
    return (X >= threshold).astype(np.float32)

def energy(v: np.ndarray, h: np.ndarray, W: np.ndarray, bv: np.ndarray, bh: np.ndarray) -> float:
    """Energía de una configuración (para debug/diagnóstico)."""
    return - (np.sum(v @ W * h) + v.dot(bv).sum() + h.dot(bh).sum())
