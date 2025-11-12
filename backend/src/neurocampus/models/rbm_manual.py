# backend/src/neurocampus/models/rbm_manual.py
from __future__ import annotations
import numpy as np
from typing import Optional
from .utils_boltzmann import (
    EPS, sigmoid, bernoulli_sample, check_numeric_matrix, batch_iter, binarize
)

class RestrictedBoltzmannMachine:
    """
    RBM binaria (visibles y ocultas en {0,1}) con CD-1.
    Interfaz:
      - fit(X, epochs, batch_size, verbose)
      - transform_hidden(X) -> probs h|v
      - reconstruct(X) -> v' (probs)
    """
    def __init__(
        self,
        n_visible: int,
        n_hidden: int = 64,
        learning_rate: float = 0.05,
        seed: int = 42,
        l2: float = 0.0,
        clip_grad: Optional[float] = 1.0,
        binarize_input: bool = False,
        input_bin_threshold: float = 0.5,
    ):
        self.n_visible = int(n_visible)
        self.n_hidden  = int(n_hidden)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.clip_grad = None if clip_grad is None else float(clip_grad)
        self.binarize_input = bool(binarize_input)
        self.input_bin_threshold = float(input_bin_threshold)

        self.rng = np.random.default_rng(seed)
        # Xavier-like init
        scale = 1.0 / np.sqrt(self.n_visible + self.n_hidden)
        self.W  = self.rng.normal(0.0, scale, size=(self.n_visible, self.n_hidden)).astype(np.float32)
        self.bv = np.zeros(self.n_visible, dtype=np.float32)
        self.bh = np.zeros(self.n_hidden, dtype=np.float32)

    # ---- Condicionales ----
    def _p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.bh)

    def _p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.bv)

    def _sample_h(self, v: np.ndarray) -> np.ndarray:
        return bernoulli_sample(self._p_h_given_v(v), self.rng)

    def _sample_v(self, h: np.ndarray) -> np.ndarray:
        return bernoulli_sample(self._p_v_given_h(h), self.rng)

    # ---- API pública ----
    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 64, verbose: int = 1) -> "RestrictedBoltzmannMachine":
        X = np.asarray(X, dtype=np.float32)
        check_numeric_matrix(X, "X")
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)

        lr = self.learning_rate
        for ep in range(1, epochs + 1):
            for v0 in batch_iter(X, batch_size, self.rng):
                # Fase positiva
                ph_v0 = self._p_h_given_v(v0)
                h0    = bernoulli_sample(ph_v0, self.rng)

                # Fase negativa (CD-1)
                pv_h0 = self._p_v_given_h(h0)
                v1    = bernoulli_sample(pv_h0, self.rng)
                ph_v1 = self._p_h_given_v(v1)

                # Gradientes
                dW = (v0.T @ ph_v0 - v1.T @ ph_v1) / v0.shape[0]
                dbv = np.mean(v0 - v1, axis=0)
                dbh = np.mean(ph_v0 - ph_v1, axis=0)

                # Regularización L2
                if self.l2 > 0.0:
                    dW -= self.l2 * self.W

                # Clipping
                if self.clip_grad is not None:
                    np.clip(dW, -self.clip_grad, self.clip_grad, out=dW)
                    np.clip(dbv, -self.clip_grad, self.clip_grad, out=dbv)
                    np.clip(dbh, -self.clip_grad, self.clip_grad, out=dbh)

                # Update
                self.W  += lr * dW.astype(np.float32)
                self.bv += lr * dbv.astype(np.float32)
                self.bh += lr * dbh.astype(np.float32)

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                # Reconstrucción simple para logging
                pvh = self._p_v_given_h(self._p_h_given_v(X[:128]))
                mse = float(np.mean((X[:128] - pvh)**2))
                print(f"[RBM] epoch={ep:03d} mse_recon={mse:.6f}")
        return self

    def transform_hidden(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)
        return self._p_h_given_v(X).astype(np.float32)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)
        H = self._p_h_given_v(X)
        V = self._p_v_given_h(H)
        return V.astype(np.float32)
