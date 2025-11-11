# backend/src/neurocampus/models/bm_manual.py
from __future__ import annotations
import numpy as np
from typing import Optional
from .utils_boltzmann import (
    EPS, sigmoid, bernoulli_sample, check_numeric_matrix, batch_iter, binarize
)

class BoltzmannMachine:
    """
    Máquina de Boltzmann “general” binaria con dos capas (visibles/ocultas) y
    posibilidad de acoplos intra-capa (Wvv, Whh). Por defecto solo usa Wvh como una RBM,
    pero se exponen matrices intra-capa para extender.
    Interfaz:
      - fit(X, epochs, batch_size, verbose)
      - transform_hidden(X) -> probs h|v
      - reconstruct(X) -> v' (probs)
    NOTA: El entrenamiento sigue un CD-1 aproximado.
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
        use_intra_connect: bool = False,   # activa Wvv y Whh
        intra_scale: float = 0.0,          # init std para intra-capa
    ):
        self.n_visible = int(n_visible)
        self.n_hidden  = int(n_hidden)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.clip_grad = None if clip_grad is None else float(clip_grad)
        self.binarize_input = bool(binarize_input)
        self.input_bin_threshold = float(input_bin_threshold)
        self.use_intra_connect = bool(use_intra_connect)

        self.rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.n_visible + self.n_hidden)

        # Inter-capa (como RBM)
        self.Wvh = self.rng.normal(0.0, scale, size=(self.n_visible, self.n_hidden)).astype(np.float32)
        self.bv  = np.zeros(self.n_visible, dtype=np.float32)
        self.bh  = np.zeros(self.n_hidden,  dtype=np.float32)

        # Intra-capa (opcional)
        if self.use_intra_connect:
            s = float(intra_scale) if intra_scale > 0 else scale * 0.1
            self.Wvv = self.rng.normal(0.0, s, size=(self.n_visible, self.n_visible)).astype(np.float32)
            self.Whh = self.rng.normal(0.0, s, size=(self.n_hidden,  self.n_hidden )).astype(np.float32)
            np.fill_diagonal(self.Wvv, 0.0)
            np.fill_diagonal(self.Whh, 0.0)
        else:
            self.Wvv = np.zeros((self.n_visible, self.n_visible), dtype=np.float32)
            self.Whh = np.zeros((self.n_hidden,  self.n_hidden ), dtype=np.float32)

    # ---- Campos locales con intra-capa ----
    def _field_h(self, v: np.ndarray, h: Optional[np.ndarray] = None) -> np.ndarray:
        base = v @ self.Wvh + self.bh
        if self.use_intra_connect and h is not None:
            base = base + h @ self.Whh.T
        return base

    def _field_v(self, h: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
        base = h @ self.Wvh.T + self.bv
        if self.use_intra_connect and v is not None:
            base = base + v @ self.Wvv.T
        return base

    def _p_h_given(self, v: np.ndarray, h: Optional[np.ndarray] = None) -> np.ndarray:
        return sigmoid(self._field_h(v, h))

    def _p_v_given(self, h: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
        return sigmoid(self._field_v(h, v))

    # ---- API pública (CD-1) ----
    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 64, verbose: int = 1) -> "BoltzmannMachine":
        X = np.asarray(X, dtype=np.float32)
        check_numeric_matrix(X, "X")
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)

        lr = self.learning_rate
        for ep in range(1, epochs + 1):
            for v0 in batch_iter(X, batch_size, self.rng):
                # Fase positiva
                ph_v0 = self._p_h_given(v0)
                h0    = bernoulli_sample(ph_v0, self.rng)

                # Fase negativa (un paso)
                pv_h0 = self._p_v_given(h0, v0 if self.use_intra_connect else None)
                v1    = bernoulli_sample(pv_h0, self.rng)
                ph_v1 = self._p_h_given(v1, h0 if self.use_intra_connect else None)

                # Gradientes inter-capa (RBM-like)
                dWvh = (v0.T @ ph_v0 - v1.T @ ph_v1) / v0.shape[0]
                dbv  = np.mean(v0 - v1, axis=0)
                dbh  = np.mean(ph_v0 - ph_v1, axis=0)

                # Gradientes intra-capa (si aplica) - heurística simétrica
                if self.use_intra_connect:
                    dWvv = ((v0.T @ v0) - (v1.T @ v1)) / v0.shape[0]
                    dWhh = ((h0.T @ h0) - (ph_v1.T @ ph_v1)) / v0.shape[0]
                    np.fill_diagonal(dWvv, 0.0)
                    np.fill_diagonal(dWhh, 0.0)
                else:
                    dWvv = np.zeros_like(self.Wvv)
                    dWhh = np.zeros_like(self.Whh)

                # Regularización
                if self.l2 > 0.0:
                    dWvh -= self.l2 * self.Wvh
                    if self.use_intra_connect:
                        dWvv -= self.l2 * self.Wvv
                        dWhh -= self.l2 * self.Whh

                # Clipping
                if self.clip_grad is not None:
                    np.clip(dWvh, -self.clip_grad, self.clip_grad, out=dWvh)
                    np.clip(dbv,  -self.clip_grad, self.clip_grad, out=dbv)
                    np.clip(dbh,  -self.clip_grad, self.clip_grad, out=dbh)
                    if self.use_intra_connect:
                        np.clip(dWvv, -self.clip_grad, self.clip_grad, out=dWvv)
                        np.clip(dWhh, -self.clip_grad, self.clip_grad, out=dWhh)

                # Update
                self.Wvh += lr * dWvh.astype(np.float32)
                self.bv  += lr * dbv.astype(np.float32)
                self.bh  += lr * dbh.astype(np.float32)
                if self.use_intra_connect:
                    self.Wvv += lr * dWvv.astype(np.float32)
                    self.Whh += lr * dWhh.astype(np.float32)

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                H = self._p_h_given(X[:128])
                V = self._p_v_given(H)
                mse = float(np.mean((X[:128] - V)**2))
                print(f"[BM]  epoch={ep:03d} mse_recon={mse:.6f}")
        return self

    def transform_hidden(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)
        return self._p_h_given(X).astype(np.float32)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)
        H = self._p_h_given(X)
        V = self._p_v_given(H)
        return V.astype(np.float32)
