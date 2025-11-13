# backend/src/neurocampus/models/rbm_manual.py
from __future__ import annotations
from typing import Optional

import numpy as np

from .utils_boltzmann import (
    EPS,
    sigmoid,
    bernoulli_sample,
    check_numeric_matrix,
    batch_iter,
    binarize,
    gibbs_step_vh,
)


class RestrictedBoltzmannMachine:
    """
    RBM binaria (visibles y ocultas en {0,1}) con soporte para CD-k y PCD.

    Interfaz:
      - fit(X, epochs, batch_size, verbose)
      - transform_hidden(X) -> probs h|v
      - reconstruct(X) -> v' (probs)

    Parámetros principales:
      n_visible           : nº de unidades visibles
      n_hidden            : nº de unidades ocultas
      learning_rate       : tasa de aprendizaje
      l2                  : regularización L2 sobre W
      clip_grad           : clipping de gradiente (None para desactivar)
      binarize_input      : si True, binariza X con input_bin_threshold
      input_bin_threshold : umbral de binarización
      cd_k                : pasos de Gibbs para CD-k (k >= 1)
      use_pcd             : si True, usa Persistent Contrastive Divergence
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
        cd_k: int = 1,
        use_pcd: bool = False,
    ):
        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.clip_grad = None if clip_grad is None else float(clip_grad)
        self.binarize_input = bool(binarize_input)
        self.input_bin_threshold = float(input_bin_threshold)

        self.cd_k = max(1, int(cd_k))
        self.use_pcd = bool(use_pcd)

        self.rng = np.random.default_rng(seed)
        # Xavier-like init
        scale = 1.0 / np.sqrt(self.n_visible + self.n_hidden)
        self.W = self.rng.normal(
            0.0, scale, size=(self.n_visible, self.n_hidden)
        ).astype(np.float32)
        self.bv = np.zeros(self.n_visible, dtype=np.float32)
        self.bh = np.zeros(self.n_hidden, dtype=np.float32)

        # Cadena persistente para PCD (batch_size, n_visible)
        self._pcd_chain: Optional[np.ndarray] = None

    # ---- Condicionales ----
    def _p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.bh)

    def _p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.bv)

    def _sample_h(self, v: np.ndarray) -> np.ndarray:
        return bernoulli_sample(self._p_h_given_v(v), self.rng)

    def _sample_v(self, h: np.ndarray) -> np.ndarray:
        return bernoulli_sample(self._p_v_given_h(h), self.rng)

    # ---- Gibbs helpers (CD-k / PCD) ----
    def _negative_phase_cd_k(self, v0: np.ndarray):
        """
        Ejecuta CD-k estándar empezando en v0.

        Devuelve:
          v_k, p(v_k), h_k, p(h_k)
        """
        v = v0
        v_prob = None
        h_sample = None
        h_prob = None
        for _ in range(self.cd_k):
            v, v_prob, h_sample, h_prob = gibbs_step_vh(
                v, self.W, self.bv, self.bh, self.rng
            )
        # v, v_prob, h_sample, h_prob vienen del último paso
        return v, v_prob, h_sample, h_prob

    def _negative_phase_pcd(self, v0: np.ndarray):
        """
        Ejecuta PCD-k usando una cadena persistente.
        Si la cadena aún no existe o el tamaño de batch cambia, se reinicia con v0.
        """
        if self._pcd_chain is None or self._pcd_chain.shape != v0.shape:
            self._pcd_chain = v0.copy()

        v = self._pcd_chain
        v_prob = None
        h_sample = None
        h_prob = None
        for _ in range(self.cd_k):
            v, v_prob, h_sample, h_prob = gibbs_step_vh(
                v, self.W, self.bv, self.bh, self.rng
            )

        # Actualizar la cadena persistente con el último v
        self._pcd_chain = v
        return v, v_prob, h_sample, h_prob

    # ---- API pública ----
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: int = 1,
    ) -> "RestrictedBoltzmannMachine":
        X = np.asarray(X, dtype=np.float32)
        check_numeric_matrix(X, "X")
        if self.binarize_input:
            X = binarize(X, self.input_bin_threshold)

        lr = self.learning_rate

        for ep in range(1, epochs + 1):
            for v0 in batch_iter(X, batch_size, self.rng):
                # Fase positiva
                ph_v0 = self._p_h_given_v(v0)  # probs h|v0

                # Fase negativa: CD-k o PCD
                if self.use_pcd:
                    v_neg, pv_neg, h_neg, ph_neg = self._negative_phase_pcd(v0)
                else:
                    v_neg, pv_neg, h_neg, ph_neg = self._negative_phase_cd_k(v0)

                # Gradientes (esperanza positiva - negativa)
                # Usamos probs, no muestras, como en la versión original
                dW = (v0.T @ ph_v0 - pv_neg.T @ ph_neg) / v0.shape[0]
                dbv = np.mean(v0 - pv_neg, axis=0)
                dbh = np.mean(ph_v0 - ph_neg, axis=0)

                # Regularización L2
                if self.l2 > 0.0:
                    dW -= self.l2 * self.W

                # Clipping
                if self.clip_grad is not None:
                    np.clip(dW, -self.clip_grad, self.clip_grad, out=dW)
                    np.clip(dbv, -self.clip_grad, self.clip_grad, out=dbv)
                    np.clip(dbh, -self.clip_grad, self.clip_grad, out=dbh)

                # Update
                self.W += lr * dW.astype(np.float32)
                self.bv += lr * dbv.astype(np.float32)
                self.bh += lr * dbh.astype(np.float32)

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                # Reconstrucción simple para logging
                pvh = self._p_v_given_h(self._p_h_given_v(X[:128]))
                mse = float(np.mean((X[:128] - pvh) ** 2))
                print(
                    f"[RBM] epoch={ep:03d} cd_k={self.cd_k} "
                    f"{'PCD' if self.use_pcd else 'CD'} mse_recon={mse:.6f}"
                )

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
