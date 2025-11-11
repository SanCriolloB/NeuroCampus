# backend/src/neurocampus/models/bm_manual.py
import numpy as np
from typing import Optional
from .utils_boltzmann import sigmoid, bernoulli_sample, check_numeric_matrix, binarize

class BoltzmannMachine:
    """
    Máquina de Boltzmann (totalmente conectada visible/oculta) simplificada.
    Nota: Para entrenamiento práctico suele preferirse RBM + stacking.
    Aquí se deja una versión de aprendizaje contrastivo básico (no simetriza W).
    """
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.05,
        seed: Optional[int] = 42,
        l2: float = 0.0,
        clip_grad: Optional[float] = None,
        binarize_input: bool = False,
        input_bin_threshold: float = 0.5,
    ):
        self.n_visible = int(n_visible)
        self.n_hidden  = int(n_hidden)
        self.lr        = float(learning_rate)
        self.l2        = float(l2)
        self.clip_grad = clip_grad
        self.binarize_input = bool(binarize_input)
        self.input_bin_threshold = float(input_bin_threshold)

        self.rng = np.random.default_rng(seed)
        scale = 0.01
        self.W  = self.rng.normal(0.0, scale, size=(self.n_visible, self.n_hidden)).astype(np.float32)
        self.bv = np.zeros(self.n_visible, dtype=np.float32)
        self.bh = np.zeros(self.n_hidden,  dtype=np.float32)

    def sample_h(self, v: np.ndarray):
        p_h = sigmoid(v @ self.W + self.bh)
        h   = bernoulli_sample(p_h, self.rng)
        return p_h, h

    def sample_v(self, h: np.ndarray):
        p_v = sigmoid(h @ self.W.T + self.bv)
        v   = bernoulli_sample(p_v, self.rng)
        return p_v, v

    def _cd1(self, v0: np.ndarray):
        v0_use = binarize(v0) if self.binarize_input else v0
        p_h0, h0 = self.sample_h(v0_use)
        p_v1, v1 = self.sample_v(h0)
        p_h1, h1 = self.sample_h(v1)

        dW  = v0_use.T @ p_h0 - v1.T @ p_h1
        dbv = np.mean(v0_use - v1, axis=0)
        dbh = np.mean(p_h0 - p_h1, axis=0)

        if self.l2 > 0.0:
            dW -= self.l2 * self.W

        if self.clip_grad is not None:
            c = self.clip_grad
            dW  = np.clip(dW,  -c, c)
            dbv = np.clip(dbv, -c, c)
            dbh = np.clip(dbh, -c, c)

        bs = float(v0.shape[0])
        self.W  += self.lr * dW / bs
        self.bv += self.lr * dbv
        self.bh += self.lr * dbh

    def fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 64, verbose: int = 1) -> "BoltzmannMachine":
        check_numeric_matrix(X, "X")
        n = X.shape[0]
        for e in range(1, epochs + 1):
            # barajado simple por lotes
            idx = np.arange(n)
            self.rng.shuffle(idx)
            for i in range(0, n, batch_size):
                batch = X[idx[i:i+batch_size]].astype(np.float32)
                self._cd1(batch)
            if verbose and (e == 1 or e % max(1, epochs // 5) == 0 or e == epochs):
                meanW = float(np.mean(np.abs(self.W)))
                print(f"[BM]  epoch {e:03d}/{epochs}  |mean|W|={meanW:.6f}")
        return self

    def transform_hidden(self, X: np.ndarray) -> np.ndarray:
        check_numeric_matrix(X, "X")
        return sigmoid(X @ self.W + self.bh)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        check_numeric_matrix(X, "X")
        p_h = sigmoid(X @ self.W + self.bh)
        p_v = sigmoid(p_h @ self.W.T + self.bv)
        return p_v
