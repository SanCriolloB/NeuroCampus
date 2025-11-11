# backend/src/neurocampus/models/rbm_manual.py
import numpy as np
from typing import Tuple, Optional
from .utils_boltzmann import sigmoid, bernoulli_sample, check_numeric_matrix, batch_iter, binarize

class RestrictedBoltzmannMachine:
    """
    RBM binaria (visible/oculta) implementada a mano con NumPy.
    - Sin frameworks externos
    - Contrastive Divergence (CD-1)
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
        # Inicialización pequeña y simétrica
        scale = 0.01
        self.W  = self.rng.normal(0.0, scale, size=(self.n_visible, self.n_hidden)).astype(np.float32)
        self.bv = np.zeros(self.n_visible, dtype=np.float32)
        self.bh = np.zeros(self.n_hidden,  dtype=np.float32)

    # --------- muestreos básicos ---------
    def sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_h = sigmoid(v @ self.W + self.bh)
        h   = bernoulli_sample(p_h, self.rng)
        return p_h, h

    def sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_v = sigmoid(h @ self.W.T + self.bv)
        v   = bernoulli_sample(p_v, self.rng)
        return p_v, v

    # --------- paso de entrenamiento (CD-1) ---------
    def _cd1(self, v0: np.ndarray) -> None:
        # Entrada segura/binarizada opcional
        v0_use = binarize(v0) if self.binarize_input else v0

        ph0, h0 = self.sample_hidden(v0_use)
        pv1, v1 = self.sample_visible(h0)
        ph1, h1 = self.sample_hidden(v1)

        # Gradientes (promedio por batch)
        dW = v0_use.T @ ph0 - v1.T @ ph1
        dbv = np.mean(v0_use - v1, axis=0)
        dbh = np.mean(ph0 - ph1, axis=0)

        # L2 opcional
        if self.l2 > 0.0:
            dW -= self.l2 * self.W

        # Clipping opcional
        if self.clip_grad is not None:
            c = self.clip_grad
            dW  = np.clip(dW,  -c, c)
            dbv = np.clip(dbv, -c, c)
            dbh = np.clip(dbh, -c, c)

        # Actualización
        bs = float(v0.shape[0])
        self.W  += self.lr * dW  / bs
        self.bv += self.lr * dbv
        self.bh += self.lr * dbh

    # --------- API pública ---------
    def fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 64, verbose: int = 1) -> "RestrictedBoltzmannMachine":
        check_numeric_matrix(X, "X")
        for e in range(1, epochs + 1):
            for batch in batch_iter(X, batch_size, self.rng):
                self._cd1(batch.astype(np.float32))
            if verbose and (e == 1 or e % max(1, epochs // 5) == 0 or e == epochs):
                meanW = float(np.mean(np.abs(self.W)))
                print(f"[RBM] epoch {e:03d}/{epochs}  |mean|W|={meanW:.6f}")
        return self

    def transform_hidden(self, X: np.ndarray) -> np.ndarray:
        """Devuelve p(h=1|v) para features ocultos (útil como embedding)."""
        check_numeric_matrix(X, "X")
        return sigmoid(X @ self.W + self.bh)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstrucción esperada p(v=1|h) con h = p(h|v)."""
        check_numeric_matrix(X, "X")
        p_h = sigmoid(X @ self.W + self.bh)
        p_v = sigmoid(p_h @ self.W.T + self.bv)
        return p_v
