# backend/src/neurocampus/models/strategies/rbm_manual_strategy.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from neurocampus.models import RestrictedBoltzmannMachine

class RBMManualStrategy:
    """
    Wrapper Strategy para usar la RBM "a mano" (NumPy) dentro del pipeline estándar.
    Expone interfaz: fit(X), transform(X), reconstruct(X), get_params()
    """
    def __init__(
        self,
        n_hidden: int = 64,
        learning_rate: float = 0.05,
        seed: int = 42,
        l2: float = 0.0,
        clip_grad: Optional[float] = 1.0,
        binarize_input: bool = False,
        input_bin_threshold: float = 0.5,
        epochs: int = 20,
        batch_size: int = 64,
    ):
        self.cfg: Dict[str, Any] = dict(
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            seed=seed,
            l2=l2,
            clip_grad=clip_grad,
            binarize_input=binarize_input,
            input_bin_threshold=input_bin_threshold,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.model: Optional[RestrictedBoltzmannMachine] = None
        self.n_visible: Optional[int] = None

    def fit(self, X: np.ndarray) -> "RBMManualStrategy":
        X = np.asarray(X, dtype=np.float32)
        self.n_visible = X.shape[1]
        self.model = RestrictedBoltzmannMachine(
            n_visible=self.n_visible,
            n_hidden=self.cfg["n_hidden"],
            learning_rate=self.cfg["learning_rate"],
            seed=self.cfg["seed"],
            l2=self.cfg["l2"],
            clip_grad=self.cfg["clip_grad"],
            binarize_input=self.cfg["binarize_input"],
            input_bin_threshold=self.cfg["input_bin_threshold"],
        )
        self.model.fit(X, epochs=self.cfg["epochs"], batch_size=self.cfg["batch_size"], verbose=1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RBMManualStrategy no entrenada; llama a fit() primero.")
        X = np.asarray(X, dtype=np.float32)
        return self.model.transform_hidden(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RBMManualStrategy no entrenada; llama a fit() primero.")
        X = np.asarray(X, dtype=np.float32)
        return self.model.reconstruct(X)

    def get_params(self) -> Dict[str, Any]:
        base = dict(self.cfg)
        base["n_visible"] = self.n_visible
        return base
