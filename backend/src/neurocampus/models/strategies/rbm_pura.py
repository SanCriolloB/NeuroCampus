# backend/src/neurocampus/models/strategies/rbm_pura.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pandas as pd

__all__ = ["RBM", "ModeloRBMPura"]

# --------------------------
# Utilidades de normalización
# --------------------------
@dataclass
class _MinMax:
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        # Asegurar tipo y tratar inf/-inf
        X = np.asarray(X, dtype=np.float32, order="C")
        X_clean = np.where(np.isfinite(X), X, np.nan)

        # Columnas completamente NaN
        all_nan = np.isnan(X_clean).all(axis=0)
        X_stats = X_clean.copy()
        if all_nan.any():
            # columnas sin datos → las ponemos a 0 temporalmente
            X_stats[:, all_nan] = 0.0

        self.min_ = np.nanmin(X_stats, axis=0).astype(np.float32)
        self.max_ = np.nanmax(X_stats, axis=0).astype(np.float32)

        # Columnas sin info real → rango [0,1]
        if all_nan.any():
            self.min_[all_nan] = 0.0
            self.max_[all_nan] = 1.0

        # evitar división por cero
        self.max_ = np.where((self.max_ - self.min_) < 1e-9,
                             self.min_ + 1.0,
                             self.max_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler no entrenado.")

        X = np.asarray(X, dtype=np.float32, order="C")
        # Tratar NaN/inf usando la min_ como fallback
        X = np.where(np.isfinite(X), X, self.min_[None, :])

        X = (X - self.min_) / (self.max_ - self.min_)
        X = np.clip(X, 0.0, 1.0)
        # Airbag extra
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        return X

# --------------------------
# Núcleo RBM bipartita (W, b_v, b_h)
# --------------------------
class _RBMCore(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int, cd_k: int = 1, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.W   = nn.Parameter(torch.randn(n_visible, n_hidden, generator=g) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.cd_k = int(cd_k)

    @staticmethod
    def _sigmoid(x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def sample_h(self, v: Tensor) -> Tuple[Tensor, Tensor]:
        p_h = self._sigmoid(v @ self.W + self.b_h)
        # Airbag: limpiar NaN/inf y forzar [0,1]
        p_h = torch.nan_to_num(p_h, nan=0.5, posinf=1.0, neginf=0.0)
        p_h = torch.clamp(p_h, 0.0, 1.0)
        h = torch.bernoulli(p_h)
        return p_h, h

    def sample_v(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        p_v = self._sigmoid(h @ self.W.t() + self.b_v)
        p_v = torch.nan_to_num(p_v, nan=0.5, posinf=1.0, neginf=0.0)
        p_v = torch.clamp(p_v, 0.0, 1.0)
        v = torch.bernoulli(p_v)
        return p_v, v


    @torch.no_grad()
    def cd_step(self, v0: Tensor) -> Dict[str, float]:
        ph0, h0 = self.sample_h(v0)
        vk, hk = v0, h0
        pvk, phk = None, None
        for _ in range(self.cd_k):
            pvk, vk = self.sample_v(hk)
            phk, hk = self.sample_h(vk)

        if pvk is None:
            pvk, _ = self.sample_v(h0)
        if phk is None:
            phk, _ = self.sample_h(vk)

        pos = v0.t() @ ph0
        neg = vk.t() @ phk
        dW  = (pos - neg) / v0.shape[0]
        dbv = torch.mean(v0 - pvk, dim=0)
        dbh = torch.mean(ph0 - phk, dim=0)

        # En lugar de .backward(), seteamos grad manual y el optimizador actualiza:
        self.W.grad   = -dW
        self.b_v.grad = -dbv
        self.b_h.grad = -dbh

        recon = torch.mean((v0 - pvk) ** 2).item()
        return {"recon_error": recon, "grad_norm": torch.linalg.vector_norm(dW).item()}

    def hidden_probs(self, v: Tensor) -> Tensor:
        return self._sigmoid(v @ self.W + self.b_h)

# --------------------------------------
# Wrapper de alto nivel con API consistente
# --------------------------------------
class RBM:
    """
    RBM Bernoulli–Bernoulli mínima, bipartita, entrenada con CD-k.
    API compatible con el auditor: fit, predict_proba, predict.
    """
    def __init__(
        self,
        n_hidden: int = 64,
        cd_k: int = 1,
        lr: float = 1e-2,
        batch_size: int = 64,
        epochs: int = 5,
        seed: int = 42,
        device: Optional[str] = None,
        **_
    ):
        self.n_hidden = int(n_hidden)
        self.cd_k = int(cd_k)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.core: Optional[_RBMCore] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.scaler = _MinMax()

    # ---------- Fit ----------
    def _iter_batches(self, X: Tensor):
        idx = torch.randperm(X.shape[0], device=X.device)
        for s in range(0, len(idx), self.batch_size):
            sel = idx[s:s+self.batch_size]
            yield X[sel]

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        # Solo no-supervisado: y se ignora (mantener firma compatible)
        X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        Xs = self.scaler.fit(X_np).transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)

        n_visible = Xt.shape[1]
        torch.manual_seed(self.seed)
        self.core = _RBMCore(n_visible, self.n_hidden, self.cd_k, self.seed).to(self.device)
        self.opt = torch.optim.SGD(self.core.parameters(), lr=self.lr, momentum=0.5)

        for _ in range(self.epochs):
            for v0 in self._iter_batches(Xt):
                self.opt.zero_grad(set_to_none=True)
                _ = self.core.cd_step(v0)
                self.opt.step()
        return self

    # ---------- Transform / Infer ----------
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)
        with torch.no_grad():
            H = self.core.hidden_probs(Xt)
        return H.detach().cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Para comparación con el auditor multiclase, devolvemos una “pseudo-proba”
        de 3 clases construida a partir de proyecciones de H.
        En producción, esto normalmente se conecta a una cabeza supervisada.
        """
        H = self.transform(X)  # [N, n_hidden]
        # Heurística: 3 “clusters” blandos por softmax de proyección lineal fija:
        Wlog = np.stack([
            np.linspace(-1, 1, self.n_hidden),
            np.linspace(0,  1, self.n_hidden),
            np.linspace(1,  0, self.n_hidden),
        ], axis=1).astype(np.float32)  # [n_hidden, 3]
        logits = H @ Wlog
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exps / exps.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        idx = self.predict_proba(X).argmax(axis=1)
        inv = {0: "neg", 1: "neu", 2: "pos"}
        return [inv[i] for i in idx]

# Alias con nombre estilo proyecto
ModeloRBMPura = RBM
