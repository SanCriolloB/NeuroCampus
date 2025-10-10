# backend/src/neurocampus/models/strategies/modelo_rbm_general.py
# Implementación mínima de una RBM binaria con CD-k usando PyTorch.
# Mantiene el contrato con PlantillaEntrenamiento.

from __future__ import annotations
import os
import time
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

Tensor = torch.Tensor

def _resolve_path(ref: Optional[str]) -> Optional[str]:
    if ref is None:
        return None
    if ref.startswith("localfs://"):
        return ref.replace("localfs://", "", 1)
    return ref  # asume ruta local

def _load_matrix(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = _resolve_path(path)
    if p is None or not os.path.exists(p):
        return None
    # Soporta csv/xlsx/parquet básico (suficiente para smoke tests Día 4)
    if p.endswith(".csv"):
        df = pd.read_csv(p)
    elif p.endswith(".xlsx"):
        df = pd.read_excel(p)
    elif p.endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)  # fallback
    # Selección naive: columnas numéricas → matriz [0,1] normalizada
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return None
    # Normaliza a [0,1] (si las preguntas van 0..50 esto cuadra con el plan)
    arr = num.to_numpy(dtype=np.float32)
    arr = (arr - np.nanmin(arr, axis=0)) / (np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0) + 1e-9)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr

class RBM:
    def __init__(self, n_visible: int, n_hidden: int, lr: float, momentum: float, weight_decay: float, cd_k: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = torch.randn(n_visible, n_hidden, generator=g) * 0.01
        self.b_v = torch.zeros(n_visible)
        self.b_h = torch.zeros(n_hidden)
        self.W_m = torch.zeros_like(self.W)  # momentum terms
        self.bv_m = torch.zeros_like(self.b_v)
        self.bh_m = torch.zeros_like(self.b_h)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cd_k = cd_k

    def _sigmoid(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def _sample_h(self, v: Tensor) -> Tuple[Tensor, Tensor]:
        p_h = self._sigmoid(v @ self.W + self.b_h)
        h = torch.bernoulli(p_h)
        return p_h, h

    def _sample_v(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        p_v = self._sigmoid(h @ self.W.T + self.b_v)
        v = torch.bernoulli(p_v)
        return p_v, v

    def step(self, v0: Tensor) -> Dict[str, float]:
        # Positive phase
        ph0, h0 = self._sample_h(v0)

        # Gibbs sampling k pasos
        vk = v0.clone()
        hk = h0
        for _ in range(self.cd_k):
            pvk, vk = self._sample_v(hk)
            phk, hk = self._sample_h(vk)

        # Gradientes (CD-k)
        pos_grad = v0.T @ ph0
        neg_grad = vk.T @ phk

        # Recon error (MSE entre v0 y prob. reconstr. pvk)
        recon_error = torch.mean((v0 - pvk) ** 2).item()

        # Actualizaciones con momentum + weight decay
        dW = (pos_grad - neg_grad) / v0.shape[0] - self.weight_decay * self.W
        dbv = torch.mean(v0 - pvk, dim=0)
        dbh = torch.mean(ph0 - phk, dim=0)

        self.W_m = self.momentum * self.W_m + self.lr * dW
        self.bv_m = self.momentum * self.bv_m + self.lr * dbv
        self.bh_m = self.momentum * self.bh_m + self.lr * dbh

        self.W += self.W_m
        self.b_v += self.bv_m
        self.b_h += self.bh_m

        # Grad norm (para monitoreo)
        grad_norm = torch.linalg.vector_norm(dW).item()

        return {"loss": recon_error, "recon_error": recon_error, "grad_norm": grad_norm}

class RBMGeneral:
    """Estrategia RBM 'general' — binaria, CD-k, contrato con PlantillaEntrenamiento."""
    def __init__(self):
        self.rbm: Optional[RBM] = None
        self.data: Optional[Tensor] = None
        self.batch_size: int = 64

    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        seed = int(hparams.get("seed", 42) or 42)
        np.random.seed(seed); torch.manual_seed(seed)
        arr = _load_matrix(data_ref)
        if arr is None:
            # Dataset mínimo aleatorio (solo para no fallar si no hay data)
            arr = np.clip(np.random.rand(256, int(hparams.get("n_visible") or 16)).astype(np.float32), 0, 1)
        n_visible = int(hparams.get("n_visible") or arr.shape[1])
        n_hidden = int(hparams.get("n_hidden", 32))
        lr = float(hparams.get("lr", 0.01))
        self.batch_size = int(hparams.get("batch_size", 64))
        cd_k = int(hparams.get("cd_k", 1))
        momentum = float(hparams.get("momentum", 0.5))
        weight_decay = float(hparams.get("weight_decay", 0.0))
        self.data = torch.from_numpy(arr)
        self.rbm = RBM(n_visible, n_hidden, lr, momentum, weight_decay, cd_k, seed)

    def _iter_minibatches(self) -> Tensor:
        assert self.data is not None
        idx = torch.randperm(self.data.shape[0])
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self.data[batch_idx]

    def train_step(self, epoch: int) -> Tuple[float, Dict]:
        """Ejecuta 1 época (sobre todos los minibatches). Devuelve (loss_promedio, metrics)."""
        assert self.rbm is not None
        t0 = time.perf_counter()
        losses = []
        grad_norms = []
        for v0 in self._iter_minibatches():
            metrics = self.rbm.step(v0)
            losses.append(metrics["loss"])
            grad_norms.append(metrics["grad_norm"])
        dt = (time.perf_counter() - t0) * 1000.0
        loss_mean = float(np.mean(losses))
        grad_mean = float(np.mean(grad_norms))
        return loss_mean, {
            "recon_error": loss_mean,
            "grad_norm": grad_mean,
            "time_epoch_ms": dt
        }

# En la versión 'restringida' usamos la misma implementación base
class RBMRestringida(RBMGeneral):
    pass
