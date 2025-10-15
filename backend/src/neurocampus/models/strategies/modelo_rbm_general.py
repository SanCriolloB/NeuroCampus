# backend/src/neurocampus/models/strategies/modelo_rbm_general.py
# RBM Student "general" con cabeza supervisada para {neg, neu, pos}.
# - Preprocesa calif_1..calif_10: imputación + escalado a [0,1] (minmax o [0,5]).
# - (opcional) añade p_neg/p_neu/p_pos como features (use_text_probs=True).
# - (opcional) añade embeddings de texto x_text_* (use_text_embeds=True, prefijo configurable).
# - Entrena RBM (CD-k) y cabeza softmax (PyTorch) con pesos por clase.
# - Expone predict_proba/predict y versiones para DataFrame (predict_proba_df/predict_df).
# - Guarda/recupera: vectorizer.json, rbm.pt, head.pt, meta.json (feat_cols_, hparams clave).
#
# Uso típico:
#   strat = RBMGeneral()
#   strat.setup(data_ref="data/labeled/evaluaciones_2025_beto.parquet", hparams={
#       "scale_mode": "minmax",
#       "n_hidden": 64,
#       "cd_k": 1,
#       "epochs_rbm": 1,
#       "use_text_probs": True,
#       "use_text_embeds": True,
#       "text_embed_prefix": "x_text_",
#       "max_calif": 10,
#   })
#   for epoch in range(1, N+1):
#       loss, metrics = strat.train_step(epoch)
#   proba = strat.predict_proba_df(df_val)  # inferencia con DataFrame (recomendado)
#   yhat  = strat.predict_df(df_val)

from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ["RBMGeneral", "ModeloRBMGeneral"]

# -------------------------
# Constantes / utilidades
# -------------------------

_META_EXCLUDE = {
    "id", "codigo", "codigo_materia", "codigo materia", "materia", "asignatura",
    "grupo", "periodo", "semestre", "docente", "profesor", "fecha"
}

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}
_CLASSES = ["neg", "neu", "pos"]


def _safe_lower(s) -> str:
    try:
        return str(s).lower()
    except Exception:
        return ""


def _suffix_index(name: str, prefix: str) -> int:
    """Devuelve el índice entero después de prefix, p.ej. x_text_12 -> 12; si falla, 1e9."""
    try:
        return int(name.replace(prefix, "", 1))
    except Exception:
        return 10**9


def _pick_feature_cols(
    df: pd.DataFrame,
    *,
    max_calif: int = 10,
    include_text_probs: bool = False,
    include_text_embeds: bool = False,
    text_embed_prefix: str = "x_text_"
) -> List[str]:
    """Devuelve el orden de columnas de entrada para el Student:
       - calif_1..calif_{max_calif} si existen.
       - Si se pide, añade p_neg/p_neu/p_pos (si existen).
       - Si se pide, añade columnas con prefijo text_embed_prefix (x_text_*) ordenadas por sufijo numérico.
       - Evita metadatos y mantiene un orden estable."""
    cols = list(df.columns)

    # 1) calificaciones explícitas
    califs = [c for c in cols if c.startswith("calif_")]
    if califs:
        def _idx(c: str):
            try:
                return int(c.split("_")[1])
            except Exception:
                return 10**9
        califs = sorted(califs, key=_idx)[:max_calif]
    else:
        # Fallback poco usado: numéricas excepto metadatos
        num = df.select_dtypes(include=["number"]).columns.tolist()
        califs = [c for c in num if _safe_lower(c) not in _META_EXCLUDE][:max_calif]

    features: List[str] = list(califs)

    # 2) probs de texto (3 dims)
    if include_text_probs and all(k in df.columns for k in ["p_neg", "p_neu", "p_pos"]):
        features += ["p_neg", "p_neu", "p_pos"]

    # 3) embeddings de texto (k dims)
    if include_text_embeds:
        embed_cols = [c for c in cols if c.startswith(text_embed_prefix)]
        if embed_cols:
            embed_cols = sorted(embed_cols, key=lambda c: _suffix_index(c, text_embed_prefix))
            features += embed_cols

    return features


@dataclass
class _Vectorizer:
    """Imputación y escalado a [0,1]. Guarda stats para inferencia."""
    mean_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    mode: str = "minmax"  # "minmax" o "scale_0_5"

    def fit(self, X: np.ndarray, mode: str = "minmax") -> "_Vectorizer":
        self.mode = mode
        self.mean_ = np.nanmean(X, axis=0)
        if self.mode == "scale_0_5":
            self.min_ = np.zeros(X.shape[1], dtype=np.float32)
            self.max_ = np.ones(X.shape[1], dtype=np.float32) * 5.0
        else:
            self.min_ = np.nanmin(X, axis=0)
            self.max_ = np.nanmax(X, axis=0)
        # bordes seguros si columnas constantes
        self.max_ = np.where((self.max_ - self.min_) < 1e-9, self.min_ + 1.0, self.max_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Protección: verificar misma dimensionalidad que la usada en fit
        if self.mean_ is not None and X.shape[1] != len(self.mean_):
            raise ValueError(
                f"Vectorizer.transform: dimensión de entrada {X.shape[1]} != {len(self.mean_)} usada en fit."
            )
        X = X.astype(np.float32, copy=False)
        X = np.where(np.isnan(X), self.mean_[None, :], X)
        if self.mode == "scale_0_5":
            X = X / 5.0
        else:
            X = (X - self.min_[None, :]) / (self.max_[None, :] - self.min_[None, :])
        return np.clip(X, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray, mode: str = "minmax") -> np.ndarray:
        return self.fit(X, mode=mode).transform(X)

    def to_dict(self) -> Dict:
        return {
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
            "min_":  None if self.min_  is None else self.min_.tolist(),
            "max_":  None if self.max_  is None else self.max_.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "_Vectorizer":
        v = cls()
        v.mean_ = None if d["mean_"] is None else np.array(d["mean_"], dtype=np.float32)
        v.min_  = None if d["min_"]  is None else np.array(d["min_"],  dtype=np.float32)
        v.max_  = None if d["max_"]  is None else np.array(d["max_"],  dtype=np.float32)
        v.mode  = d.get("mode", "minmax")
        return v


# -------------
# Núcleo de RBM
# -------------

class _RBM(nn.Module):
    """RBM Bernoulli-Bernoulli con CD-k para features en [0,1]."""
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
        h = torch.bernoulli(p_h)
        return p_h, h

    def sample_v(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        p_v = self._sigmoid(h @ self.W.t() + self.b_v)
        v = torch.bernoulli(p_v)
        return p_v, v

    def cd_step(self, v0: Tensor) -> Dict[str, float]:
        ph0, h0 = self.sample_h(v0)
        vk, hk, pvk, phk = v0, h0, None, None
        for _ in range(self.cd_k):
            pvk, vk = self.sample_v(hk)
            phk, hk = self.sample_h(vk)

        pos = v0.t() @ ph0
        neg = vk.t() @ phk
        dW  = (pos - neg) / v0.shape[0]
        dbv = torch.mean(v0 - pvk, dim=0)
        dbh = torch.mean(ph0 - phk, dim=0)

        recon = torch.mean((v0 - pvk) ** 2).item()

        self.W.grad   = -dW
        self.b_v.grad = -dbv
        self.b_h.grad = -dbh

        grad_norm = torch.linalg.vector_norm(dW.detach()).item()
        return {"recon_error": recon, "grad_norm": grad_norm}

    def hidden_probs(self, v: Tensor) -> Tensor:
        return self._sigmoid(v @ self.W + self.b_h)


# ----------------------------
# Student: RBM + cabeza softmax
# ----------------------------

class RBMGeneral:
    """Estrategia RBM 'general' para {neg, neu, pos} con selección de features flexible."""
    def __init__(self,
    n_visible=None,
    n_hidden=None,
    cd_k=None,
    lr_rbm=None,
    lr_head=None,
    momentum=None,
    weight_decay=None,
    seed=None,
    device=None,
    **extra,):
        # Objetos que se inicializan en setup()
        
        # Construye hparams con solo los valores no-nulos
        hp = {}
        for k, v in dict(
            n_visible=n_visible, n_hidden=n_hidden, cd_k=cd_k,
            lr_rbm=lr_rbm, lr_head=lr_head, momentum=momentum,
            weight_decay=weight_decay, seed=seed, device=device,
        ).items():
            if v is not None:
                hp[k] = v
        hp.update(extra or {})

        # Llama a setup (si tu setup ya pone defaults, perfecto)
        self.setup(data_ref=None, hparams=hp)
        self.device: str = "cpu"
        self.vec: _Vectorizer = _Vectorizer()
        self.rbm: Optional[_RBM] = None
        self.head: Optional[nn.Module] = None

        # Datos en torch
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None

        # Hparams
        self.batch_size: int = 64
        self.lr_rbm: float = 1e-2
        self.lr_head: float = 1e-2
        self.momentum: float = 0.5
        self.weight_decay: float = 0.0
        self.cd_k: int = 1
        self.epochs_rbm: int = 1
        self.seed: int = 42
        self.scale_mode: str = "minmax"  # o "scale_0_5"

        # Features seleccionadas en setup()
        self.feat_cols_: List[str] = []
        self.text_embed_prefix_: str = "x_text_"

        # Optimizadores
        self.opt_rbm: Optional[torch.optim.Optimizer] = None
        self.opt_head: Optional[torch.optim.Optimizer] = None

        # Estado
        self._epoch: int = 0
        self.classes_: List[str] = _CLASSES

    # ---------- compatibilidad hacia atrás ----------
    @property
    def feature_columns(self) -> List[str]:
        """Compat: algunos códigos antiguos leían strat.feature_columns."""
        return list(self.feat_cols_)

    # ---------- helpers de IO ----------

    def _load_df(self, ref: str) -> pd.DataFrame:
        p = str(ref)
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)

    def _resolve_labels(self, df: pd.DataFrame, require_accept: bool = False, threshold: float = 0.80) -> Optional[np.ndarray]:
        if "y_sentimiento" in df.columns:
            y = df["y_sentimiento"].astype(str).str.lower()
        elif "y" in df.columns:
            y = df["y"].astype(str).str.lower()
        elif "sentiment_label_teacher" in df.columns:
            y = df["sentiment_label_teacher"].astype(str).str.lower()
            if require_accept:
                if "accepted_by_teacher" in df.columns:
                    mask = df["accepted_by_teacher"].fillna(0).astype(int) == 1
                    y = y.where(mask)
                elif "sentiment_conf" in df.columns:
                    mask = df["sentiment_conf"].fillna(0.0) >= float(threshold)
                    y = y.where(mask)
        else:
            return None

        y = y.map({"neg": "neg", "negative": "neg", "negativo": "neg",
                   "neu": "neu", "neutral": "neu",
                   "pos": "pos", "positive": "pos", "positivo": "pos"})
        return y.to_numpy()

    def _prepare_xy(
        self,
        df: pd.DataFrame,
        *,
        accept_teacher: bool,
        threshold: float,
        max_calif: int,
        include_text_probs: bool,
        include_text_embeds: bool,
        text_embed_prefix: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        feat_cols = _pick_feature_cols(
            df,
            max_calif=max_calif,
            include_text_probs=include_text_probs,
            include_text_embeds=include_text_embeds,
            text_embed_prefix=text_embed_prefix
        )
        X = df[feat_cols].to_numpy(dtype=np.float32)

        y_raw = self._resolve_labels(df, require_accept=accept_teacher, threshold=threshold)
        y = None
        if y_raw is not None:
            y = np.array([_LABEL_MAP[l] if isinstance(l, str) and l in _LABEL_MAP else -1 for l in y_raw],
                         dtype=np.int64)
            mask = y >= 0
            X = X[mask]
            y = y[mask]

        return X, y, feat_cols

    # ---------- API pública ----------

    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        """
        data_ref: ruta a parquet/csv con calif_1..N y (opcional) etiquetas + columnas de texto (p_*, x_text_*).
        hparams:
            - n_hidden (int), cd_k (int), batch_size (int)
            - lr_rbm (float), lr_head (float), momentum (float), weight_decay (float)
            - seed (int), scale_mode {"minmax","scale_0_5"}
            - accept_teacher (bool), accept_threshold (float)
            - max_calif (int, por defecto 10)
            - use_text_probs (bool)
            - use_text_embeds (bool)
            - text_embed_prefix (str, por defecto "x_text_")
        """
        # Hparams base
        self.seed = int(hparams.get("seed", 42) or 42)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() and bool(hparams.get("use_cuda", False)) else "cpu"

        self.batch_size   = int(hparams.get("batch_size", 64))
        self.cd_k         = int(hparams.get("cd_k", 1))
        self.lr_rbm       = float(hparams.get("lr_rbm", 1e-2))
        self.lr_head      = float(hparams.get("lr_head", 1e-2))
        self.momentum     = float(hparams.get("momentum", 0.5))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))
        self.epochs_rbm   = int(hparams.get("epochs_rbm", 1))
        self.scale_mode   = str(hparams.get("scale_mode", "minmax"))

        accept_teacher    = bool(hparams.get("accept_teacher", True))
        accept_threshold  = float(hparams.get("accept_threshold", 0.80))

        max_calif         = int(hparams.get("max_calif", 10))
        include_text_probs  = bool(hparams.get("use_text_probs", False))
        include_text_embeds = bool(hparams.get("use_text_embeds", False))
        self.text_embed_prefix_ = str(hparams.get("text_embed_prefix", "x_text_"))

        # Dataframe
        if data_ref:
            df = self._load_df(data_ref)
        else:
            # Data dummy si no hay ruta (tests)
            df = pd.DataFrame({f"calif_{i+1}": np.random.rand(256).astype(np.float32) * 5.0 for i in range(max_calif)})

        # Preparar datos
        X_np, y_np, feat_cols = self._prepare_xy(
            df,
            accept_teacher=accept_teacher,
            threshold=accept_threshold,
            max_calif=max_calif,
            include_text_probs=include_text_probs,
            include_text_embeds=include_text_embeds,
            text_embed_prefix=self.text_embed_prefix_
        )

        # Guardar orden de columnas seleccionadas (para inferencia y persistencia)
        self.feat_cols_ = list(feat_cols)

        # Vectorizer sobre TODAS las columnas seleccionadas
        self.vec = _Vectorizer().fit(X_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
        X_np = self.vec.transform(X_np)

        # Tensores y modelos
        X_t = torch.from_numpy(X_np).to(self.device)
        self.X = X_t

        n_visible = X_np.shape[1]
        n_hidden  = int(hparams.get("n_hidden", 32))
        self.rbm  = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
        self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum, weight_decay=self.weight_decay)

        self.head = nn.Linear(n_hidden, len(_CLASSES)).to(self.device)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        self.y = torch.from_numpy(y_np).to(self.device) if y_np is not None else None
        self._epoch = 0

    # ---------- Mini-batches ----------

    def _iter_minibatches(self, X: Tensor, y: Optional[Tensor]):
        idx = torch.randperm(X.shape[0], device=X.device)
        for start in range(0, len(idx), self.batch_size):
            sel = idx[start:start + self.batch_size]
            yield X[sel], (None if y is None else y[sel])

    # ---------- Entrenamiento ----------

    def train_step(self, epoch: int) -> Tuple[float, Dict]:
        """1 época: RBM (reconstrucción) + cabeza supervisada (si hay y)."""
        assert self.rbm is not None and self.opt_rbm is not None
        self._epoch = epoch

        rbm_losses, rbm_grad = [], []
        cls_losses = []

        # 1) RBM
        self.rbm.train()
        for _ in range(max(1, self.epochs_rbm)):
            for xb, _ in self._iter_minibatches(self.X, self.y):
                self.opt_rbm.zero_grad(set_to_none=True)
                m = self.rbm.cd_step(xb)
                rbm_losses.append(m["recon_error"])
                rbm_grad.append(m["grad_norm"])
                # SGD manual (grad ya está en parámetros con signo negativo)
                for p in self.rbm.parameters():
                    if p.grad is not None:
                        p.data -= self.lr_rbm * p.grad

        # 2) Cabeza supervisada (pesos por clase inversos a la frecuencia)
        if self.y is not None:
            counts = torch.bincount(self.y, minlength=3).float()
            weights = (counts.sum() / (counts + 1e-9))
            weights = weights / weights.sum() * 3.0
            weights = weights.to(self.device)

            self.head.train()
            for xb, yb in self._iter_minibatches(self.X, self.y):
                with torch.no_grad():
                    H = self.rbm.hidden_probs(xb)
                self.opt_head.zero_grad(set_to_none=True)
                logits = self.head(H)
                loss = F.cross_entropy(logits, yb, weight=weights)
                loss.backward()
                self.opt_head.step()
                cls_losses.append(float(loss.detach().cpu()))

        metrics = {
            "epoch": float(epoch),
            "recon_error": float(np.mean(rbm_losses)) if rbm_losses else 0.0,
            "rbm_grad_norm": float(np.mean(rbm_grad)) if rbm_grad else 0.0,
            "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "time_epoch_ms": 0.0
        }
        return metrics["recon_error"] + metrics["cls_loss"], metrics

    # ---------- Transformaciones / Inferencia ----------

    def _df_to_X(self, df: pd.DataFrame) -> np.ndarray:
        """Construye X con el mismo orden de columnas de entrenamiento.
           Si falta alguna columna, la rellena con 0.0."""
        assert len(self.feat_cols_) > 0, "El modelo no tiene feat_cols_ configuradas."
        missing = [c for c in self.feat_cols_ if c not in df.columns]
        if missing:
            # columnas faltantes -> 0.0
            for c in missing:
                df[c] = 0.0
        X_np = df[self.feat_cols_].to_numpy(dtype=np.float32)
        return X_np

    def _transform_np(self, X_np: np.ndarray) -> Tensor:
        # Verificación de dim varianza/fit -> evita errores de broadcast
        if self.vec.mean_ is not None and X_np.shape[1] != len(self.vec.mean_):
            raise ValueError(
                f"Entrada con {X_np.shape[1]} columnas no coincide con vectorizer ({len(self.vec.mean_)}). "
                f"Usa predict_proba_df(df) para construir automáticamente las columnas."
            )
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs.astype(np.float32, copy=False)).to(self.device)
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
        return H

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        """Inferencia a partir de DataFrame con columnas crudas (recomendado)."""
        X_np = self._df_to_X(df.copy())
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            proba = F.softmax(self.head(H), dim=1).cpu().numpy()
        return proba

    def predict_df(self, df: pd.DataFrame) -> List[str]:
        idx = self.predict_proba_df(df).argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    def predict_proba(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Compatibilidad hacia atrás: acepta array con la misma dimensionalidad de entrenamiento
           o un DataFrame (preferido)."""
        if isinstance(X_or_df, pd.DataFrame):
            return self.predict_proba_df(X_or_df)
        X_np = np.asarray(X_or_df, dtype=np.float32)
        assert X_np.shape[1] == len(self.feat_cols_), (
            f"Dimensión de entrada {X_np.shape[1]} != {len(self.feat_cols_)} (entrenamiento). "
            "Usa predict_proba_df(df) para construir automáticamente las columnas."
        )
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            return F.softmax(self.head(H), dim=1).cpu().numpy()

    def predict(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        proba = self.predict_proba(X_or_df)
        idx = proba.argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    # ---------- Persistencia ----------

    def save(self, out_dir: str) -> None:
        """Guarda vectorizer.json, rbm.pt, head.pt y meta.json en out_dir."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # vectorizer
        with open(os.path.join(out_dir, "vectorizer.json"), "w", encoding="utf-8") as f:
            json.dump(self.vec.to_dict(), f, ensure_ascii=False, indent=2)
        # rbm
        torch.save(
            {"state_dict": self.rbm.state_dict(),
             "n_visible": self.rbm.W.shape[0],
             "n_hidden":  self.rbm.W.shape[1],
             "cd_k":      self.rbm.cd_k},
            os.path.join(out_dir, "rbm.pt")
        )
        # head
        torch.save(
            {"state_dict": self.head.state_dict(),
             "classes": self.classes_},
            os.path.join(out_dir, "head.pt")
        )
        # meta
        meta = {
            "feat_cols": self.feat_cols_,
            "scale_mode": self.scale_mode,
            "classes": self.classes_,
            "text_embed_prefix": self.text_embed_prefix_,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: str, device: Optional[str] = None) -> "RBMGeneral":
        """Restaura un modelo guardado con save(). Soporta meta.json opcional."""
        obj = cls()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.device = device

        # vectorizer
        with open(os.path.join(in_dir, "vectorizer.json"), "r", encoding="utf-8") as f:
            obj.vec = _Vectorizer.from_dict(json.load(f))

        # rbm
        rbm_ckpt = torch.load(os.path.join(in_dir, "rbm.pt"), map_location=device)
        obj.rbm = _RBM(
            n_visible=rbm_ckpt["n_visible"],
            n_hidden=rbm_ckpt["n_hidden"],
            cd_k=rbm_ckpt.get("cd_k", 1)
        )
        obj.rbm.load_state_dict(rbm_ckpt["state_dict"])
        obj.rbm.to(device)
        obj.opt_rbm = torch.optim.SGD(obj.rbm.parameters(), lr=1e-6)  # dummy para API

        # head
        head_ckpt = torch.load(os.path.join(in_dir, "head.pt"), map_location=device)
        obj.head = nn.Linear(rbm_ckpt["n_hidden"], len(_CLASSES)).to(device)
        obj.head.load_state_dict(head_ckpt["state_dict"])
        obj.opt_head = torch.optim.Adam(obj.head.parameters(), lr=1e-6)  # dummy
        obj.classes_ = head_ckpt.get("classes", _CLASSES)

        # meta (si existe)
        meta_path = os.path.join(in_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Compatibilidad: diferentes claves posibles
            obj.feat_cols_ = list(
                meta.get("feat_cols")
                or meta.get("feature_cols")
                or meta.get("feature_columns")
                or []
            )
            obj.scale_mode = str(meta.get("scale_mode", obj.vec.mode))
            obj.text_embed_prefix_ = str(meta.get("text_embed_prefix", "x_text_"))
        else:
            # compatibilidad: si no existe meta, intenta suponer calif_1..10
            obj.feat_cols_ = [f"calif_{i+1}" for i in range(10)]
            obj.text_embed_prefix_ = "x_text_"

        obj.X = None
        obj.y = None
        obj._epoch = 0
        return obj
    
# (Opcional pero recomendable) al final del archivo general:
class ModeloRBMGeneral(RBMGeneral):
    """Alias legacy para compatibilidad retro."""
    pass

# ---- Compatibilidad con imports antiguos ----
# Ej.: from neurocampus.models.strategies.modelo_rbm_general import ModeloRBMGeneral
ModeloRBMGeneral = RBMGeneral
