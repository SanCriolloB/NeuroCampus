"""
neurocampus.models.strategies.modelo_rbm_restringida
===================================================

Estrategia **RBM Restringida** con cabeza supervisada para clases {neg, neu, pos}.

Esta variante se enfoca en un set de features reducido (por ejemplo ``calif_1..calif_10``)
y opcionalmente incluye embeddings de texto (``x_text_*``) si se habilita.

Métricas reales + split train/val
--------------------------------------------
- Split real train/val por:
  - columna ``split`` (train/val) si existe, o
  - ``split_mode``: ``temporal`` (por periodo) o ``random``
  - ``val_ratio``: proporción de validación.
- Métricas por época para UI:
  - ``accuracy``, ``f1_macro``, ``val_accuracy``, ``val_f1_macro``, ``val_loss``
  - diagnósticos ``recon_error``/``cls_loss``.
- Confusion matrix 3x3 (se devuelve como parte de metrics; no siempre se grafica por época).

Relación Docente–Materia con embeddings
---------------------------------------------------------------
Se incorpora soporte para capturar relaciones Docente–Materia **sin** vectores one-hot gigantes.

Nuevo parámetro:
- ``teacher_materia_mode``: ``embed`` | ``hash`` | ``none``

Modos:
- ``embed``:
  - teacher_key/materia_key -> índices por hashing estable (md5 mod N buckets)
  - se aprenden embeddings, y la head concatena:
    ``H_rbm + emb_teacher + emb_materia (+ interacción emb_teacher * emb_materia)``.
- ``hash`` (legacy):
  - crea columnas one-hot por hashing: ``teacher_h_*`` y ``materia_h_*``,
    que entran como visibles a la RBM.
- ``none``:
  - ignora teacher/materia.

Objetivo (target) y fuga de información
--------------------------------------------------
Por defecto, el flujo actualizado usa ``target_mode='sentiment_probs'``:

- ``p_neg,p_neu,p_pos`` se usan como **target soft**, NO como features.
- El entrenamiento de la cabeza supervisada usa *soft cross-entropy*:
  ``loss = -sum(y_soft * log_softmax(logits))``.

Fallback (controlado):
- Si ``target_mode='sentiment_probs'`` y faltan ``p_*``:
  - intenta usar una columna de label hard (sent_label/sentiment_label/etc.) y cambia
    internamente a ``target_mode='label'``;
  - si tampoco hay label hard, lanza error (evita “entrenar sin supervisión” por accidente).

.. important::
   Si ``target_mode='sentiment_probs'``, se fuerza ``use_text_probs=False`` para evitar
   leakage (no usar p_* como features si también son el target).

Persistencia
------------
La persistencia de runs/champion se gestiona fuera (router + runs_io).
Esta estrategia implementa :meth:`save(out_dir)` y :meth:`load(in_dir)` para guardar:

- ``rbm.pt`` / ``head.pt`` (pesos)
- ``vectorizer.json`` (normalización)
- ``meta.json`` (feat_cols, split/target, y configuración teacher/materia)

Métricas utilitarias
--------------------
Este archivo usa :mod:`neurocampus.models.utils.metrics` para:

- accuracy
- f1_macro
- confusion_matrix
- normalize_probs
- soft_to_hard

Así se evita duplicación entre estrategias (RBMGeneral/RBMRestringida).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neurocampus.models.utils.metrics import (
    accuracy,
    confusion_matrix,
    f1_macro,
    normalize_probs,
    soft_to_hard,
)

__all__ = ["RBMRestringida", "ModeloRBMRestringida"]


# =============================================================================
# Constantes / utilidades
# =============================================================================

_META_EXCLUDE = {
    "id",
    "codigo",
    "codigo_materia",
    "codigo materia",
    "materia",
    "asignatura",
    "grupo",
    "periodo",
    "semestre",
    "docente",
    "profesor",
    "fecha",
}

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}
_CLASSES = ["neg", "neu", "pos"]

_PROB_COLS = ["p_neg", "p_neu", "p_pos"]

_LABEL_COL_CANDIDATES = [
    "sent_label",
    "sentiment_label",
    "sentiment_label_teacher",
    "teacher_label",
    "y_sentimiento",
    "y",
    "label",
]


def _safe_lower(v: Any) -> str:
    """
    Lowercase seguro para valores heterogéneos.

    :param v: Valor cualquiera.
    :return: String lower-case.
    """
    try:
        return str(v).strip().lower()
    except Exception:
        return ""


def _norm_label(v: Any) -> str:
    """
    Normaliza etiquetas a {neg, neu, pos} soportando variantes en español/inglés.

    :param v: Valor original de etiqueta.
    :return: 'neg'|'neu'|'pos'|'' (si no reconoce).
    """
    s = _safe_lower(v)
    if s in ("neg", "negative", "negativo", "negat"):
        return "neg"
    if s in ("neu", "neutral", "neutro"):
        return "neu"
    if s in ("pos", "positive", "positivo"):
        return "pos"
    return ""


def _suffix_index(name: str, prefix: str) -> int:
    """
    Convierte sufijo de ``x_text_12`` -> 12 para ordenar embeddings.

    :param name: Nombre de columna.
    :param prefix: Prefijo embeddings.
    :return: Índice entero; si falla retorna un valor grande.
    """
    try:
        return int(name.replace(prefix, "", 1))
    except Exception:
        return 10**9


def _parse_periodo_to_sortkey(v: Any) -> Tuple[int, int]:
    """
    Convierte '2025-1' -> (2025, 1) para sorting temporal.

    :param v: Periodo (string o similar).
    :return: (year, term) o (0,0) si no parsea.
    """
    s = _safe_lower(v)
    m = re.match(r"^(\d{4})[-_/ ]?([12])$", s)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _stable_hash_index(text: str, dim: int) -> int:
    """
    Hash estable (md5) para índices 0..dim-1.

    :param text: Texto a hashear.
    :param dim: Número de buckets.
    :return: Índice en [0, dim-1].
    """
    if dim <= 0:
        return 0
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


# =============================================================================
# Selección de features (restringida)
# =============================================================================

def _pick_feature_cols(
    df: pd.DataFrame,
    *,
    max_calif: int = 10,
    include_text_probs: bool = False,
    include_text_embeds: bool = False,
    text_embed_prefix: str = "x_text_",
    target_mode: str = "sentiment_probs",
) -> List[str]:
    """
    Define el orden de columnas de entrada (visibles de la RBM).

    Regla base:
    - ``calif_1..calif_max`` si existen; de lo contrario usa numéricas no-meta.

    Opcional:
    - p_* (solo si include_text_probs y target_mode != sentiment_probs)
    - embeddings x_text_* (solo si include_text_embeds)

    .. important::
       Si ``target_mode='sentiment_probs'``, NO se deben incluir p_* como features.

    :param df: DataFrame con columnas.
    :param max_calif: Máximo de columnas calif_* a usar.
    :param include_text_probs: Si True, permite agregar p_* como feature SOLO si no son target.
    :param include_text_embeds: Si True, agrega embeddings x_text_*.
    :param text_embed_prefix: Prefijo de embeddings de texto.
    :param target_mode: 'sentiment_probs' o 'label'.
    :return: Lista ordenada de columnas.
    """
    cols = list(df.columns)

    # calificaciones
    califs = [c for c in cols if c.startswith("calif_")]
    if califs:

        def _idx(c: str) -> int:
            try:
                return int(c.split("_")[1])
            except Exception:
                return 10**9

        califs = sorted(califs, key=_idx)[:max_calif]
    else:
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        califs = [c for c in nums if _safe_lower(c) not in _META_EXCLUDE][:max_calif]

    features: List[str] = list(califs)

    # p_* como features (NO permitido si target_mode=sentiment_probs por leakage)
    if target_mode != "sentiment_probs":
        if include_text_probs and all(k in df.columns for k in _PROB_COLS):
            features += list(_PROB_COLS)

    # embeddings x_text_*
    if include_text_embeds:
        embed_cols = [c for c in cols if c.startswith(text_embed_prefix)]
        embed_cols = sorted(embed_cols, key=lambda c: _suffix_index(c, text_embed_prefix))
        features += embed_cols

    # dedup
    features = list(dict.fromkeys(features))
    return features


# =============================================================================
# Vectorizador robusto
# =============================================================================

@dataclass
class _Vectorizer:
    """
    Normalizador robusto para features numéricas.

    - ``minmax``: escala cada columna a [0,1] usando min/max observados.
    - ``scale_0_5``: asume escala original [0,5] y normaliza a [0,1].
    """
    mean_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    mode: str = "minmax"  # "minmax" | "scale_0_5"

    def fit(self, X: np.ndarray, mode: str = "minmax") -> "_Vectorizer":
        """
        Ajusta estadísticas de normalización.

        :param X: Matriz (n, d).
        :param mode: 'minmax' o 'scale_0_5'.
        :return: self.
        """
        if X is None or X.size == 0:
            raise ValueError("Vectorizer.fit recibió una matriz vacía.")
        self.mode = mode

        X = X.astype(np.float32, copy=False)
        X_clean = np.where(np.isfinite(X), X, np.nan)

        # columnas totalmente NaN
        all_nan = np.isnan(X_clean).all(axis=0)

        X_stats = X_clean.copy()
        if all_nan.any():
            X_stats[:, all_nan] = 0.0

        self.mean_ = np.nanmean(X_stats, axis=0).astype(np.float32)

        if self.mode == "scale_0_5":
            self.min_ = np.zeros(X_stats.shape[1], dtype=np.float32)
            self.max_ = np.ones(X_stats.shape[1], dtype=np.float32) * 5.0
        else:
            self.min_ = np.nanmin(X_stats, axis=0).astype(np.float32)
            self.max_ = np.nanmax(X_stats, axis=0).astype(np.float32)

        # sanear columnas inválidas
        bad = (~np.isfinite(self.mean_) | ~np.isfinite(self.min_) | ~np.isfinite(self.max_))
        if np.any(bad):
            self.mean_[bad] = 0.0
            self.min_[bad] = 0.0
            self.max_[bad] = 1.0

        # evitar denom ~ 0
        denom = self.max_ - self.min_
        tiny = denom < 1e-9
        if np.any(tiny):
            self.max_[tiny] = self.min_[tiny] + 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica normalización a [0,1].

        :param X: Matriz (n, d).
        :return: Matriz normalizada float32.
        """
        if self.mean_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("Vectorizer no entrenado (fit no ejecutado).")

        X = X.astype(np.float32, copy=False)
        X = np.where(np.isfinite(X), X, np.nan)
        Xc = np.where(np.isnan(X), self.mean_[None, :], X)

        if self.mode == "scale_0_5":
            Xs = Xc / 5.0
        else:
            Xs = (Xc - self.min_[None, :]) / (self.max_[None, :] - self.min_[None, :])

        Xs = np.clip(Xs, 0.0, 1.0)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)
        return Xs.astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el vectorizer a dict."""
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "min": None if self.min_ is None else self.min_.tolist(),
            "max": None if self.max_ is None else self.max_.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "_Vectorizer":
        """
        Reconstruye el vectorizer desde dict.

        :param d: Dict serializado.
        :return: Instancia.
        """
        obj = cls()
        if not d:
            return obj
        obj.mode = d.get("mode", "minmax")
        obj.mean_ = np.array(d["mean"], dtype=np.float32) if d.get("mean") is not None else None
        obj.min_ = np.array(d["min"], dtype=np.float32) if d.get("min") is not None else None
        obj.max_ = np.array(d["max"], dtype=np.float32) if d.get("max") is not None else None
        return obj


# =============================================================================
# RBM Bernoulli-Bernoulli
# =============================================================================

class _RBM(nn.Module):
    """
    RBM Bernoulli-Bernoulli con CD-k para entradas en [0,1].

    La RBM aprende una representación latente ``H`` a partir de visibles ``V``.
    """

    def __init__(self, n_visible: int, n_hidden: int, cd_k: int = 1, seed: int = 42):
        """
        :param n_visible: Dimensión visible.
        :param n_hidden: Dimensión oculta.
        :param cd_k: Pasos CD-k.
        :param seed: Semilla para inicialización determinística.
        """
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden, generator=g) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.cd_k = int(cd_k)

    def hidden_probs(self, v: Tensor) -> Tensor:
        """Probabilidades de hidden: sigmoid(vW + b_h)."""
        return torch.sigmoid(v @ self.W + self.b_h)

    def sample_h(self, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Muestrea hidden.

        :return: (p_h, h_sample)
        """
        p_h = self.hidden_probs(v)
        h = torch.bernoulli(p_h)
        return p_h, h

    def sample_v(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Muestrea visibles.

        :return: (p_v, v_sample)
        """
        p_v = torch.sigmoid(h @ self.W.t() + self.b_v)
        v = torch.bernoulli(p_v)
        return p_v, v

    def cd_step(self, v0: Tensor) -> Dict[str, float]:
        """
        Ejecuta un paso CD-k y deja gradientes en parámetros.

        Retorna diagnósticos:
        - recon_error: MSE entre v0 y pvk
        - grad_norm: norma del gradiente aproximado en W

        :param v0: batch de visibles en [0,1].
        :return: Dict diagnósticos.
        """
        ph0, _ = self.sample_h(v0)
        vk = v0
        hk = torch.bernoulli(ph0)
        pvk = None
        phk = None

        for _ in range(max(1, self.cd_k)):
            pvk, vk = self.sample_v(hk)
            phk, hk = self.sample_h(vk)

        assert pvk is not None and phk is not None

        pos = v0.t() @ ph0
        neg = vk.t() @ phk
        dW = (pos - neg) / v0.shape[0]
        dbv = torch.mean(v0 - pvk, dim=0)
        dbh = torch.mean(ph0 - phk, dim=0)

        # set grads (negativo por convención CD)
        self.W.grad = -dW
        self.b_v.grad = -dbv
        self.b_h.grad = -dbh

        recon = torch.mean((v0 - pvk) ** 2).item()
        grad_norm = torch.linalg.vector_norm(dW.detach()).item()
        return {"recon_error": float(recon), "grad_norm": float(grad_norm)}


# =============================================================================
# Head supervisada con embeddings teacher/materia (Commit 5)
# =============================================================================

class _TeacherMateriaHead(nn.Module):
    """
    Cabeza supervisada con embeddings aprendibles para teacher/materia (Commit 5).

    Combina:
    - representación RBM (H: n_hidden)
    - embedding docente (E_t: emb_dim)
    - embedding materia (E_m: emb_dim)
    - (opcional) interacción E_t * E_m

    Luego aplica una capa lineal a 3 clases (neg/neu/pos).

    :param n_hidden: Dimensión de H.
    :param emb_dim: Dimensión embeddings.
    :param teacher_buckets: Cantidad de buckets para teacher hashing -> Embedding.
    :param materia_buckets: Cantidad de buckets para materia hashing -> Embedding.
    :param use_interaction: Si True concatena E_t * E_m.
    """

    def __init__(
        self,
        n_hidden: int,
        emb_dim: int,
        teacher_buckets: int,
        materia_buckets: int,
        use_interaction: bool = True,
    ) -> None:
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.emb_dim = int(emb_dim)
        self.teacher_buckets = int(teacher_buckets)
        self.materia_buckets = int(materia_buckets)
        self.use_interaction = bool(use_interaction)

        self.teacher_emb = nn.Embedding(self.teacher_buckets, self.emb_dim)
        self.materia_emb = nn.Embedding(self.materia_buckets, self.emb_dim)

        in_dim = self.n_hidden + 2 * self.emb_dim + (self.emb_dim if self.use_interaction else 0)
        self.classifier = nn.Linear(in_dim, 3)

    def forward(self, h: Tensor, teacher_idx: Optional[Tensor] = None, materia_idx: Optional[Tensor] = None) -> Tensor:
        """
        :param h: Tensor (batch, n_hidden) con activaciones RBM.
        :param teacher_idx: Tensor (batch,) long con índices docente.
        :param materia_idx: Tensor (batch,) long con índices materia.
        :return: Logits (batch, 3).
        """
        bsz = h.shape[0]
        dev = h.device

        if teacher_idx is None:
            teacher_idx = torch.zeros((bsz,), dtype=torch.long, device=dev)
        if materia_idx is None:
            materia_idx = torch.zeros((bsz,), dtype=torch.long, device=dev)

        et = self.teacher_emb(teacher_idx)
        em = self.materia_emb(materia_idx)

        parts = [h, et, em]
        if self.use_interaction:
            parts.append(et * em)

        x = torch.cat(parts, dim=1)
        return self.classifier(x)


# =============================================================================
# Estrategia RBMRestringida
# =============================================================================

class RBMRestringida:
    """
    Estrategia RBM Restringida con métricas reales (Commit 4) y teacher/materia embeddings (Commit 5).

    Hparams principales
    -------------------
    - ``seed`` (int)
    - ``batch_size`` (int)
    - ``lr_rbm`` (float)
    - ``lr_head`` (float)
    - ``epochs_rbm`` (int)
    - ``n_hidden`` (int)
    - ``cd_k`` (int)
    - ``scale_mode``: minmax | scale_0_5

    Split / validación (Commit 4)
    -----------------------------
    - ``split_mode``: temporal | random (default temporal)
    - ``val_ratio``: float (default 0.2)
    - si hay columna ``split`` (train/val), tiene prioridad.

    Target (Commit 6)
    -----------------
    - ``target_mode``: sentiment_probs | label (default sentiment_probs)

    Features restringidas
    ---------------------
    - ``max_calif``: int (default 10)
    - ``use_text_probs``: bool (default False)  # deshabilitado si target_mode=sentiment_probs
    - ``use_text_embeds``: bool (default False)
    - ``text_embed_prefix``: str (default 'x_text_')

    Teacher/Materia (Commit 5)
    --------------------------
    - ``include_teacher_materia``: bool (default True)
    - ``teacher_materia_mode``: embed | hash | none (default embed)

    Modo embed:
    - ``tm_emb_dim``: int (default 16)
    - ``teacher_emb_buckets``: int (default 2048)
    - ``materia_emb_buckets``: int (default 2048)
    - ``tm_use_interaction``: bool (default True)

    Modo hash (legacy):
    - ``teacher_hash_dim`` / ``materia_hash_dim``: int (default 128)
    """

    def __init__(self, **kwargs: Any) -> None:
        # device/seed
        self.seed: int = int(kwargs.get("seed", 42))
        self.device: str = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        # hparams default
        self.batch_size: int = 64
        self.lr_rbm: float = 1e-2
        self.lr_head: float = 1e-2
        self.momentum: float = 0.5
        self.weight_decay: float = 0.0
        self.cd_k: int = 1
        self.epochs_rbm: int = 1
        self.scale_mode: str = "minmax"

        # split/target
        self.split_mode: str = "temporal"
        self.val_ratio: float = 0.2
        self.target_mode: str = "sentiment_probs"

        # restringida (features)
        self.max_calif: int = 10
        self.use_text_probs: bool = False
        self.use_text_embeds: bool = False
        self.text_embed_prefix_: str = "x_text_"

        # teacher/materia
        self.include_teacher_materia: bool = True
        self.teacher_materia_mode: str = "embed"  # embed | hash | none

        # hash legacy
        self.teacher_hash_dim: int = 128
        self.materia_hash_dim: int = 128

        # embed (commit 5)
        self.tm_emb_dim: int = 16
        self.teacher_emb_buckets: int = 2048
        self.materia_emb_buckets: int = 2048
        self.tm_use_interaction: bool = True

        # modelos
        self.vec: _Vectorizer = _Vectorizer()
        self.rbm: Optional[_RBM] = None
        self.head: Optional[nn.Module] = None
        self.opt_rbm: Optional[torch.optim.Optimizer] = None
        self.opt_head: Optional[torch.optim.Optimizer] = None

        # columnas/features (visibles RBM)
        self.feat_cols_: List[str] = []
        self.classes_: List[str] = list(_CLASSES)

        # datasets preparados
        self.X_train: Optional[Tensor] = None
        self.X_val: Optional[Tensor] = None
        self.y_train_hard: Optional[Tensor] = None
        self.y_val_hard: Optional[Tensor] = None
        self.y_train_soft: Optional[Tensor] = None
        self.y_val_soft: Optional[Tensor] = None

        # teacher/materia idx (solo modo embed)
        self.teacher_idx_train: Optional[Tensor] = None
        self.teacher_idx_val: Optional[Tensor] = None
        self.materia_idx_train: Optional[Tensor] = None
        self.materia_idx_val: Optional[Tensor] = None

        # tracking
        self._epoch: int = 0
        self._last_confusion_matrix: Optional[List[List[int]]] = None

    @property
    def feature_columns(self) -> List[str]:
        """Alias histórico para columnas de features."""
        return list(self.feat_cols_)

    # -------------------------------------------------------------------------
    # IO
    # -------------------------------------------------------------------------

    def _load_df(self, ref: str) -> pd.DataFrame:
        """
        Carga DF desde parquet/csv.

        :param ref: Ruta local al dataset.
        :return: DataFrame.
        """
        if ref is None:
            raise ValueError("data_ref is None")
        p = str(ref)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)

    # -------------------------------------------------------------------------
    # Teacher/Materia: hash (legacy) y embed (Commit 5)
    # -------------------------------------------------------------------------

    def _add_teacher_materia_hash_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega columnas hash one-hot determinísticas para teacher_key/materia_key (modo ``hash``).

        Crea:
          - teacher_h_0..teacher_h_(dim-1)
          - materia_h_0..materia_h_(dim-1)

        Si el modo no es ``hash``, no altera el DF.

        :param df: DataFrame de entrada.
        :return: DataFrame con columnas adicionales (solo si aplica).
        """
        if not self.include_teacher_materia:
            return df
        if self.teacher_materia_mode != "hash":
            return df

        out = df

        if "teacher_key" in out.columns and self.teacher_hash_dim > 0:
            idxs = out["teacher_key"].astype("string").fillna("").map(
                lambda s: _stable_hash_index(str(s), self.teacher_hash_dim)
            ).to_numpy(dtype=np.int64)
            mat = np.zeros((len(out), self.teacher_hash_dim), dtype=np.float32)
            mat[np.arange(len(out)), idxs] = 1.0
            for j in range(self.teacher_hash_dim):
                out[f"teacher_h_{j}"] = mat[:, j]

        if "materia_key" in out.columns and self.materia_hash_dim > 0:
            idxs = out["materia_key"].astype("string").fillna("").map(
                lambda s: _stable_hash_index(str(s), self.materia_hash_dim)
            ).to_numpy(dtype=np.int64)
            mat = np.zeros((len(out), self.materia_hash_dim), dtype=np.float32)
            mat[np.arange(len(out)), idxs] = 1.0
            for j in range(self.materia_hash_dim):
                out[f"materia_h_{j}"] = mat[:, j]

        return out

    def _build_teacher_materia_indices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye índices discretos para teacher/materia usando hashing estable (modo ``embed``).

        Esto evita mantener un vocabulario explícito y produce índices estables entre ejecuciones.

        :param df: DataFrame (debe tener teacher_key/materia_key para aprovecharlo).
        :return: (teacher_idx, materia_idx), arrays int64 tamaño n.
        """
        n = len(df)

        if not self.include_teacher_materia or self.teacher_materia_mode != "embed":
            return (
                np.zeros((n,), dtype=np.int64),
                np.zeros((n,), dtype=np.int64),
            )

        if "teacher_key" in df.columns:
            t_raw = df["teacher_key"].astype("string").fillna("")
        else:
            t_raw = pd.Series([""] * n)

        if "materia_key" in df.columns:
            m_raw = df["materia_key"].astype("string").fillna("")
        else:
            m_raw = pd.Series([""] * n)

        teacher_idx = t_raw.map(lambda s: _stable_hash_index(str(s), self.teacher_emb_buckets)).to_numpy(np.int64)
        materia_idx = m_raw.map(lambda s: _stable_hash_index(str(s), self.materia_emb_buckets)).to_numpy(np.int64)
        return teacher_idx, materia_idx

    # -------------------------------------------------------------------------
    # Targets (Commit 6)
    # -------------------------------------------------------------------------

    def _extract_targets(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Construye targets hard y/o soft.

        Commit 6:
        - Si ``target_mode='sentiment_probs'`` y existen p_*:
          - y_soft = normalize(p_*)
          - y_hard = argmax(y_soft) (para métricas/confusion)
        - Si ``target_mode='sentiment_probs'`` y faltan p_*:
          - fallback controlado a label hard si existe
          - si no existe label hard, error explícito

        En modo ``label``:
        - usa label hard si existe
        - si no existe label hard pero hay p_*, usa argmax(p_*) SOLO para métricas (sin soft targets)

        :return: (y_hard[int64] o None, y_soft[float32](n,3) o None)
        """
        # (A) Soft labels (probabilidades)
        if self.target_mode == "sentiment_probs":
            if all(c in df.columns for c in _PROB_COLS):
                probs = normalize_probs(df[_PROB_COLS].to_numpy(dtype=np.float32))
                y_soft = probs.astype(np.float32, copy=False)
                y_hard = soft_to_hard(probs)
                return y_hard, y_soft

            # fallback a label hard si faltan p_*
            label_col = next((c for c in _LABEL_COL_CANDIDATES if c in df.columns), None)
            if label_col is None:
                raise ValueError(
                    "target_mode='sentiment_probs' pero faltan columnas p_neg/p_neu/p_pos "
                    "y no existe ninguna columna de label hard (sent_label/sentiment_label/etc)."
                )
            # cambiar modo internamente para consistencia del run
            self.target_mode = "label"

        # (B) Hard labels
        label_col = next((c for c in _LABEL_COL_CANDIDATES if c in df.columns), None)

        if label_col is None and all(c in df.columns for c in _PROB_COLS):
            # fallback: métricas usando argmax(prob) (sin soft targets)
            probs = normalize_probs(df[_PROB_COLS].to_numpy(dtype=np.float32))
            y_hard = soft_to_hard(probs)
            return y_hard, None

        if label_col is None:
            return None, None

        y_raw = df[label_col].astype("string").fillna("").map(_norm_label)
        y_hard_full = np.array([_LABEL_MAP.get(s, -1) for s in y_raw.tolist()], dtype=np.int64)
        return y_hard_full, None

    # -------------------------------------------------------------------------
    # Split train/val (Commit 4)
    # -------------------------------------------------------------------------

    def _split_indices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determina índices train/val.

        Prioridad:
        1) columna ``split`` (train/val)
        2) temporal por ``periodo`` si split_mode=temporal
        3) random con val_ratio

        :param df: DataFrame.
        :return: (idx_train, idx_val)
        """
        n = len(df)
        if n < 2:
            idx = np.arange(n, dtype=np.int64)
            return idx, idx

        if "split" in df.columns:
            s = df["split"].astype("string").fillna("").str.lower()
            idx_train = np.where(s == "train")[0].astype(np.int64)
            idx_val = np.where(s.isin(["val", "valid", "validation"]))[0].astype(np.int64)
            if idx_train.size > 0 and idx_val.size > 0:
                return idx_train, idx_val

        if str(self.split_mode).lower() == "temporal" and "periodo" in df.columns:
            keys = df["periodo"].map(_parse_periodo_to_sortkey).to_list()
            order = np.argsort(np.array(keys, dtype=object), kind="stable")
            n_val = max(1, int(round(n * float(self.val_ratio))))
            idx_val = order[-n_val:].astype(np.int64)
            idx_train = order[:-n_val].astype(np.int64)
            if idx_train.size == 0:
                idx_train = idx_val
            return idx_train, idx_val

        rng = np.random.RandomState(self.seed)
        order = rng.permutation(n).astype(np.int64)
        n_val = max(1, int(round(n * float(self.val_ratio))))
        idx_val = order[:n_val]
        idx_train = order[n_val:]
        if idx_train.size == 0:
            idx_train = idx_val
        return idx_train, idx_val

    # -------------------------------------------------------------------------
    # Head helper (embed vs legacy)
    # -------------------------------------------------------------------------

    def _head_logits(
        self,
        h: Tensor,
        *,
        teacher_idx: Optional[Tensor] = None,
        materia_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calcula logits usando la head correspondiente (embed o legacy).

        :param h: Hidden RBM (batch, n_hidden).
        :param teacher_idx: Índices teacher si modo embed.
        :param materia_idx: Índices materia si modo embed.
        :return: Logits (batch, 3).
        """
        if self.head is None:
            raise RuntimeError("Head no inicializada.")
        if self.teacher_materia_mode == "embed":
            assert isinstance(self.head, _TeacherMateriaHead)
            return self.head(h, teacher_idx=teacher_idx, materia_idx=materia_idx)
        assert isinstance(self.head, nn.Linear)
        return self.head(h)

    # -------------------------------------------------------------------------
    # API pública (PlantillaEntrenamiento)
    # -------------------------------------------------------------------------

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        """
        Prepara el entrenamiento (Commit 4 + Commit 5 + Commit 6).

        Pasos:
        - Leer DF.
        - Configurar split/target y flags de features.
        - Teacher/Materia:
          - si mode='hash': generar columnas one-hot por hashing (visibles)
          - si mode='embed': construir índices discretos (head)
          - si mode='none': ignorar
        - Construir targets según target_mode (Commit 6).
        - Validar que exista supervisión (soft o hard); si no, error.
        - Split train/val.
        - Vectorización (fit SOLO en train).
        - Inicializar RBM + head (embed o legacy).
        - Preentrenar RBM (unsupervised) usando train.
        """
        hp = {str(k).lower(): v for k, v in (hparams or {}).items()}

        # seed/device
        self.seed = int(hp.get("seed", self.seed))
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        use_cuda = bool(hp.get("use_cuda", False))
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else self.device

        # core hparams
        self.batch_size = int(hp.get("batch_size", self.batch_size))
        self.cd_k = int(hp.get("cd_k", self.cd_k))
        self.lr_rbm = float(hp.get("lr_rbm", self.lr_rbm))
        self.lr_head = float(hp.get("lr_head", self.lr_head))
        self.momentum = float(hp.get("momentum", self.momentum))
        self.weight_decay = float(hp.get("weight_decay", self.weight_decay))
        self.epochs_rbm = int(hp.get("epochs_rbm", self.epochs_rbm))
        self.scale_mode = str(hp.get("scale_mode", self.scale_mode))

        # split/target
        self.split_mode = str(hp.get("split_mode", self.split_mode)).lower()
        self.val_ratio = float(hp.get("val_ratio", self.val_ratio))
        self.target_mode = str(hp.get("target_mode", self.target_mode)).lower()

        # restringida feature flags
        self.max_calif = int(hp.get("max_calif", self.max_calif))
        self.use_text_probs = bool(hp.get("use_text_probs", self.use_text_probs))
        self.use_text_embeds = bool(hp.get("use_text_embeds", self.use_text_embeds))
        self.text_embed_prefix_ = str(hp.get("text_embed_prefix", self.text_embed_prefix_))

        # leakage guard
        if self.target_mode == "sentiment_probs":
            self.use_text_probs = False

        # teacher/materia
        self.include_teacher_materia = bool(hp.get("include_teacher_materia", self.include_teacher_materia))
        self.teacher_materia_mode = str(hp.get("teacher_materia_mode", self.teacher_materia_mode)).lower()

        if self.teacher_materia_mode not in ("embed", "hash", "none"):
            self.teacher_materia_mode = "embed"
        if not self.include_teacher_materia:
            self.teacher_materia_mode = "none"

        # hash legacy dims
        self.teacher_hash_dim = int(hp.get("teacher_hash_dim", self.teacher_hash_dim))
        self.materia_hash_dim = int(hp.get("materia_hash_dim", self.materia_hash_dim))

        # embed
        self.tm_emb_dim = int(hp.get("tm_emb_dim", self.tm_emb_dim))
        self.teacher_emb_buckets = int(hp.get("teacher_emb_buckets", self.teacher_emb_buckets))
        self.materia_emb_buckets = int(hp.get("materia_emb_buckets", self.materia_emb_buckets))
        self.tm_use_interaction = bool(hp.get("tm_use_interaction", self.tm_use_interaction))

        # load df
        df = self._load_df(data_ref)

        # apply hash features only if mode == 'hash'
        df = self._add_teacher_materia_hash_features(df)

        # build indices only if mode == 'embed' (otherwise zeros)
        teacher_idx_full, materia_idx_full = self._build_teacher_materia_indices(df)

        # targets (Commit 6)
        y_hard_full, y_soft_full = self._extract_targets(df)

        # Validación: debe existir supervisión (soft o hard)
        if y_hard_full is None and y_soft_full is None:
            raise ValueError(
                "No se encontraron targets para entrenamiento. "
                "Revisa que existan p_neg/p_neu/p_pos (para sentiment_probs) "
                "o una columna de label hard (sent_label/sentiment_label/etc)."
            )

        # filtrar labels inválidas si es hard (-1)
        if y_hard_full is not None:
            valid = (y_hard_full >= 0) & (y_hard_full <= 2)
            if valid.sum() == 0:
                raise ValueError("No hay etiquetas válidas (neg/neu/pos) para entrenamiento.")
            if valid.sum() < len(df):
                df = df.loc[valid].reset_index(drop=True)
                teacher_idx_full = teacher_idx_full[valid]
                materia_idx_full = materia_idx_full[valid]
                y_hard_full = y_hard_full[valid]
                if y_soft_full is not None:
                    y_soft_full = y_soft_full[valid]

        # feature columns (visibles RBM)
        feat_cols = _pick_feature_cols(
            df,
            max_calif=self.max_calif,
            include_text_probs=self.use_text_probs,
            include_text_embeds=self.use_text_embeds,
            text_embed_prefix=self.text_embed_prefix_,
            target_mode=self.target_mode,
        )

        # en modo hash, incluir columnas one-hot
        if self.teacher_materia_mode == "hash":
            hash_cols = [c for c in df.columns if c.startswith("teacher_h_") or c.startswith("materia_h_")]
            hash_cols = sorted(hash_cols)
            feat_cols = list(dict.fromkeys(feat_cols + hash_cols))

        if not feat_cols:
            raise ValueError("No se detectaron columnas de features para RBMRestringida.")

        self.feat_cols_ = list(feat_cols)
        X_full = df[self.feat_cols_].to_numpy(dtype=np.float32)

        # split
        idx_train, idx_val = self._split_indices(df)
        X_tr = X_full[idx_train]
        X_va = X_full[idx_val]

        # vectorizer fit SOLO en train
        mode = "scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"
        self.vec = _Vectorizer().fit(X_tr, mode=mode)
        X_tr_s = self.vec.transform(X_tr)
        X_va_s = self.vec.transform(X_va)

        self.X_train = torch.from_numpy(X_tr_s).to(self.device)
        self.X_val = torch.from_numpy(X_va_s).to(self.device)

        if y_hard_full is not None:
            self.y_train_hard = torch.from_numpy(y_hard_full[idx_train].astype(np.int64)).to(self.device)
            self.y_val_hard = torch.from_numpy(y_hard_full[idx_val].astype(np.int64)).to(self.device)
        else:
            self.y_train_hard = None
            self.y_val_hard = None

        if y_soft_full is not None:
            self.y_train_soft = torch.from_numpy(y_soft_full[idx_train].astype(np.float32)).to(self.device)
            self.y_val_soft = torch.from_numpy(y_soft_full[idx_val].astype(np.float32)).to(self.device)
        else:
            self.y_train_soft = None
            self.y_val_soft = None

        # teacher/materia idx tensors (solo embed)
        if self.teacher_materia_mode == "embed":
            self.teacher_idx_train = torch.from_numpy(teacher_idx_full[idx_train].astype(np.int64)).to(self.device)
            self.teacher_idx_val = torch.from_numpy(teacher_idx_full[idx_val].astype(np.int64)).to(self.device)
            self.materia_idx_train = torch.from_numpy(materia_idx_full[idx_train].astype(np.int64)).to(self.device)
            self.materia_idx_val = torch.from_numpy(materia_idx_full[idx_val].astype(np.int64)).to(self.device)
        else:
            self.teacher_idx_train = None
            self.teacher_idx_val = None
            self.materia_idx_train = None
            self.materia_idx_val = None

        # init rbm/head
        n_visible = int(self.X_train.shape[1])
        n_hidden = int(hp.get("n_hidden", 32))

        self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)

        if self.teacher_materia_mode == "embed":
            self.head = _TeacherMateriaHead(
                n_hidden=n_hidden,
                emb_dim=self.tm_emb_dim,
                teacher_buckets=self.teacher_emb_buckets,
                materia_buckets=self.materia_emb_buckets,
                use_interaction=self.tm_use_interaction,
            ).to(self.device)
        else:
            self.head = nn.Linear(n_hidden, 3).to(self.device)

        self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        # pretrain rbm (unsupervised)
        self._pretrain_rbm()

        self._epoch = 0
        self._last_confusion_matrix = None

    def _pretrain_rbm(self) -> None:
        """
        Preentrenamiento no-supervisado de RBM sobre X_train.

        .. note::
           Este paso ocurre antes del ciclo de épocas visible en UI.
        """
        if self.rbm is None or self.opt_rbm is None or self.X_train is None:
            raise RuntimeError("RBMRestringida no inicializada (setup no ejecutado).")

        self.rbm.train()
        X = self.X_train

        for _ in range(max(1, int(self.epochs_rbm))):
            self.opt_rbm.zero_grad(set_to_none=True)
            _ = self.rbm.cd_step(X)
            self.opt_rbm.step()

        self.rbm.eval()

    def train_step(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """
        Ejecuta una época de entrenamiento supervisado (head) y reporta métricas.

        Flujo:
        (A) RBM: un paso CD por época (diagnóstico recon_error)
        (B) Head: entrenamiento supervisado con targets soft/hard (Commit 6)
        (C) Métricas reales train/val + confusion matrix

        :param epoch: Número de época (1-indexed).
        :return: (loss_total, metrics)
        """
        t0 = time.perf_counter()

        if self.rbm is None or self.head is None or self.opt_rbm is None or self.opt_head is None:
            raise RuntimeError("RBMRestringida no inicializada (setup no ejecutado).")
        if self.X_train is None or self.X_val is None:
            raise RuntimeError("X_train/X_val no disponibles.")

        self._epoch = int(epoch)

        # ---------------------------------------------------------
        # (A) RBM: un paso CD por época (diagnóstico recon_error)
        # ---------------------------------------------------------
        self.rbm.train()
        self.opt_rbm.zero_grad(set_to_none=True)
        rbm_diag = self.rbm.cd_step(self.X_train)
        self.opt_rbm.step()
        recon_error = float(rbm_diag.get("recon_error", 0.0))
        rbm_grad_norm = float(rbm_diag.get("grad_norm", 0.0))

        # ---------------------------------------------------------
        # (B) Head: supervised (soft o hard)
        # ---------------------------------------------------------
        self.rbm.eval()
        self.head.train()

        X = self.X_train
        n = int(X.shape[0])
        bs = max(1, int(self.batch_size))

        rng = np.random.RandomState(self.seed + epoch)
        order = rng.permutation(n)

        cls_losses: List[float] = []

        for i0 in range(0, n, bs):
            idx = order[i0 : i0 + bs]
            xb = X[idx]

            with torch.no_grad():
                hb = self.rbm.hidden_probs(xb)

            # teacher/materia idx batch si embed
            t_idx = None
            m_idx = None
            if self.teacher_materia_mode == "embed":
                if self.teacher_idx_train is not None:
                    t_idx = self.teacher_idx_train[idx]
                if self.materia_idx_train is not None:
                    m_idx = self.materia_idx_train[idx]

            logits = self._head_logits(hb, teacher_idx=t_idx, materia_idx=m_idx)

            # Commit 6: soft cross-entropy si hay soft targets
            if self.target_mode == "sentiment_probs" and self.y_train_soft is not None:
                yb = self.y_train_soft[idx]
                logp = F.log_softmax(logits, dim=1)
                loss = -(yb * logp).sum(dim=1).mean()
            else:
                if self.y_train_hard is None:
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    yb = self.y_train_hard[idx]
                    loss = F.cross_entropy(logits, yb)

            self.opt_head.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_head.step()

            cls_losses.append(float(loss.detach().cpu().item()))

        cls_loss = float(np.mean(cls_losses)) if cls_losses else 0.0

        # ---------------------------------------------------------
        # (C) Métricas train/val reales
        # ---------------------------------------------------------
        metrics = self._evaluate(train_loss=float(recon_error + cls_loss))

        metrics["recon_error"] = float(recon_error)
        metrics["rbm_grad_norm"] = float(rbm_grad_norm)
        metrics["cls_loss"] = float(cls_loss)

        # confusion matrix + labels
        metrics["confusion_matrix"] = self._last_confusion_matrix
        metrics["labels"] = list(_CLASSES)

        metrics["teacher_materia_mode"] = str(self.teacher_materia_mode)
        metrics["target_mode"] = str(self.target_mode)
        metrics["time_epoch_ms"] = float((time.perf_counter() - t0) * 1000.0)

        loss_out = float(recon_error + cls_loss)
        return loss_out, metrics

    def _evaluate(self, train_loss: float) -> Dict[str, Any]:
        """
        Evalúa métricas en train y val.

        Retorna métricas numéricas:
          - loss, accuracy, f1_macro
          - val_loss, val_accuracy, val_f1_macro

        Además actualiza ``self._last_confusion_matrix`` para val.

        Commit 6:
        - val_loss usa soft CE si target_mode=sentiment_probs y y_val_soft existe.

        :param train_loss: Loss total reportado en train.
        :return: Dict de métricas.
        """
        assert self.rbm is not None and self.head is not None
        assert self.X_train is not None and self.X_val is not None

        self.rbm.eval()
        self.head.eval()

        # ---- train ----
        with torch.no_grad():
            Htr = self.rbm.hidden_probs(self.X_train)

            t_tr = self.teacher_idx_train if self.teacher_materia_mode == "embed" else None
            m_tr = self.materia_idx_train if self.teacher_materia_mode == "embed" else None

            logits_tr = self._head_logits(Htr, teacher_idx=t_tr, materia_idx=m_tr)
            proba_tr = F.softmax(logits_tr, dim=1)
            pred_tr = torch.argmax(proba_tr, dim=1).cpu().numpy()

        if self.y_train_hard is not None:
            ytr = self.y_train_hard.cpu().numpy()
        elif self.y_train_soft is not None:
            ytr = torch.argmax(self.y_train_soft, dim=1).cpu().numpy()
        else:
            ytr = pred_tr.copy()

        acc_tr = accuracy(ytr, pred_tr)
        f1_tr = f1_macro(ytr, pred_tr, n_classes=3)

        # ---- val ----
        with torch.no_grad():
            Hva = self.rbm.hidden_probs(self.X_val)

            t_va = self.teacher_idx_val if self.teacher_materia_mode == "embed" else None
            m_va = self.materia_idx_val if self.teacher_materia_mode == "embed" else None

            logits_va = self._head_logits(Hva, teacher_idx=t_va, materia_idx=m_va)
            proba_va = F.softmax(logits_va, dim=1)
            pred_va = torch.argmax(proba_va, dim=1).cpu().numpy()

        if self.y_val_hard is not None:
            yva = self.y_val_hard.cpu().numpy()
        elif self.y_val_soft is not None:
            yva = torch.argmax(self.y_val_soft, dim=1).cpu().numpy()
        else:
            yva = pred_va.copy()

        # val loss coherente con target_mode
        if self.target_mode == "sentiment_probs" and self.y_val_soft is not None:
            y_soft = self.y_val_soft
            logp = F.log_softmax(logits_va, dim=1)
            vloss = float((-(y_soft * logp).sum(dim=1).mean()).detach().cpu().item())
        else:
            y_hard = torch.from_numpy(yva.astype(np.int64)).to(self.device)
            vloss = float(F.cross_entropy(logits_va, y_hard).detach().cpu().item())

        acc_va = accuracy(yva, pred_va)
        f1_va = f1_macro(yva, pred_va, n_classes=3)

        self._last_confusion_matrix = confusion_matrix(yva, pred_va, n_classes=3)

        return {
            "loss": float(train_loss),
            "accuracy": float(acc_tr),
            "f1_macro": float(f1_tr),
            "val_loss": float(vloss),
            "val_accuracy": float(acc_va),
            "val_f1_macro": float(f1_va),
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
        }

    # -------------------------------------------------------------------------
    # Inferencia
    # -------------------------------------------------------------------------

    def _df_to_X(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convierte DF a X respetando ``feat_cols_``.

        - Si faltan columnas, se rellenan con 0.0.
        - Si el modelo fue entrenado en modo ``hash``, recrea teacher_h_*/materia_h_*.

        :param df: DataFrame de entrada.
        :return: Matriz numpy float32.
        """
        if not self.feat_cols_:
            raise RuntimeError("El modelo no tiene feat_cols_ configuradas.")

        if self.teacher_materia_mode == "hash" and self.include_teacher_materia:
            df = self._add_teacher_materia_hash_features(df.copy())

        missing = [c for c in self.feat_cols_ if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0

        return df[self.feat_cols_].to_numpy(dtype=np.float32)

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades (n,3) para un DataFrame.

        - En modo ``embed`` calcula índices teacher/materia on-the-fly.
        - En modo ``hash`` reconstruye columnas hash si es necesario.

        :param df: DataFrame de entrada.
        :return: numpy array (n,3) con orden [neg, neu, pos].
        """
        if self.rbm is None or self.head is None:
            raise RuntimeError("Modelo no cargado/entrenado.")

        X_np = self._df_to_X(df.copy())
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)

        t_idx = None
        m_idx = None
        if self.teacher_materia_mode == "embed" and self.include_teacher_materia:
            t_np, m_np = self._build_teacher_materia_indices(df)
            t_idx = torch.from_numpy(t_np.astype(np.int64)).to(self.device)
            m_idx = torch.from_numpy(m_np.astype(np.int64)).to(self.device)

        self.rbm.eval()
        self.head.eval()
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
            logits = self._head_logits(H, teacher_idx=t_idx, materia_idx=m_idx)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_df(self, df: pd.DataFrame) -> List[str]:
        """
        Predice etiquetas string {neg,neu,pos} para un DataFrame.

        :param df: DataFrame.
        :return: Lista de etiquetas.
        """
        idx = self.predict_proba_df(df).argmax(axis=1)
        return [_INV_LABEL_MAP[int(i)] for i in idx]

    def predict_proba(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Wrapper: acepta np.ndarray o DataFrame.

        .. note::
           Si usas np.ndarray en modo embed, no podrás pasar teacher/materia; usa DataFrame.

        :param X_or_df: Matriz o DataFrame.
        :return: Probabilidades (n,3).
        """
        if isinstance(X_or_df, pd.DataFrame):
            return self.predict_proba_df(X_or_df)

        if self.rbm is None or self.head is None:
            raise RuntimeError("Modelo no cargado/entrenado.")

        X_np = np.asarray(X_or_df, dtype=np.float32)
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)

        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
            logits = self._head_logits(H, teacher_idx=None, materia_idx=None)
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """
        Predice etiquetas string para np.ndarray o DataFrame.

        :param X_or_df: Matriz o DataFrame.
        :return: Lista de etiquetas.
        """
        proba = self.predict_proba(X_or_df)
        idx = proba.argmax(axis=1)
        return [_INV_LABEL_MAP[int(i)] for i in idx]

    # -------------------------------------------------------------------------
    # Persistencia (para artifacts/runs y champions)
    # -------------------------------------------------------------------------

    def save(self, out_dir: str) -> None:
        """
        Guarda pesos y metadatos del modelo.

        Archivos:
        - vectorizer.json
        - rbm.pt
        - head.pt
        - meta.json

        :param out_dir: Directorio de salida.
        """
        if self.rbm is None or self.head is None:
            raise RuntimeError("No se puede guardar: modelo no entrenado/cargado.")

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # vectorizer
        with open(os.path.join(out_dir, "vectorizer.json"), "w", encoding="utf-8") as f:
            json.dump(self.vec.to_dict(), f, ensure_ascii=False, indent=2)

        # rbm
        torch.save(
            {
                "state_dict": self.rbm.state_dict(),
                "n_visible": int(self.rbm.W.shape[0]),
                "n_hidden": int(self.rbm.W.shape[1]),
                "cd_k": int(self.rbm.cd_k),
                "seed": int(self.seed),
            },
            os.path.join(out_dir, "rbm.pt"),
        )

        # head
        torch.save(
            {"state_dict": self.head.state_dict(), "classes": list(self.classes_)},
            os.path.join(out_dir, "head.pt"),
        )

        # meta
        meta = {
            "feat_cols_": list(self.feat_cols_),
            "scale_mode": self.scale_mode,
            "target_mode": self.target_mode,
            "split_mode": self.split_mode,
            "val_ratio": float(self.val_ratio),
            "max_calif": int(self.max_calif),
            "use_text_probs": bool(self.use_text_probs),
            "use_text_embeds": bool(self.use_text_embeds),
            "text_embed_prefix": self.text_embed_prefix_,

            # teacher/materia
            "include_teacher_materia": bool(self.include_teacher_materia),
            "teacher_materia_mode": str(self.teacher_materia_mode),

            # hash legacy
            "teacher_hash_dim": int(self.teacher_hash_dim),
            "materia_hash_dim": int(self.materia_hash_dim),

            # embed
            "tm_emb_dim": int(self.tm_emb_dim),
            "teacher_emb_buckets": int(self.teacher_emb_buckets),
            "materia_emb_buckets": int(self.materia_emb_buckets),
            "tm_use_interaction": bool(self.tm_use_interaction),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: str, device: Optional[str] = None) -> "RBMRestringida":
        """
        Carga un modelo guardado con :meth:`save`.

        Compatibilidad:
        - si ``meta.json`` no tiene teacher_materia_mode (runs antiguos),
          se asume ``hash`` si hay dims > 0; en caso contrario ``none``.

        :param in_dir: Directorio con rbm.pt/head.pt/vectorizer.json/meta.json.
        :param device: 'cpu' o 'cuda' (default auto).
        :return: Instancia cargada.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj = cls()
        obj.device = device

        # meta
        meta_path = os.path.join(in_dir, "meta.json")
        meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

            obj.feat_cols_ = list(meta.get("feat_cols_") or meta.get("feat_cols") or [])
            obj.scale_mode = str(meta.get("scale_mode", obj.scale_mode))
            obj.target_mode = str(meta.get("target_mode", obj.target_mode))
            obj.split_mode = str(meta.get("split_mode", obj.split_mode))
            obj.val_ratio = float(meta.get("val_ratio", obj.val_ratio))
            obj.max_calif = int(meta.get("max_calif", obj.max_calif))
            obj.use_text_probs = bool(meta.get("use_text_probs", obj.use_text_probs))
            obj.use_text_embeds = bool(meta.get("use_text_embeds", obj.use_text_embeds))
            obj.text_embed_prefix_ = str(meta.get("text_embed_prefix", obj.text_embed_prefix_))

            obj.include_teacher_materia = bool(meta.get("include_teacher_materia", obj.include_teacher_materia))

            # teacher_materia_mode (compat)
            tmm = meta.get("teacher_materia_mode")
            if tmm is None:
                th = int(meta.get("teacher_hash_dim", 0))
                mh = int(meta.get("materia_hash_dim", 0))
                tmm = "hash" if (th > 0 or mh > 0) else "none"
            obj.teacher_materia_mode = str(tmm).lower()

            # legacy hash dims
            obj.teacher_hash_dim = int(meta.get("teacher_hash_dim", obj.teacher_hash_dim))
            obj.materia_hash_dim = int(meta.get("materia_hash_dim", obj.materia_hash_dim))

            # embed
            obj.tm_emb_dim = int(meta.get("tm_emb_dim", obj.tm_emb_dim))
            obj.teacher_emb_buckets = int(meta.get("teacher_emb_buckets", obj.teacher_emb_buckets))
            obj.materia_emb_buckets = int(meta.get("materia_emb_buckets", obj.materia_emb_buckets))
            obj.tm_use_interaction = bool(meta.get("tm_use_interaction", obj.tm_use_interaction))

        # vectorizer
        vec_path = os.path.join(in_dir, "vectorizer.json")
        if os.path.exists(vec_path):
            obj.vec = _Vectorizer.from_dict(json.loads(Path(vec_path).read_text(encoding="utf-8")))

        # rbm
        rbm_ckpt = torch.load(os.path.join(in_dir, "rbm.pt"), map_location=device)
        obj.seed = int(rbm_ckpt.get("seed", obj.seed))
        obj.rbm = _RBM(
            n_visible=int(rbm_ckpt["n_visible"]),
            n_hidden=int(rbm_ckpt["n_hidden"]),
            cd_k=int(rbm_ckpt.get("cd_k", 1)),
            seed=int(obj.seed),
        ).to(device)
        obj.rbm.load_state_dict(rbm_ckpt["state_dict"])

        # head (según modo)
        n_hidden = int(rbm_ckpt["n_hidden"])
        if obj.teacher_materia_mode == "embed" and obj.include_teacher_materia:
            obj.head = _TeacherMateriaHead(
                n_hidden=n_hidden,
                emb_dim=obj.tm_emb_dim,
                teacher_buckets=obj.teacher_emb_buckets,
                materia_buckets=obj.materia_emb_buckets,
                use_interaction=obj.tm_use_interaction,
            ).to(device)
        else:
            obj.head = nn.Linear(n_hidden, 3).to(device)

        head_ckpt = torch.load(os.path.join(in_dir, "head.pt"), map_location=device)
        obj.head.load_state_dict(head_ckpt["state_dict"])
        obj.classes_ = list(head_ckpt.get("classes", _CLASSES))

        # runtime tensors (no se restauran)
        obj.X_train = None
        obj.X_val = None
        obj.y_train_hard = None
        obj.y_val_hard = None
        obj.y_train_soft = None
        obj.y_val_soft = None
        obj.teacher_idx_train = None
        obj.teacher_idx_val = None
        obj.materia_idx_train = None
        obj.materia_idx_val = None
        obj._epoch = 0
        obj._last_confusion_matrix = None

        return obj

    # -------------------------------------------------------------------------
    # Compat: fit() legacy (para scripts antiguos)
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, List[int]]] = None,
        epochs: int = 1,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Entrenamiento "legacy" (no recomendado para la app).

        Si se provee X/y:
        - Crea un pseudo dataset en memoria (sin split sofisticado)
        - Ejecuta train_step por `epochs`

        .. note::
           En la aplicación, el entrenamiento oficial se ejecuta con PlantillaEntrenamiento
           que llama a :meth:`setup` con un data_ref real.
        """
        if X is None:
            raise RuntimeError("fit() legacy requiere X (np.ndarray o DataFrame).")

        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if y is not None:
                df["y"] = np.asarray(y, dtype=np.int64)
            tmp = df
        else:
            X_np = np.asarray(X, dtype=np.float32)
            df = pd.DataFrame(X_np, columns=[f"x_{i}" for i in range(X_np.shape[1])])
            if y is not None:
                df["y"] = np.asarray(y, dtype=np.int64)
            tmp = df

        # construir un split simple random dentro del DF
        rng = np.random.RandomState(self.seed)
        n = len(tmp)
        order = rng.permutation(n)
        n_val = max(1, int(round(n * float(self.val_ratio))))
        val_idx = set(order[:n_val].tolist())
        tmp["split"] = ["val" if i in val_idx else "train" for i in range(n)]

        # materializar a parquet temporal para reusar setup()
        tmp_dir = Path(".tmp_rbm_restringida")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "fit_tmp.parquet"
        tmp.to_parquet(tmp_path)

        self.setup(str(tmp_path), hparams={})
        last_metrics: Dict[str, Any] = {}
        last_loss = 0.0
        for ep in range(1, int(epochs) + 1):
            last_loss, last_metrics = self.train_step(ep)

        out = dict(last_metrics)
        out["loss"] = float(last_loss)
        return out


# Alias para compatibilidad
ModeloRBMRestringida = RBMRestringida
