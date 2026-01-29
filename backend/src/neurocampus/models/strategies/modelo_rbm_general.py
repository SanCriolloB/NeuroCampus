"""
neurocampus.models.strategies.modelo_rbm_general
===============================================

Estrategia **RBM General** con:

- Normalización robusta (Vectorizer minmax o scale_0_5).
- Soporte de entrenamiento orquestado por
  :class:`~neurocampus.models.templates.plantilla_entrenamiento.PlantillaEntrenamiento`
  mediante el contrato:
  - :meth:`setup(data_ref, hparams)`
  - :meth:`train_step(epoch) -> (loss, metrics)`

Métricas reales (train/val) + confusion matrix
---------------------------------------------------------
Se agregan:

- **Split real** train/val configurable por:
  - ``split_mode``: ``temporal`` | ``random``
  - ``val_ratio``: proporción de validación (default 0.2)
  - Si existe columna ``split`` (train/val/test), se usa como fuente prioritaria.
- Métricas por época (numéricas) que el frontend puede graficar vía eventos:
  - ``accuracy``, ``f1_macro``, ``val_accuracy``, ``val_f1_macro``, ``val_loss``
- Métrica final no-numérica:
  - ``confusion_matrix`` (3x3) + ``labels``

Objetivo (target) y fugas de información
----------------------------------------
El flujo actualizado define que el objetivo sea **sentiment_probs** (``p_neg,p_neu,p_pos``).
Por lo tanto:

- Si ``target_mode='sentiment_probs'``:
  - **Los p_*** se usan como **target** (soft labels), NO como features.
- Si ``target_mode='label'``:
  - Se usan columnas de etiqueta (``sent_label``/``sentiment_label``/etc.) como target hard.

.. important::
   Evitar incluir ``p_neg/p_neu/p_pos`` como features cuando también son el target
   (esto inflaría métricas por leakage).

Teacher/Materia (opcional)
--------------------------
Si ``include_teacher_materia=True`` y el DataFrame trae columnas string
``teacher_key`` y/o ``materia_key``, se agregan features con **hashing one-hot**
de dimensión fija (por defecto 128 cada uno) para capturar relaciones sin crear
one-hot gigantes.

Persistencia
------------
La persistencia de runs/champion se maneja fuera (router + runs_io).
Esta estrategia implementa ``save(out_dir)`` para guardar:

- ``rbm.pt`` y ``head.pt`` (pesos)
- ``vectorizer.json`` y ``meta.json`` (feat_cols, hashing dims, etc.)

Notas sobre utilidades de métricas
----------------------------------
Este archivo usa :mod:`neurocampus.models.utils.metrics` para:

- accuracy
- f1_macro
- confusion_matrix
- normalize_probs
- soft_to_hard

Así se evita duplicación entre estrategias.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurocampus.models.utils.metrics import (
    accuracy,
    confusion_matrix,
    f1_macro,
    normalize_probs,
    soft_to_hard,
)

# =============================================================================
# Etiquetas / mapeos
# =============================================================================

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}

_PROB_COLS = ["p_neg", "p_neu", "p_pos"]

# Prefijos candidatos de embeddings de texto (autodetección)
_CANDIDATE_EMBED_PREFIXES = [
    "x_text_",         # por defecto del proyecto
    "text_embed_",
    "text_",
    "feat_text_",
    "feat_t_",
]

# Columnas candidatas de etiqueta hard
_LABEL_COL_CANDIDATES = [
    "sent_label",
    "sentiment_label",
    "sentiment_label_teacher",
    "teacher_label",
    "label",
    "sentiment_label_annotator",
]


def _suffix_index(name: str, prefix: str) -> int:
    """Convierte el sufijo de una columna tipo x_text_12 a int(12)."""
    try:
        return int(name[len(prefix):])
    except Exception:
        return 0


def _norm_label(v: Any) -> str:
    """
    Normaliza etiquetas a {neg, neu, pos}.

    Acepta variantes en español/inglés.
    """
    if not isinstance(v, str):
        return ""
    s = v.strip().lower()
    if s in ("neg", "negative", "negativo", "negat"):
        return "neg"
    if s in ("neu", "neutral", "neutro", "neutralo"):
        return "neu"
    if s in ("pos", "positive", "positivo", "posi"):
        return "pos"
    return ""


def _auto_pick_embed_prefix(columns: List[str]) -> Optional[str]:
    """Intenta detectar automáticamente el prefijo de embeddings de texto."""
    for pr in _CANDIDATE_EMBED_PREFIXES:
        if any(c.startswith(pr) for c in columns):
            return pr
    return None


def _parse_periodo_to_sortkey(v: Any) -> Tuple[int, int]:
    """
    Convierte '2025-1' -> (2025, 1) para sorting temporal.

    Si no se puede parsear, retorna (0,0).
    """
    if v is None:
        return (0, 0)
    s = str(v).strip()
    m = re.match(r"^(\d{4})[-_/ ]?([12])$", s)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _stable_hash_index(text: str, dim: int) -> int:
    """
    Índice hash estable (no depende del hash randomizado de Python).

    Usa md5(text) y mod dim.
    """
    if dim <= 0:
        return 0
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


# =============================================================================
# Vectorizador robusto (minmax / scale_0_5)
# =============================================================================

@dataclass
class _Vectorizer:
    """
    Normalizador robusto para features numéricas.

    - ``minmax``: escala cada columna a [0,1] usando min/max observados.
    - ``scale_0_5``: asume escala original [0,5] y normaliza a [0,1].

    Maneja columnas con NaN/inf y columnas completamente NaN.
    """
    mean_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    mode: str = "minmax"

    def fit(self, X: np.ndarray, mode: str = "minmax") -> "_Vectorizer":
        if X is None or X.size == 0:
            raise ValueError("Vectorizer.fit recibió una matriz vacía.")

        self.mode = mode
        X = X.astype(np.float32, copy=False)
        X_clean = np.where(np.isfinite(X), X, np.nan)

        all_nan = np.isnan(X_clean).all(axis=0)

        X_stats = X_clean.copy()
        if all_nan.any():
            X_stats[:, all_nan] = 0.0

        self.mean_ = np.nanmean(X_stats, axis=0)

        if self.mode == "scale_0_5":
            self.min_ = np.zeros(X_stats.shape[1], dtype=np.float32)
            self.max_ = np.ones(X_stats.shape[1], dtype=np.float32) * 5.0
        else:
            self.min_ = np.nanmin(X_stats, axis=0)
            self.max_ = np.nanmax(X_stats, axis=0)

        if all_nan.any():
            self.mean_[all_nan] = 0.0
            self.min_[all_nan] = 0.0
            self.max_[all_nan] = 1.0

        denom = self.max_ - self.min_
        denom_too_small = denom < 1e-9
        if np.any(denom_too_small):
            self.max_[denom_too_small] = self.min_[denom_too_small] + 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("Vectorizer no está ajustado (llama a fit primero).")

        X = X.astype(np.float32, copy=False)
        X_clean = np.where(np.isfinite(X), X, self.mean_)
        Xs = (X_clean - self.min_) / (self.max_ - self.min_)
        Xs = np.clip(Xs, 0.0, 1.0)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)
        return Xs.astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "min":  None if self.min_  is None else self.min_.tolist(),
            "max":  None if self.max_  is None else self.max_.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "_Vectorizer":
        obj = cls()
        if not d:
            return obj
        obj.mode = d.get("mode", "minmax")
        obj.mean_ = np.array(d["mean"], dtype=np.float32) if d.get("mean") is not None else None
        obj.min_  = np.array(d["min"],  dtype=np.float32) if d.get("min")  is not None else None
        obj.max_  = np.array(d["max"],  dtype=np.float32) if d.get("max")  is not None else None
        return obj


# =============================================================================
# Núcleo RBM
# =============================================================================

class _RBM(nn.Module):
    """
    Restricted Boltzmann Machine Bernoulli-Bernoulli.

    .. note::
       La entrada visible se asume en [0,1] (por eso el Vectorizer).
    """

    def __init__(self, n_visible: int, n_hidden: int, cd_k: int = 1, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden, generator=g) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.cd_k = int(cd_k)

    def hidden_logits(self, v: Tensor) -> Tensor:
        return F.linear(v, self.W.t(), self.b_h)

    def hidden_probs(self, v: Tensor) -> Tensor:
        return torch.sigmoid(self.hidden_logits(v))

    def visible_logits(self, h: Tensor) -> Tensor:
        return F.linear(h, self.W, self.b_v)

    def sample_hidden(self, v: Tensor) -> Tensor:
        return torch.bernoulli(self.hidden_probs(v))

    def sample_visible(self, h: Tensor) -> Tensor:
        return torch.bernoulli(torch.sigmoid(self.visible_logits(h)))

    def free_energy(self, v: Tensor) -> Tensor:
        vbias_term = (v * self.b_v).sum(dim=1)
        wx_b = self.hidden_logits(v)
        hidden_term = torch.log1p(torch.exp(wx_b)).sum(dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence_step(self, v0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Ejecuta CD-k desde visibles v0.

        :return: (vk, hk) últimas muestras.
        """
        vk = v0
        for _ in range(max(1, int(self.cd_k))):
            hk = self.sample_hidden(vk)
            vk = self.sample_visible(hk)
        return vk, self.sample_hidden(vk)


# =============================================================================
# Estrategia RBMGeneral (compatible con PlantillaEntrenamiento)
# =============================================================================

class RBMGeneral:
    """
    Estrategia RBMGeneral con cabeza supervisada (3 clases).

    Flujo:
    - ``setup``: carga DF, arma features/targets, split train/val, vectoriza, inicializa RBM+head,
      preentrena RBM (unsupervised).
    - ``train_step``: entrena la head 1 época (supervised/soft labels), evalúa métricas train/val.

    Parámetros en ``hparams`` (principales)
    --------------------------------------
    - ``seed`` (int)
    - ``batch_size`` (int)
    - ``lr_rbm`` (float)
    - ``lr_head`` (float)
    - ``epochs_rbm`` (int): pretraining RBM (antes de las épocas reportadas)
    - ``n_hidden`` (int)
    - ``cd_k`` (int)
    - ``scale_mode``: ``minmax`` | ``scale_0_5``

    Split / validación (Commit 4)
    -----------------------------
    - ``split_mode``: ``temporal`` | ``random`` (default ``temporal``)
    - ``val_ratio``: float (default 0.2)

    Target (Commit 4)
    -----------------
    - ``target_mode``: ``sentiment_probs`` | ``label`` (default ``sentiment_probs``)
    """

    def __init__(
        self,
        n_visible: Optional[int] = None,
        n_hidden: Optional[int] = None,
        cd_k: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # hiperparámetros base
        self.n_visible = int(n_visible) if n_visible is not None else None
        self.n_hidden = int(n_hidden) if n_hidden is not None else None
        self.cd_k = int(cd_k) if cd_k is not None else 1
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed) if seed is not None else 42

        # hparams por defecto
        self.batch_size: int = 64
        self.lr_rbm: float = 1e-2
        self.lr_head: float = 1e-2
        self.momentum: float = 0.9
        self.weight_decay: float = 0.0
        self.epochs_rbm: int = 1
        self.scale_mode: str = "minmax"

        # commit 4: split/target
        self.split_mode: str = "temporal"
        self.val_ratio: float = 0.2
        self.target_mode: str = "sentiment_probs"

        # teacher/materia hashing
        self.include_teacher_materia: bool = True
        self.teacher_hash_dim: int = 128
        self.materia_hash_dim: int = 128

        # features/artefactos
        self.vec: _Vectorizer = _Vectorizer()
        self.rbm: Optional[_RBM] = None
        self.head: Optional[nn.Module] = None
        self.opt_rbm = None
        self.opt_head = None

        # columnas y prefijo embeddings
        self.feat_cols_: List[str] = []
        self.text_embed_prefix_: str = "x_text_"

        # datasets preparados
        self.X_train: Optional[Tensor] = None
        self.X_val: Optional[Tensor] = None
        self.y_train_hard: Optional[Tensor] = None
        self.y_val_hard: Optional[Tensor] = None
        self.y_train_soft: Optional[Tensor] = None
        self.y_val_soft: Optional[Tensor] = None

        # tracking interno
        self._epoch: int = 0
        self._last_confusion_matrix: Optional[List[List[int]]] = None

    # -------------------------------------------------------------------------
    # IO
    # -------------------------------------------------------------------------

    def _load_df(self, path: str) -> pd.DataFrame:
        """
        Carga DF desde parquet/csv/xlsx.

        :param path: ruta local al dataset.
        """
        if path is None:
            raise ValueError("data_ref is None")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        raise ValueError("Formato no soportado: " + ext)

    # -------------------------------------------------------------------------
    # Feature engineering (incluye hashing opcional teacher/materia)
    # -------------------------------------------------------------------------

    def _add_teacher_materia_hash_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega features hash one-hot para teacher_key/materia_key si aplica.

        - Crea columnas: teacher_h_0..teacher_h_(dim-1) y materia_h_0..materia_h_(dim-1)
        - No requiere conocer el vocabulario completo.
        - Es determinístico (md5).

        Si no hay columnas teacher_key/materia_key, retorna df sin cambios.
        """
        if not self.include_teacher_materia:
            return df

        out = df

        # Teacher
        if "teacher_key" in out.columns and self.teacher_hash_dim > 0:
            idxs = out["teacher_key"].astype("string").fillna("").map(
                lambda s: _stable_hash_index(str(s), self.teacher_hash_dim)
            ).to_numpy(dtype=np.int64)
            mat = np.zeros((len(out), self.teacher_hash_dim), dtype=np.float32)
            mat[np.arange(len(out)), idxs] = 1.0
            for j in range(self.teacher_hash_dim):
                out[f"teacher_h_{j}"] = mat[:, j]

        # Materia
        if "materia_key" in out.columns and self.materia_hash_dim > 0:
            idxs = out["materia_key"].astype("string").fillna("").map(
                lambda s: _stable_hash_index(str(s), self.materia_hash_dim)
            ).to_numpy(dtype=np.int64)
            mat = np.zeros((len(out), self.materia_hash_dim), dtype=np.float32)
            mat[np.arange(len(out)), idxs] = 1.0
            for j in range(self.materia_hash_dim):
                out[f"materia_h_{j}"] = mat[:, j]

        return out

    def _pick_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Selecciona columnas de features de forma robusta.

        Regla:
        - Tomar solo columnas numéricas
        - EXCLUIR targets (según target_mode)

        Embeddings:
        - Si existe un prefijo de embeddings de texto (x_text_*) y son numéricas,
          quedan incluidas automáticamente por select_dtypes; aquí solo aseguramos orden estable.
        """
        df_num = df.select_dtypes(include=[np.number]).copy()

        # Excluir probabilidades si son target_mode sentiment_probs (evita leakage)
        if self.target_mode == "sentiment_probs":
            for c in _PROB_COLS:
                if c in df_num.columns:
                    df_num.drop(columns=[c], inplace=True)

        cols = list(df_num.columns)

        calif_cols = [c for c in cols if re.match(r"^calif_\d+$", c)]
        calif_cols = sorted(calif_cols, key=lambda c: int(c.split("_")[1]))

        pregunta_cols = [c for c in cols if re.match(r"^pregunta_\d+$", c)]
        pregunta_cols = sorted(pregunta_cols, key=lambda c: int(c.split("_")[1]))

        embed_prefix = self.text_embed_prefix_
        if not any(c.startswith(embed_prefix) for c in cols):
            auto = _auto_pick_embed_prefix(cols)
            if auto:
                embed_prefix = auto
                self.text_embed_prefix_ = auto

        embed_cols = [c for c in cols if c.startswith(embed_prefix)]
        embed_cols = sorted(embed_cols, key=lambda c: _suffix_index(c, embed_prefix))

        used = set(calif_cols + pregunta_cols + embed_cols)
        rest = sorted([c for c in cols if c not in used])

        feat_cols = list(dict.fromkeys(calif_cols + pregunta_cols + embed_cols + rest))
        return feat_cols

    # -------------------------------------------------------------------------
    # Target building
    # -------------------------------------------------------------------------

    def _extract_targets(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Construye targets hard y/o soft.

        :return: (y_hard[int64] o None, y_soft[float32](n,3) o None)
        """
        # Soft labels (probabilidades)
        if self.target_mode == "sentiment_probs" and all(c in df.columns for c in _PROB_COLS):
            probs = normalize_probs(df[_PROB_COLS].to_numpy(dtype=np.float32))
            y_soft = probs.astype(np.float32, copy=False)
            y_hard = soft_to_hard(probs)
            return y_hard, y_soft

        # Hard labels (label)
        label_col = next((c for c in _LABEL_COL_CANDIDATES if c in df.columns), None)

        if label_col is None and all(c in df.columns for c in _PROB_COLS):
            # fallback: si no hay label, usar argmax(prob) solo para métricas
            probs = normalize_probs(df[_PROB_COLS].to_numpy(dtype=np.float32))
            y_hard = soft_to_hard(probs)
            return y_hard, None

        if label_col is None:
            return None, None

        y_raw = df[label_col].astype("string").fillna("").map(_norm_label)
        mask_valid = y_raw.isin(["neg", "neu", "pos"]).to_numpy()
        if mask_valid.sum() == 0:
            return None, None

        # NOTA: el filtrado por mask_valid se aplica afuera para mantener X/y alineados
        y_hard_full = np.array([_LABEL_MAP.get(s, -1) for s in y_raw.tolist()], dtype=np.int64)
        return y_hard_full, None

    # -------------------------------------------------------------------------
    # Split train/val (Commit 4)
    # -------------------------------------------------------------------------

    def _split_indices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determina índices de train/val.

        Prioridad:
        1) Columna ``split`` si existe (train/val/test).
        2) Si split_mode=temporal y existe ``periodo``: ordenar por periodo y tomar cola como val.
        3) Random con val_ratio.

        :return: (idx_train, idx_val)
        """
        n = len(df)
        if n < 2:
            idx = np.arange(n, dtype=np.int64)
            return idx, idx

        # 1) split column
        if "split" in df.columns:
            s = df["split"].astype("string").fillna("").str.lower()
            idx_train = np.where(s == "train")[0].astype(np.int64)
            idx_val = np.where(s.isin(["val", "valid", "validation"]))[0].astype(np.int64)
            if idx_train.size > 0 and idx_val.size > 0:
                return idx_train, idx_val

        # 2) temporal por periodo
        if str(self.split_mode).lower() == "temporal" and "periodo" in df.columns:
            keys = df["periodo"].map(_parse_periodo_to_sortkey).to_list()
            order = np.argsort(np.array(keys, dtype=object), kind="stable")
            n_val = max(1, int(round(n * float(self.val_ratio))))
            idx_val = order[-n_val:].astype(np.int64)
            idx_train = order[:-n_val].astype(np.int64)
            if idx_train.size == 0:
                idx_train = idx_val
            return idx_train, idx_val

        # 3) random
        rng = np.random.RandomState(self.seed)
        order = rng.permutation(n).astype(np.int64)
        n_val = max(1, int(round(n * float(self.val_ratio))))
        idx_val = order[:n_val]
        idx_train = order[n_val:]
        if idx_train.size == 0:
            idx_train = idx_val
        return idx_train, idx_val

    # -------------------------------------------------------------------------
    # Public API requerido por PlantillaEntrenamiento
    # -------------------------------------------------------------------------

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        """
        Prepara el entrenamiento.

        Pasos:
        - Leer dataset.
        - (Opcional) agregar hashing teacher/materia.
        - Seleccionar features numéricas sin leakage.
        - Construir targets hard/soft según target_mode.
        - Split train/val.
        - Vectorizar (fit SOLO en train).
        - Inicializar RBM + head.
        - Preentrenar RBM (unsupervised) con train.
        """
        hp = {str(k).lower(): v for k, v in (hparams or {}).items()}

        # seeds / device
        self.seed = int(hp.get("seed", self.seed))
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        use_cuda = bool(hp.get("use_cuda", False))
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else self.device

        # hparams
        self.batch_size = int(hp.get("batch_size", self.batch_size))
        self.cd_k = int(hp.get("cd_k", self.cd_k))
        self.lr_rbm = float(hp.get("lr_rbm", self.lr_rbm))
        self.lr_head = float(hp.get("lr_head", self.lr_head))
        self.momentum = float(hp.get("momentum", self.momentum))
        self.weight_decay = float(hp.get("weight_decay", self.weight_decay))
        self.epochs_rbm = int(hp.get("epochs_rbm", self.epochs_rbm))
        self.scale_mode = str(hp.get("scale_mode", self.scale_mode))

        # commit 4 split/target
        self.split_mode = str(hp.get("split_mode", self.split_mode)).lower()
        self.val_ratio = float(hp.get("val_ratio", self.val_ratio))
        self.target_mode = str(hp.get("target_mode", self.target_mode)).lower()

        # teacher/materia hashing
        self.include_teacher_materia = bool(hp.get("include_teacher_materia", self.include_teacher_materia))
        self.teacher_hash_dim = int(hp.get("teacher_hash_dim", self.teacher_hash_dim))
        self.materia_hash_dim = int(hp.get("materia_hash_dim", self.materia_hash_dim))

        # embeddings prefix (si aplica)
        self.text_embed_prefix_ = str(hp.get("text_embed_prefix", self.text_embed_prefix_))

        # cargar DF
        df = self._load_df(data_ref)

        # opcional: hashing keys
        df = self._add_teacher_materia_hash_features(df)

        # targets
        y_hard_full, y_soft_full = self._extract_targets(df)

        # filtrar labels inválidas si y_hard_full trae -1
        if y_hard_full is not None:
            valid = (y_hard_full >= 0) & (y_hard_full <= 2)
            if valid.sum() == 0:
                raise ValueError("No hay etiquetas válidas (neg/neu/pos) para entrenamiento.")
            if valid.sum() < len(df):
                df = df.loc[valid].reset_index(drop=True)
                y_hard_full = y_hard_full[valid]
                if y_soft_full is not None:
                    y_soft_full = y_soft_full[valid]

        # features
        feat_cols = self._pick_feature_cols(df)
        if not feat_cols:
            raise ValueError("No se detectaron columnas numéricas de features para entrenar RBM.")

        self.feat_cols_ = feat_cols
        X_full = df[feat_cols].to_numpy(dtype=np.float32)

        # split indices
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

        # targets tensors
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

        # init models
        n_visible = int(self.X_train.shape[1])
        n_hidden = int(hp.get("n_hidden", self.n_hidden or 32))

        self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
        self.head = nn.Sequential(nn.Linear(n_hidden, 3)).to(self.device)

        self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        # pretrain RBM
        self._pretrain_rbm()

        self._epoch = 0
        self._last_confusion_matrix = None

    def _pretrain_rbm(self) -> None:
        """
        Preentrena la RBM de forma no-supervisada sobre X_train.

        .. note::
           Este paso ocurre antes de las épocas reportadas al frontend.
        """
        if self.rbm is None or self.opt_rbm is None or self.X_train is None:
            raise RuntimeError("RBMGeneral no está inicializado (setup no ejecutado).")

        self.rbm.train()
        X = self.X_train

        for _ in range(max(1, int(self.epochs_rbm))):
            self.opt_rbm.zero_grad()
            vk, _ = self.rbm.contrastive_divergence_step(X)
            loss_rbm = self.rbm.free_energy(X).mean() - self.rbm.free_energy(vk).mean()
            loss_rbm.backward()
            self.opt_rbm.step()

        self.rbm.eval()

    def train_step(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """
        Ejecuta una época de entrenamiento supervisado de la cabeza (head).

        Retorna:
          - loss promedio en train (para curva)
          - métricas train/val + confusion matrix.
        """
        if self.rbm is None or self.head is None or self.opt_head is None:
            raise RuntimeError("RBMGeneral no está inicializado (setup no ejecutado).")
        if self.X_train is None or self.X_val is None:
            raise RuntimeError("X_train/X_val no están preparados.")

        self._epoch = int(epoch)

        self.rbm.eval()   # RBM fijo durante head training
        self.head.train()

        X = self.X_train
        n = int(X.shape[0])
        bs = max(1, int(self.batch_size))

        rng = np.random.RandomState(self.seed + epoch)
        order = rng.permutation(n)

        total_loss = 0.0
        total_count = 0

        for i0 in range(0, n, bs):
            idx = order[i0:i0 + bs]
            xb = X[idx]

            with torch.no_grad():
                hb = self.rbm.hidden_probs(xb)

            logits = self.head(hb)

            # target: soft o hard
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

            self.opt_head.zero_grad()
            loss.backward()
            self.opt_head.step()

            total_loss += float(loss.detach().cpu().item()) * len(idx)
            total_count += int(len(idx))

        train_loss = total_loss / max(1, total_count)

        metrics = self._evaluate(train_loss=train_loss)

        # métricas no numéricas (se preservan como "final_metrics" por la plantilla/router)
        metrics["confusion_matrix"] = self._last_confusion_matrix
        metrics["labels"] = ["neg", "neu", "pos"]

        return float(train_loss), metrics

    def _evaluate(self, train_loss: float) -> Dict[str, Any]:
        """
        Evalúa métricas en train y val.
        """
        assert self.rbm is not None and self.head is not None
        self.rbm.eval()
        self.head.eval()

        # ---- train preds ----
        with torch.no_grad():
            Htr = self.rbm.hidden_probs(self.X_train)
            logits_tr = self.head(Htr)
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

        # ---- val preds ----
        with torch.no_grad():
            Hva = self.rbm.hidden_probs(self.X_val)
            logits_va = self.head(Hva)
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
    # Predicción
    # -------------------------------------------------------------------------

    def _df_to_X(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convierte un DataFrame a matriz X respetando ``feat_cols_``.

        Si faltan columnas, se rellenan con 0.0. Si hay teacher_key/materia_key y el
        modelo fue entrenado con hashing, se generan esas columnas antes.
        """
        if not self.feat_cols_:
            raise RuntimeError("El modelo no tiene feat_cols_ configuradas (no entrenado o falta meta).")

        needs_teacher = any(c.startswith("teacher_h_") for c in self.feat_cols_)
        needs_materia = any(c.startswith("materia_h_") for c in self.feat_cols_)
        if (needs_teacher or needs_materia) and (("teacher_key" in df.columns) or ("materia_key" in df.columns)):
            df = self._add_teacher_materia_hash_features(df.copy())

        missing = [c for c in self.feat_cols_ if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0

        return df[self.feat_cols_].to_numpy(dtype=np.float32)

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades (n,3) para un DataFrame.

        :return: numpy array con orden [neg, neu, pos].
        """
        if self.rbm is None or self.head is None:
            raise RuntimeError("Modelo no cargado/entrenado.")
        X_np = self._df_to_X(df.copy())
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)

        self.rbm.eval()
        self.head.eval()
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
            proba = F.softmax(self.head(H), dim=1).cpu().numpy()
        return proba

    def predict_df(self, df: pd.DataFrame) -> List[str]:
        """Predice etiquetas string {neg,neu,pos} para un DataFrame."""
        idx = self.predict_proba_df(df).argmax(axis=1)
        return [_INV_LABEL_MAP[int(i)] for i in idx]

    def predict_proba(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Wrapper: acepta np.ndarray (ya alineado a feat_cols_) o DataFrame."""
        if isinstance(X_or_df, pd.DataFrame):
            return self.predict_proba_df(X_or_df)
        if self.rbm is None or self.head is None:
            raise RuntimeError("Modelo no cargado/entrenado.")
        X_np = np.asarray(X_or_df, dtype=np.float32)
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs).to(self.device)
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
            return F.softmax(self.head(H), dim=1).cpu().numpy()

    def predict(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """Predice etiquetas string para np.ndarray o DataFrame."""
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
        """
        if self.rbm is None or self.head is None:
            raise RuntimeError("No se puede guardar: modelo no entrenado/cargado.")

        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "vectorizer.json"), "w", encoding="utf-8") as fh:
            json.dump(self.vec.to_dict(), fh, indent=2, ensure_ascii=False)

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

        torch.save({"state_dict": self.head.state_dict()}, os.path.join(out_dir, "head.pt"))

        meta = {
            "feat_cols_": self.feat_cols_,
            "text_embed_prefix": self.text_embed_prefix_,
            "scale_mode": self.scale_mode,
            "target_mode": self.target_mode,
            "split_mode": self.split_mode,
            "val_ratio": float(self.val_ratio),
            "include_teacher_materia": bool(self.include_teacher_materia),
            "teacher_hash_dim": int(self.teacher_hash_dim),
            "materia_hash_dim": int(self.materia_hash_dim),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, in_dir: str, device: Optional[str] = None) -> "RBMGeneral":
        """
        Carga un modelo guardado con :meth:`save`.

        :param in_dir: directorio con rbm.pt/head.pt/vectorizer.json/meta.json
        :param device: 'cpu' o 'cuda' (default auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        obj = cls()
        obj.device = device

        meta_path = os.path.join(in_dir, "meta.json")
        if os.path.exists(meta_path):
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            obj.feat_cols_ = list(meta.get("feat_cols_", []))
            obj.text_embed_prefix_ = str(meta.get("text_embed_prefix", obj.text_embed_prefix_))
            obj.scale_mode = str(meta.get("scale_mode", obj.scale_mode))
            obj.target_mode = str(meta.get("target_mode", obj.target_mode))
            obj.split_mode = str(meta.get("split_mode", obj.split_mode))
            obj.val_ratio = float(meta.get("val_ratio", obj.val_ratio))
            obj.include_teacher_materia = bool(meta.get("include_teacher_materia", obj.include_teacher_materia))
            obj.teacher_hash_dim = int(meta.get("teacher_hash_dim", obj.teacher_hash_dim))
            obj.materia_hash_dim = int(meta.get("materia_hash_dim", obj.materia_hash_dim))

        vec_path = os.path.join(in_dir, "vectorizer.json")
        if os.path.exists(vec_path):
            obj.vec = _Vectorizer.from_dict(json.loads(Path(vec_path).read_text(encoding="utf-8")))

        rbm_ckpt = torch.load(os.path.join(in_dir, "rbm.pt"), map_location=device)
        obj.seed = int(rbm_ckpt.get("seed", obj.seed))
        obj.rbm = _RBM(
            n_visible=int(rbm_ckpt["n_visible"]),
            n_hidden=int(rbm_ckpt["n_hidden"]),
            cd_k=int(rbm_ckpt.get("cd_k", 1)),
            seed=int(obj.seed),
        ).to(device)
        obj.rbm.load_state_dict(rbm_ckpt["state_dict"])

        head_ckpt = torch.load(os.path.join(in_dir, "head.pt"), map_location=device)
        obj.head = nn.Sequential(nn.Linear(int(rbm_ckpt["n_hidden"]), 3)).to(device)
        obj.head.load_state_dict(head_ckpt["state_dict"])

        return obj

    # -------------------------------------------------------------------------
    # Compat: fit() (scripts legacy)
    # -------------------------------------------------------------------------

    def fit(self, data_ref: str, epochs: int = 10, hparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Entrenamiento "legacy" para scripts antiguos.

        Internamente:
        - llama a :meth:`setup`
        - corre :meth:`train_step` por `epochs`
        - retorna métricas finales
        """
        self.setup(data_ref, hparams or {})
        last_loss = 0.0
        last_metrics: Dict[str, Any] = {}
        for ep in range(1, int(epochs) + 1):
            last_loss, last_metrics = self.train_step(ep)

        out = dict(last_metrics)
        out["loss"] = float(last_loss)
        return out


# Alias histórico
ModeloRBMGeneral = RBMGeneral
