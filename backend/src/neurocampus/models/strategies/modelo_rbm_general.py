# backend/src/neurocampus/models/strategies/modelo_rbm_general.py
# Versión con fit(...) robusto y compatible con train_rbm.py / cmd_autoretrain.py
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.metrics import mae as _mae, rmse as _rmse, r2_score as _r2_score
from ..utils.metrics import accuracy as _accuracy, f1_macro as _f1_macro, confusion_matrix as _confusion_matrix


# ============================
# Mapeos y utilidades
# ============================

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {0: "neg", 1: "neu", 2: "pos"}

# FIX: default labels (evita NameError en save/meta)
_DEFAULT_LABELS = ["neg", "neu", "pos"]


# Patrón de columnas numéricas aceptadas
_NUMERIC_PATTERNS = [
    r"^calif_\d+$",     # calif_1..N
    r"^pregunta_\d+$",  # pregunta_1..N
]

# Columnas de probas del teacher
_PROB_COLS = ["p_neg", "p_neu", "p_pos"]

# Prefijos candidatos de embeddings de texto (autodetección)
_CANDIDATE_EMBED_PREFIXES = [
    "x_text_",         # por defecto del proyecto
    "text_embed_",
    "text_",
    "feat_text_",
    "feat_t_",
]

def _suffix_index(name: str, prefix: str) -> int:
    try:
        return int(name[len(prefix):])
    except Exception:
        return 0

def _norm_label(v) -> str:
    if not isinstance(v, str):
        return ""
    s = v.strip().lower()
    if s in ("neg", "negative", "negativo", "negat"): return "neg"
    if s in ("neu", "neutral", "neutro", "neutralo"): return "neu"
    if s in ("pos", "positive", "positivo", "posi"):  return "pos"
    return ""

def _matches_any(col: str, patterns: List[str]) -> bool:
    return any(re.match(p, col) for p in patterns)

def _auto_pick_embed_prefix(columns: List[str]) -> Optional[str]:
    for pr in _CANDIDATE_EMBED_PREFIXES:
        if any(c.startswith(pr) for c in columns):
            return pr
    return None


# ============================
# Vectorizador (minmax / 0..5)
# ============================

@dataclass
class _Vectorizer:
    mean_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    mode: str = "minmax"

    def fit(self, X: np.ndarray, mode: str = "minmax") -> "_Vectorizer":
        """
        Ajuste robusto: soporta columnas completamente NaN/inf
        sin propagar NaNs al RBM.
        """
        if X is None or X.size == 0:
            raise ValueError("Vectorizer.fit recibió una matriz vacía.")

        self.mode = mode

        # Aseguramos float32 y tratamos inf/-inf como NaN
        X = X.astype(np.float32, copy=False)
        X_clean = np.where(np.isfinite(X), X, np.nan)

        # Detectar columnas completamente NaN
        all_nan = np.isnan(X_clean).all(axis=0)

        # Para calcular estadísticos, reemplazamos esas columnas por 0 temporalmente
        X_stats = X_clean.copy()
        if all_nan.any():
            X_stats[:, all_nan] = 0.0

        # Estadísticos básicos sin disparar NaNs
        self.mean_ = np.nanmean(X_stats, axis=0)

        if self.mode == "scale_0_5":
            self.min_ = np.zeros(X_stats.shape[1], dtype=np.float32)
            self.max_ = np.ones(X_stats.shape[1], dtype=np.float32) * 5.0
        else:
            self.min_ = np.nanmin(X_stats, axis=0)
            self.max_ = np.nanmax(X_stats, axis=0)

        # Columnas sin información real → rango neutro [0,1], media 0
        if all_nan.any():
            self.mean_[all_nan] = 0.0
            self.min_[all_nan] = 0.0
            self.max_[all_nan] = 1.0

        # Evitar divisiones por casi 0
        denom = self.max_ - self.min_
        denom_too_small = denom < 1e-9
        if np.any(denom_too_small):
            self.max_[denom_too_small] = self.min_[denom_too_small] + 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza a [0,1] y elimina cualquier NaN/inf residual.
        """
        if self.mean_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("Vectorizer no está ajustado (llama a fit primero).")

        X = X.astype(np.float32, copy=False)

        # Reemplazar NaN/inf por la media de la columna
        X_clean = np.where(np.isfinite(X), X, self.mean_)

        Xs = (X_clean - self.min_) / (self.max_ - self.min_)

        # Forzar a [0,1]
        Xs = np.clip(Xs, 0.0, 1.0)

        # Por seguridad, eliminar cualquier residuo no finito
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)

        return Xs.astype(np.float32, copy=False)

    def to_dict(self) -> Dict:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "min":  None if self.min_  is None else self.min_.tolist(),
            "max":  None if self.max_  is None else self.max_.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "_Vectorizer":
        obj = cls()
        if not d:
            return obj
        obj.mode  = d.get("mode", "minmax")
        obj.mean_ = np.array(d["mean"], dtype=np.float32) if d.get("mean") is not None else None
        obj.min_  = np.array(d["min"],  dtype=np.float32) if d.get("min")  is not None else None
        obj.max_  = np.array(d["max"],  dtype=np.float32) if d.get("max")  is not None else None
        return obj



# ============================
# RBM
# ============================

class _RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int, cd_k: int = 1, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.W   = nn.Parameter(torch.randn(n_visible, n_hidden, generator=g) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.cd_k = int(cd_k)

    def hidden_logits(self, v: Tensor) -> Tensor:
        return F.linear(v, self.W.t(), self.b_h)  # (batch, n_hidden)

    def hidden_probs(self, v: Tensor) -> Tensor:
        return torch.sigmoid(self.hidden_logits(v))

    def visible_logits(self, h: Tensor) -> Tensor:
        return F.linear(h, self.W, self.b_v)      # (batch, n_visible)

    def sample_hidden(self, v: Tensor) -> Tensor:
        p = self.hidden_probs(v)
        return torch.bernoulli(p)

    def sample_visible(self, h: Tensor) -> Tensor:
        p = torch.sigmoid(self.visible_logits(h))
        return torch.bernoulli(p)

    def free_energy(self, v: Tensor) -> Tensor:
        vbias_term = (v * self.b_v).sum(dim=1)
        wx_b = self.hidden_logits(v)
        hidden_term = torch.log1p(torch.exp(wx_b)).sum(dim=1)
        return -vbias_term - hidden_term

    def forward(self, v: Tensor) -> Tensor:
        return self.hidden_probs(v)

    def contrastive_divergence_step(self, v0: Tensor):
        vk = v0
        for _ in range(max(1, int(self.cd_k))):
            hk = self.sample_hidden(vk)
            vk = self.sample_visible(hk)
        return vk, self.sample_hidden(vk)

# ============================
# Heads supervisados
# ============================

class _RegressionHead(nn.Module):
    """Head de regresión para score_docente (0–50) con embeddings opcionales.

    Nota: No crea un "modelo nuevo"; es solo la cabeza supervisada de la estrategia RBM.
    """

    def __init__(
        self,
        *,
        n_hidden: int,
        n_teachers: int,
        n_materias: int,
        emb_dim: int = 16,
        include_ids: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.include_ids = bool(include_ids)
        self.emb_dim = int(emb_dim)

        if self.include_ids:
            self.teacher_emb = nn.Embedding(int(n_teachers), int(emb_dim))
            self.materia_emb = nn.Embedding(int(n_materias), int(emb_dim))
            in_dim = int(n_hidden) + 2 * int(emb_dim)
        else:
            self.teacher_emb = None
            self.materia_emb = None
            in_dim = int(n_hidden)

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else None
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, h: Tensor, teacher_id: Optional[Tensor] = None, materia_id: Optional[Tensor] = None) -> Tensor:
        parts: List[Tensor] = [h]
        if self.include_ids:
            assert teacher_id is not None and materia_id is not None, "teacher_id/materia_id requeridos para include_ids=True"
            parts.append(self.teacher_emb(teacher_id))  # type: ignore[arg-type]
            parts.append(self.materia_emb(materia_id))  # type: ignore[arg-type]

        x = torch.cat(parts, dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        y = self.linear(x).squeeze(1)
        return y


# ============================
# Estrategia General
# ============================

class RBMGeneral:
    def __init__(
        self,
        n_visible: Optional[int] = None,
        n_hidden: Optional[int] = None,
        cd_k: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # parámetros base
        self.n_visible = int(n_visible) if n_visible is not None else None
        self.n_hidden  = int(n_hidden)  if n_hidden  is not None else None
        self.cd_k      = int(cd_k) if cd_k is not None else 1
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed      = int(seed) if seed is not None else 42

        # artefactos
        self.vec: _Vectorizer = _Vectorizer()
        self.rbm: Optional[_RBM] = None
        self.head: Optional[nn.Module] = None
        self.opt_rbm = None
        self.opt_head = None

        # datos tensorizados
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None

        # hparams por defecto
        self.batch_size: int = 64
        self.lr_rbm: float = 1e-2
        self.lr_head: float = 1e-2
        self.momentum: float = 0.9
        self.weight_decay: float = 0.0
        self.epochs_rbm: int = 1
        self.epochs: int = 10
        self.scale_mode: str = "minmax"

        self.feat_cols_: List[str] = []
        self.text_embed_prefix_: str = "x_text_"

        self._epoch: int = 0
        self.accept_teacher: bool = False
        self.accept_threshold: float = 0.8
        self.labels = list(_DEFAULT_LABELS)
        # --- Ruta 2 / score_docente (regresión, pair-level) ---
        self.task_type: str = "classification"  # "classification" | "regression"
        self.target_col_: Optional[str] = None

        self.include_teacher_materia_: bool = True
        self.teacher_materia_mode_: str = "embed"  # "embed" | "numeric"
        self.teacher_id_col_: str = "teacher_id"
        self.materia_id_col_: str = "materia_id"
        self.teacher_vocab_size_: int = 0
        self.materia_vocab_size_: int = 0
        self.embed_dim_: int = 16
        self.target_scale_: float = 50.0  # si se escala target a 0..1, se des-escala con este factor

        # tensores (regresión) con split
        self.X_tr: Optional[Tensor] = None
        self.y_tr: Optional[Tensor] = None
        self.tid_tr: Optional[Tensor] = None
        self.mid_tr: Optional[Tensor] = None

        self.X_va: Optional[Tensor] = None
        self.y_va: Optional[Tensor] = None
        self.tid_va: Optional[Tensor] = None
        self.mid_va: Optional[Tensor] = None

        self.w_tr: Optional[Tensor] = None
        self.w_va: Optional[Tensor] = None
        self.warm_start_info_: Dict[str, Any] = {"warm_start": "skipped"}

    # --------------------------
    # Carga de dataset sencillo
    # --------------------------
    def _load_df(self, path: str) -> pd.DataFrame:
        if path is None:
            raise ValueError("data_ref is None")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext in (".csv", ".txt"):
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError("Formato no soportado: " + ext)

    # --------------------------
    # Selección de columnas feat
    # --------------------------
    def _pick_feature_cols(
        self,
        df: pd.DataFrame,
        *,
        include_text_probs: bool,
        include_text_embeds: bool,
        text_embed_prefix: str,
        max_calif: int,
    ) -> List[str]:
        cols = list(df.columns)
        features: List[str] = []

        # 1) numéricas calif_1..N (rellenadas más abajo si faltan)
        for i in range(max_calif):
            name = f"calif_{i+1}"
            if name in cols:
                features.append(name)

        # 2) numéricas pregunta_1..N (si existen)
        features += [c for c in cols if _matches_any(c, [r"^pregunta_\d+$"])]

        # 3) probas p_neg/p_neu/p_pos
        if include_text_probs:
            for p in _PROB_COLS:
                if p in cols:
                    features.append(p)

        # 4) embeddings de texto por prefijo
        if include_text_embeds:
            embed_cols = [c for c in cols if c.startswith(text_embed_prefix)]
            if not embed_cols:
                # Autodetección si el prefijo declarado no aparece
                auto = _auto_pick_embed_prefix(cols)
                if auto:
                    self.text_embed_prefix_ = auto
                    embed_cols = [c for c in cols if c.startswith(auto)]
            if embed_cols:
                embed_cols = sorted(embed_cols, key=lambda c: _suffix_index(c, self.text_embed_prefix_))
                features += embed_cols

        # deduplicar preservando orden
        features = list(dict.fromkeys(features))
        return features

    def _prepare_xy(
        self,
        df: pd.DataFrame,
        *,
        accept_teacher: bool,
        threshold: float,
        max_calif: int,
        include_text_probs: bool,
        include_text_embeds: bool,
        text_embed_prefix: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        # asegurar calif_*
        for i in range(max_calif):
            c = f"calif_{i+1}"
            if c not in df.columns:
                df[c] = 0.0

        # construir feat_cols
        feat_cols = self._pick_feature_cols(
            df,
            include_text_probs=include_text_probs,
            include_text_embeds=include_text_embeds,
            text_embed_prefix=text_embed_prefix,
            max_calif=max_calif,
        )

        X = df[feat_cols].to_numpy(dtype=np.float32)

        # etiquetas: detectar columna candidata
        possible_label_cols = [
            "label",
            "sentiment_label_teacher",
            "sentiment_label",
            "teacher_label",
            "sentiment_label_annotator",
        ]
        label_col = next((c for c in possible_label_cols if c in df.columns), None)

        # aceptación explícita (si existe)
        accept_col = next((c for c in ("accepted_by_teacher", "teacher_accepted", "accepted") if c in df.columns), None)

        if label_col is not None:
            y_raw = df[label_col].astype("string").fillna("").str.strip().str.lower()
        else:
            # Si no hay columna de etiqueta, pero sí probas p_* -> derivar etiqueta
            if all(p in df.columns for p in _PROB_COLS):
                y_raw = (
                    df[_PROB_COLS]
                    .astype(float)
                    .idxmax(axis=1)
                    .map({"p_neg": "neg", "p_neu": "neu", "p_pos": "pos"})
                    .fillna("")
                    .astype("string")
                )
            else:
                # Sin labels ni probas: devolvemos y=None y NO filtramos a vacío
                self._periodos_last_xy_ = (
                    df["periodo"].astype("string").to_numpy()
                    if "periodo" in df.columns
                    else None
                )
                return X, None, feat_cols

        # Filtro por aceptación:
        # 1) si hay columna de aceptación -> filtrarla
        if accept_col is not None:
            try:
                mask_accept = df[accept_col].astype("float").fillna(0.0) != 0.0
                if mask_accept.sum() < len(df):
                    df = df[mask_accept].reset_index(drop=True)
                    y_raw = y_raw[mask_accept].reset_index(drop=True)
                    X = df[feat_cols].to_numpy(dtype=np.float32)
            except Exception:
                pass
        # 2) si NO hay columna, pero el llamador pide accept_teacher y existen p_* -> usar umbral
        elif accept_teacher and all(p in df.columns for p in _PROB_COLS):
            pmax = df[_PROB_COLS].to_numpy(dtype=np.float32).max(axis=1)
            mask_accept = pmax >= float(threshold)
            if mask_accept.sum() > 0 and mask_accept.sum() < len(df):
                df = df[mask_accept].reset_index(drop=True)
                y_raw = y_raw[mask_accept].reset_index(drop=True)
                X = df[feat_cols].to_numpy(dtype=np.float32)

        # normalizar etiquetas
        y_norm = y_raw.apply(_norm_label)
        mask_valid = y_norm.isin(["neg", "neu", "pos"])
        if (~mask_valid).sum() > 0:
            df = df[mask_valid].reset_index(drop=True)
            y_norm = y_norm[mask_valid].reset_index(drop=True)
            X = df[feat_cols].to_numpy(dtype=np.float32)

        y_np = None if len(y_norm) == 0 else np.array([_LABEL_MAP[s] for s in y_norm.tolist()], dtype=np.int64)
        # Guardar periodo alineado con X/y (después de TODOS los filtros)
        self._periodos_last_xy_ = (
            df["periodo"].astype("string").to_numpy()
            if "periodo" in df.columns
            else None
        )
        return X, y_np, feat_cols

    # --------------------------
    # Preparación para regresión (score_docente)
    # --------------------------
    def _period_key(self, v: Any) -> Tuple[int, int]:
        """Convierte '2025-1' -> (2025,1) para orden temporal. Si falla, retorna (0,0)."""
        try:
            s = str(v)
            parts = s.replace("_", "-").split("-")
            y = int(parts[0]) if parts and parts[0].isdigit() else 0
            sem = int(parts[1]) if len(parts) > 1 and str(parts[1]).isdigit() else 0
            return (y, sem)
        except Exception:
            return (0, 0)

    def _prepare_xy_regression(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        include_teacher_materia: bool,
        teacher_materia_mode: str,
        teacher_id_col: str,
        materia_id_col: str,
        loss_weight_col: Optional[str] = "n_par",
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' no existe en el DataFrame")

        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_raw.notna()
        if mask.sum() < len(df):
            df = df[mask].reset_index(drop=True)
            y_raw = y_raw[mask].reset_index(drop=True)

        y = y_raw.astype(np.float32).to_numpy()

        # Política: entrenar en escala 0..1 para estabilidad.
        target_scale = float(getattr(self, "target_scale_", 50.0) or 50.0)
        if np.nanmax(y) <= 1.5 and target_scale > 1.5:
            target_scale = 1.0
        self.target_scale_ = float(target_scale)
        y_scaled = (y / float(target_scale)).astype(np.float32)

        tid = None
        mid = None
        if include_teacher_materia and str(teacher_materia_mode).lower() == "embed":
            if teacher_id_col not in df.columns or materia_id_col not in df.columns:
                raise ValueError(f"teacher_id/materia_id requeridos para modo embed. Faltan columnas.")

            tid = pd.to_numeric(df[teacher_id_col], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            mid = pd.to_numeric(df[materia_id_col], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

            tmax = int(np.nanmax(tid)) if tid.size else -1
            mmax = int(np.nanmax(mid)) if mid.size else -1
            unk_t = tmax + 1
            unk_m = mmax + 1
            tid = np.where(tid < 0, unk_t, tid)
            mid = np.where(mid < 0, unk_m, mid)

            self.teacher_vocab_size_ = int(unk_t + 1)
            self.materia_vocab_size_ = int(unk_m + 1)

        # features numéricas
        drop_cols = {target_col}
        if include_teacher_materia and str(teacher_materia_mode).lower() == "embed":
            if teacher_id_col in df.columns:
                drop_cols.add(teacher_id_col)
            if materia_id_col in df.columns:
                drop_cols.add(materia_id_col)

        num_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        num_df = num_df.select_dtypes(include=[np.number]).copy()
        feat_cols = list(num_df.columns)
        if not feat_cols:
            raise ValueError("No hay columnas numéricas de features en pair_matrix (excluyendo ids/target).")

        X = num_df.to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # pesos (opcional)
        w = None
        if loss_weight_col and (loss_weight_col in df.columns):
            ww = pd.to_numeric(df[loss_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            w = np.log1p(np.clip(ww, 0.0, None)).astype(np.float32)
            if float(np.max(w)) <= 0.0:
                w = None

        return X, y_scaled.astype(np.float32), feat_cols, tid, mid, w

    def _split_train_val_indices(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        split_mode: str,
        val_ratio: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (train_idx, val_idx). Retrocompatible:
        - Preferir df=... (permite split temporal por 'periodo')
        - Si df es None, permite n=... y hace split aleatorio (no temporal)
        """
        if df is not None:
            n = int(len(df))
        elif n is not None:
            n = int(n)
        else:
            raise TypeError("Se requiere df o n para _split_train_val_indices")

        if n < 2:
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

        val_ratio = min(max(float(val_ratio), 0.0), 0.9)
        n_val = max(1, int(round(n * val_ratio)))

        idx = np.arange(n, dtype=np.int64)
        sm = str(split_mode or "").lower()

        # Temporal solo si df existe y trae 'periodo'
        if df is not None and sm == "temporal" and ("periodo" in df.columns):
            order = np.argsort(df["periodo"].apply(self._period_key).to_numpy())
            idx = idx[order]
            return idx[: n - n_val], idx[n - n_val :]

        # Fallback aleatorio
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
        return idx[: n - n_val], idx[n - n_val :]

    def _try_warm_start(self, warm_start_dir: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {"warm_start": "skipped", "warm_start_dir": str(warm_start_dir)}
        try:
            in_dir = Path(str(warm_start_dir)).expanduser().resolve()
            meta_path = in_dir / "meta.json"
            rbm_path = in_dir / "rbm.pt"
            head_path = in_dir / "head.pt"
            if not meta_path.exists() or not rbm_path.exists() or not head_path.exists():
                info["reason"] = "missing_files"
                return info

            meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            prev_task = str(meta.get("task_type") or "classification").lower()
            if prev_task != "regression":
                info["reason"] = "previous_task_not_regression"
                return info

            prev_cols = meta.get("feat_cols_") or []
            if not isinstance(prev_cols, list) or list(prev_cols) != list(self.feat_cols_):
                info["reason"] = "feature_cols_mismatch"
                return info

            rbm_ckpt = torch.load(str(rbm_path), map_location=self.device)
            head_ckpt = torch.load(str(head_path), map_location=self.device)

            self.rbm.load_state_dict(rbm_ckpt["state_dict"], strict=True)
            self.head.load_state_dict(head_ckpt["state_dict"], strict=True)

            info["warm_start"] = "ok"
            info["reason"] = None
            return info
        except Exception as e:
            return {"warm_start": "error", "error": str(e), "warm_start_dir": str(warm_start_dir)}


    # --------------------------
    # setup() opcional (no usado por fit robusto, pero disponible)
    # --------------------------
    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        # Mantener compatibilidad si tu runner usa setup()
        self.seed = int(hparams.get("seed", self.seed or 42) or 42)
        np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() and bool(hparams.get("use_cuda", False)) else self.device

        self.batch_size    = int(hparams.get("batch_size", self.batch_size))
        self.cd_k          = int(hparams.get("cd_k", getattr(self, "cd_k", 1)))
        self.lr_rbm        = float(hparams.get("lr_rbm", self.lr_rbm))
        self.lr_head       = float(hparams.get("lr_head", self.lr_head))
        self.momentum      = float(hparams.get("momentum", self.momentum))
        self.weight_decay  = float(hparams.get("weight_decay", self.weight_decay))
        self.epochs_rbm    = int(hparams.get("epochs_rbm", self.epochs_rbm))
        self.epochs        = int(hparams.get("epochs", self.epochs))
        self.scale_mode    = str(hparams.get("scale_mode", self.scale_mode))

        # --- Detectar tipo de tarea (clasificación vs regresión) ---
        self.task_type = str(hparams.get("task_type", self.task_type or "classification") or "classification").lower()
        if self.task_type not in ("classification", "regression"):
            self.task_type = "classification"

        if self.task_type == "regression":
            # score_docente (pair-level)
            self.target_col_ = str(hparams.get("target_col") or hparams.get("score_col") or "target_score")
            self.include_teacher_materia_ = bool(hparams.get("include_teacher_materia", self.include_teacher_materia_))
            self.teacher_materia_mode_ = str(hparams.get("teacher_materia_mode", self.teacher_materia_mode_))
            self.teacher_id_col_ = str(hparams.get("teacher_id_col", self.teacher_id_col_))
            self.materia_id_col_ = str(hparams.get("materia_id_col", self.materia_id_col_))
            self.embed_dim_ = int(hparams.get("embed_dim", hparams.get("tm_emb_dim", self.embed_dim_)))

            split_mode = str(hparams.get("split_mode", "temporal")).lower()
            val_ratio = float(hparams.get("val_ratio", 0.2))

            if data_ref:
                df = self._load_df(data_ref)
            else:
                # Dummy para tests/manual (no usado en producción)
                n = 256
                df = pd.DataFrame({
                    "periodo": ["2025-1"] * n,
                    self.teacher_id_col_: np.random.randint(0, 20, size=n),
                    self.materia_id_col_: np.random.randint(0, 30, size=n),
                    self.target_col_: (np.random.rand(n).astype(np.float32) * 50.0),
                })
                for i in range(10):
                    df[f"calif_{i+1}"] = (np.random.rand(n).astype(np.float32) * 5.0)

            X_np, y_np, feat_cols, tid_np, mid_np, w_np = self._prepare_xy_regression(
                df.copy(),
                target_col=self.target_col_,
                include_teacher_materia=self.include_teacher_materia_,
                teacher_materia_mode=self.teacher_materia_mode_,
                teacher_id_col=self.teacher_id_col_,
                materia_id_col=self.materia_id_col_,
                loss_weight_col=str(hparams.get("loss_weight_col", "n_par")) if hparams.get("loss_weight_col", "n_par") else None,
            )

            # Reconstruir df filtrado (mismo criterio que _prepare_xy_regression)
            y_raw = pd.to_numeric(df[self.target_col_], errors="coerce")
            df_filt = df[y_raw.notna()].reset_index(drop=True)

            tr_idx, va_idx = self._split_train_val_indices(
                df_filt,
                split_mode=split_mode,
                val_ratio=val_ratio,
                seed=self.seed,
            )
            tr_t = torch.from_numpy(tr_idx).to(self.device)
            va_t = torch.from_numpy(va_idx).to(self.device)

            self.feat_cols_ = list(feat_cols)
            self.vec = _Vectorizer().fit(X_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
            X_np = self.vec.transform(X_np)

            X_all = torch.from_numpy(X_np).to(self.device)
            y_all = torch.from_numpy(y_np.astype(np.float32, copy=False)).to(self.device)

            self.X_tr = X_all[tr_t]
            self.y_tr = y_all[tr_t]
            self.X_va = X_all[va_t]
            self.y_va = y_all[va_t]

            if tid_np is not None and mid_np is not None:
                tid_all = torch.from_numpy(tid_np.astype(np.int64, copy=False)).to(self.device)
                mid_all = torch.from_numpy(mid_np.astype(np.int64, copy=False)).to(self.device)
                self.tid_tr = tid_all[tr_t]
                self.mid_tr = mid_all[tr_t]
                self.tid_va = tid_all[va_t]
                self.mid_va = mid_all[va_t]
            else:
                self.tid_tr = self.mid_tr = self.tid_va = self.mid_va = None

            if w_np is not None:
                w_all = torch.from_numpy(w_np.astype(np.float32, copy=False)).to(self.device)
                self.w_tr = w_all[tr_t]
                self.w_va = w_all[va_t]
            else:
                self.w_tr = self.w_va = None

            # compat con _iter_minibatches (usa self.X/self.y para train)
            self.X = self.X_tr
            self.y = self.y_tr

            n_visible = int(self.X_tr.shape[1])
            n_hidden = int(hparams.get("n_hidden", self.n_hidden or 32))
            self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
            self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)

            include_ids = bool(self.include_teacher_materia_ and str(self.teacher_materia_mode_).lower() == "embed")
            self.head = _RegressionHead(
                n_hidden=n_hidden,
                n_teachers=max(1, int(self.teacher_vocab_size_)),
                n_materias=max(1, int(self.materia_vocab_size_)),
                emb_dim=int(self.embed_dim_),
                include_ids=include_ids,
                dropout=float(hparams.get("dropout", 0.0)),
            ).to(self.device)
            self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

            warm_dir = str(hparams.get("warm_start_path") or "")
            self.warm_start_info_ = {"warm_start": "skipped", "warm_start_dir": warm_dir}
            if warm_dir:
                try:
                    self.warm_start_info_ = self._try_warm_start(warm_dir)
                except Exception:
                    self.warm_start_info_ = {"warm_start": "skipped", "warm_start_dir": warm_dir, "reason": "exception"}

        else:
            include_text_probs = bool(hparams.get("use_text_probs", False))
            include_text_embeds = bool(hparams.get("use_text_embeds", False))
            self.text_embed_prefix_ = str(hparams.get("text_embed_prefix", self.text_embed_prefix_))
            max_calif = int(hparams.get("max_calif", 10))

            df = self._load_df(data_ref) if data_ref else pd.DataFrame({f"calif_{i+1}": np.random.rand(256).astype(np.float32) * 5.0 for i in range(max_calif)})

            X_np, y_np, feat_cols = self._prepare_xy(
                df,
                accept_teacher=bool(hparams.get("accept_teacher", False)),
                threshold=float(hparams.get("accept_threshold", 0.8)),
                max_calif=max_calif,
                include_text_probs=include_text_probs,
                include_text_embeds=include_text_embeds or any(c.startswith(self.text_embed_prefix_) for c in df.columns),
                text_embed_prefix=self.text_embed_prefix_,
            )

            self.feat_cols_ = list(feat_cols)
            self.vec = _Vectorizer().fit(X_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
            X_np = self.vec.transform(X_np)

            X_t = torch.from_numpy(X_np).to(self.device)
            self.X = X_t

            n_visible = X_np.shape[1]
            n_hidden = int(hparams.get("n_hidden", self.n_hidden or 32))
            self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
            self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)

            self.head = nn.Sequential(nn.Linear(n_hidden, 3)).to(self.device)
            self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

            self.y = torch.from_numpy(y_np).to(self.device) if y_np is not None else None

            # split train/val para métricas de clasificación (ALINEADO con X_np filtrado)
            split_mode = str(hparams.get("split_mode", "random")).lower()
            val_ratio = float(hparams.get("val_ratio", 0.2))

            n_rows = int(X_t.shape[0])  # <- tamaño real de X (ya filtrado por _prepare_xy)
            periodos = getattr(self, "_periodos_last_xy_", None)

            if y_np is not None and len(y_np) > 4 and n_rows == int(len(y_np)):
                # Si pidieron temporal y tenemos periodos alineados, hacemos split temporal.
                if split_mode == "temporal" and periodos is not None and len(periodos) == n_rows:
                    df_split = pd.DataFrame({"periodo": periodos})
                    tr_idx, va_idx = self._split_train_val_indices(
                        df=df_split,
                        split_mode="temporal",
                        val_ratio=val_ratio,
                        seed=self.seed,
                    )
                else:
                    # Para no inventar temporal sin 'periodo', hacemos random si split_mode=temporal sin periodos.
                    sm = "random" if split_mode == "temporal" else split_mode
                    tr_idx, va_idx = self._split_train_val_indices(
                        n=n_rows,
                        split_mode=sm,
                        val_ratio=val_ratio,
                        seed=self.seed,
                    )

                # Guardas defensivas para evitar out-of-bounds silencioso
                if (tr_idx.max(initial=-1) >= n_rows) or (va_idx.max(initial=-1) >= n_rows):
                    raise RuntimeError(
                        f"Split fuera de rango: n_rows={n_rows}, tr_max={tr_idx.max(initial=-1)}, va_max={va_idx.max(initial=-1)}"
                    )

                self.X_tr = X_t[tr_idx]
                self.y_tr = torch.from_numpy(y_np[tr_idx]).to(self.device)
                self.X_va = X_t[va_idx]
                self.y_va = torch.from_numpy(y_np[va_idx]).to(self.device)
                self.X = self.X_tr   # entrenamiento solo en train split
                self.y = self.y_tr
            else:
                self.X_tr = X_t
                self.y_tr = self.y
                self.X_va = self.y_va = None

            # limpiar slots de regresión (ids/pesos — no aplican a clasificación)
            self.tid_tr = self.mid_tr = None
            self.tid_va = self.mid_va = None
            self.w_tr = self.w_va = None

        self._epoch = 0

    # ---------- Mini-batches ----------
    def _iter_minibatches(self, X: Tensor, y: Optional[Tensor]):
        idx = torch.randperm(X.shape[0], device=X.device)
        for start in range(0, len(idx), int(self.batch_size)):
            sel = idx[start:start + int(self.batch_size)]
            yield X[sel], (None if y is None else y[sel])

    # ---------- Entrenamiento (compatible con PlantillaEntrenamiento) ----------
    def train_step(self, epoch: int, hparams: Optional[Dict] = None, y: any = None):
        """
        Ejecuta 1 época de entrenamiento.

        Compatible con PlantillaEntrenamiento, que intenta llamar:
          - train_step(epoch, hparams, y=y)
          - train_step(epoch, hparams)
          - train_step(epoch)
        """
        assert self.rbm is not None and self.opt_rbm is not None and self.X is not None, (
            "RBMGeneral no está inicializado. Asegúrate de que setup(data_ref, hparams) se ejecutó."
        )
        assert self.head is not None and self.opt_head is not None, (
            "RBMGeneral.head/opt_head no inicializados. Revisa setup()."
        )

        self._epoch = int(epoch)
        t0 = time.perf_counter()

        rbm_losses: List[float] = []
        rbm_grad_norms: List[float] = []
        cls_losses: List[float] = []

        # --- RBM update ---
        self.rbm.train()
        for _ in range(max(1, int(getattr(self, "epochs_rbm", 1)))):
            for xb, _ in self._iter_minibatches(self.X, self.y):
                self.opt_rbm.zero_grad(set_to_none=True)

                vk, _hk = self.rbm.contrastive_divergence_step(xb)
                loss_rbm = self.rbm.free_energy(xb).mean() - self.rbm.free_energy(vk).mean()
                loss_rbm.backward()

                # grad norm (opcional, útil para UI/debug)
                with torch.no_grad():
                    sq = 0.0
                    for p in self.rbm.parameters():
                        if p.grad is not None:
                            sq += float((p.grad.detach() ** 2).sum().cpu())
                    rbm_grad_norms.append(float(sq ** 0.5))

                self.opt_rbm.step()

                # recon_error “amigable” (MSE entre v0 y vk)
                with torch.no_grad():
                    recon = torch.mean((vk - xb) ** 2).detach().cpu().item()
                rbm_losses.append(float(recon))

        # --- Head supervised (clasificación) / regresión (score_docente) ---
        reg_losses: List[float] = []

        if str(getattr(self, "task_type", "classification")).lower() == "regression":
            if self.X_tr is None or self.y_tr is None:
                raise RuntimeError("RBMGeneral.setup() no inicializó X_tr/y_tr para regresión")

            self.head.train()
            for b in range(0, int(self.X_tr.shape[0]), int(self.batch_size)):
                xb = self.X_tr[b : b + int(self.batch_size)]
                yb = self.y_tr[b : b + int(self.batch_size)]
                tidb = self.tid_tr[b : b + int(self.batch_size)] if self.tid_tr is not None else None
                midb = self.mid_tr[b : b + int(self.batch_size)] if self.mid_tr is not None else None
                wb = self.w_tr[b : b + int(self.batch_size)] if self.w_tr is not None else None

                with torch.no_grad():
                    H = self.rbm.hidden_probs(xb)

                self.opt_head.zero_grad(set_to_none=True)
                if isinstance(self.head, _RegressionHead) and self.head.include_ids:
                    pred = self.head(H, tidb, midb)
                else:
                    pred = self.head(H)
                    if pred.ndim > 1:
                        pred = pred.squeeze(-1)

                loss_vec = (pred - yb) ** 2
                if wb is not None:
                    loss_reg = (loss_vec * wb).sum() / (wb.sum() + 1e-8)
                else:
                    loss_reg = loss_vec.mean()

                loss_reg.backward()
                self.opt_head.step()
                reg_losses.append(float(loss_reg.detach().cpu()))

        elif self.y is not None:
            self.head.train()
            for xb, yb in self._iter_minibatches(self.X, self.y):
                with torch.no_grad():
                    H = self.rbm.hidden_probs(xb)
                self.opt_head.zero_grad(set_to_none=True)
                logits = self.head(H)
                loss_cls = F.cross_entropy(logits, yb, ignore_index=3)
                loss_cls.backward()
                self.opt_head.step()
                cls_losses.append(float(loss_cls.detach().cpu()))


        time_epoch_ms = float((time.perf_counter() - t0) * 1000.0)

        metrics = {
            "epoch": float(epoch),
            "recon_error": float(np.mean(rbm_losses)) if rbm_losses else 0.0,
            "rbm_grad_norm": float(np.mean(rbm_grad_norms)) if rbm_grad_norms else 0.0,
            "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "reg_loss": float(np.mean(reg_losses)) if reg_losses else 0.0,
            "time_epoch_ms": time_epoch_ms,
        }

        if str(getattr(self, "task_type", "classification")).lower() == "regression" and self.X_tr is not None and self.y_tr is not None:
            self.rbm.eval()
            self.head.eval()

            scale = float(getattr(self, "target_scale_", 1.0) or 1.0)

            def _pred(x: Tensor, tid: Optional[Tensor], mid: Optional[Tensor]) -> np.ndarray:
                with torch.no_grad():
                    h = self.rbm.hidden_probs(x)
                    if isinstance(self.head, _RegressionHead) and self.head.include_ids:
                        p = self.head(h, tid, mid)
                    else:
                        p = self.head(h)
                        if p.ndim > 1:
                            p = p.squeeze(-1)
                return (p.detach().cpu().numpy() * scale).astype(np.float32)

            y_tr_np = (self.y_tr.detach().cpu().numpy() * scale).astype(np.float32)
            p_tr_np = _pred(self.X_tr, self.tid_tr, self.mid_tr)

            metrics.update({
                "task_type": "regression",
                "target_col": getattr(self, "target_col_", None),
                "train_mae": float(_mae(y_tr_np, p_tr_np)),
                "train_rmse": float(_rmse(y_tr_np, p_tr_np)),
                "train_r2": float(_r2_score(y_tr_np, p_tr_np)),
                "n_train": int(self.X_tr.shape[0]),
            })

            if self.X_va is not None and self.y_va is not None:
                y_va_np = (self.y_va.detach().cpu().numpy() * scale).astype(np.float32)
                p_va_np = _pred(self.X_va, self.tid_va, self.mid_va)
                metrics.update({
                    "val_mae": float(_mae(y_va_np, p_va_np)),
                    "val_rmse": float(_rmse(y_va_np, p_va_np)),
                    "val_r2": float(_r2_score(y_va_np, p_va_np)),
                    "n_val": int(self.X_va.shape[0]),
                    "pred_min": float(np.min(p_va_np)) if p_va_np.size else None,
                    "pred_max": float(np.max(p_va_np)) if p_va_np.size else None,
                })

            if hasattr(self, "warm_start_info_") and isinstance(self.warm_start_info_, dict):
                metrics["warm_start"] = dict(self.warm_start_info_)

            metrics["loss"] = metrics["recon_error"] + metrics["reg_loss"]
        else:
            # --- Evaluación clasificación: accuracy/f1_macro en train y val ---
            self.rbm.eval()
            self.head.eval()
            n_classes = 3 if not hasattr(self.head, "out_features") else int(getattr(self.head[-1] if hasattr(self.head, "__getitem__") else self.head, "out_features", 3))
            labels_list = list(getattr(self, "labels", list(range(n_classes))))

            def _cls_eval(Xb, yb):
                with torch.no_grad():
                    H = self.rbm.hidden_probs(Xb)
                    logits = self.head(H)
                    preds = int(logits.shape[-1]) and logits.argmax(dim=-1)
                y_true_np = yb.detach().cpu().numpy().astype(int)
                y_pred_np = preds.detach().cpu().numpy().astype(int)
                return y_true_np, y_pred_np

            y_tr_true, y_tr_pred = _cls_eval(self.X_tr, self.y_tr) if (self.X_tr is not None and self.y_tr is not None) else (None, None)
            y_va_true, y_va_pred = _cls_eval(self.X_va, self.y_va) if (self.X_va is not None and self.y_va is not None) else (None, None)

            if y_tr_true is not None:
                metrics.update({
                    "task_type": "classification",
                    "labels": labels_list,
                    "n_classes": n_classes,
                    "n_train": int(len(y_tr_true)),
                    "accuracy": float(_accuracy(y_tr_true, y_tr_pred)),
                    "f1_macro": float(_f1_macro(y_tr_true, y_tr_pred, n_classes)),
                })
            if y_va_true is not None:
                metrics.update({
                    "n_val": int(len(y_va_true)),
                    "val_accuracy": float(_accuracy(y_va_true, y_va_pred)),
                    "val_f1_macro": float(_f1_macro(y_va_true, y_va_pred, n_classes)),
                    "confusion_matrix": _confusion_matrix(y_va_true, y_va_pred, n_classes),
                })
            self.rbm.train()
            self.head.train()
            metrics["loss"] = metrics["recon_error"] + metrics["cls_loss"]

        return metrics



    # --------------------------
    # Transformaciones y predict
    # --------------------------
    def _transform_np(self, X_np: np.ndarray) -> Tensor:
        Xs = self.vec.transform(X_np)
        Xt = torch.from_numpy(Xs.astype(np.float32, copy=False)).to(self.device)
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
        return H

    def _df_to_X(self, df: pd.DataFrame) -> np.ndarray:
        assert len(self.feat_cols_) > 0, "El modelo no tiene feat_cols_ configuradas."
        missing = [c for c in self.feat_cols_ if c not in df.columns]
        if missing:
            # rellenar con 0.0 si faltan columnas (tolerancia)
            for c in missing:
                df[c] = 0.0
        X_np = df[self.feat_cols_].to_numpy(dtype=np.float32)
        return X_np

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        X_np = self._df_to_X(df.copy())
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            proba = F.softmax(self.head(H), dim=1).cpu().numpy()
        return proba

    def predict_df(self, df: pd.DataFrame) -> List[str]:
        idx = self.predict_proba_df(df).argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    def predict_proba(self, X_or_df: Union[np.ndarray, pd.DataFrame], X_text_embeds: Optional[np.ndarray] = None) -> np.ndarray:
        if isinstance(X_or_df, pd.DataFrame):
            df = X_or_df.copy()
            if X_text_embeds is not None:
                X_text_embeds = np.asarray(X_text_embeds, dtype=np.float32)
                if X_text_embeds.shape[0] != len(df):
                    raise ValueError("X_text_embeds must have same number of rows as DataFrame")
                n_text = X_text_embeds.shape[1]
                for j in range(n_text):
                    df[f"{self.text_embed_prefix_}{j}"] = X_text_embeds[:, j]
            return self.predict_proba_df(df)

        X_np = np.asarray(X_or_df, dtype=np.float32)
        if X_text_embeds is not None:
            X_text_embeds = np.asarray(X_text_embeds, dtype=np.float32)
            if X_text_embeds.shape[0] != X_np.shape[0]:
                raise ValueError("X_text_embeds must have same number of rows as X_or_df")
            X_np = np.hstack([X_np, X_text_embeds])

        assert X_np.shape[1] == len(self.feat_cols_), (
            f"Dimensión de entrada {X_np.shape[1]} != {len(self.feat_cols_)} (entrenamiento). "
            "Usa predict_proba_df(df) para construir automáticamente las columnas o pasa embeddings adecuados."
        )
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            return F.softmax(self.head(H), dim=1).cpu().numpy()

    def predict(self, X_or_df: Union[np.ndarray, pd.DataFrame], X_text_embeds: Optional[np.ndarray] = None) -> List[str]:
        proba = self.predict_proba(X_or_df, X_text_embeds)
        idx = proba.argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    # --------------------------
    # Persistencia
    # --------------------------
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # vectorizer.json
        with open(os.path.join(out_dir, "vectorizer.json"), "w", encoding="utf-8") as fh:
            json.dump(self.vec.to_dict(), fh, indent=2)
        # rbm/head
        torch.save(
            {"state_dict": self.rbm.state_dict(), "n_visible": self.rbm.W.shape[0], "n_hidden": self.rbm.W.shape[1], "cd_k": self.rbm.cd_k},
            os.path.join(out_dir, "rbm.pt"),
        )
        torch.save({"state_dict": self.head.state_dict()}, os.path.join(out_dir, "head.pt"))
        # meta.json (incluye vectorizer inline para mayor robustez)
        meta = {
            "feat_cols_": self.feat_cols_,
            "vectorizer": self.vec.to_dict(),
            "labels": getattr(self, "labels", _DEFAULT_LABELS),
            "hparams": {
                "scale_mode": self.scale_mode,
                "text_embed_prefix": self.text_embed_prefix_,
                "cd_k": int(getattr(self.rbm, "cd_k", self.cd_k)),
            },
            "task_type": str(getattr(self, "task_type", "classification")).lower(),
            "target_col": getattr(self, "target_col_", None),
        }
        if meta["task_type"] == "regression":
            meta.update({
                "target_scale": float(getattr(self, "target_scale_", 1.0) or 1.0),
                "include_teacher_materia": bool(getattr(self, "include_teacher_materia_", False)),
                "teacher_materia_mode": str(getattr(self, "teacher_materia_mode_", "")),
                "teacher_id_col": str(getattr(self, "teacher_id_col_", "teacher_id")),
                "materia_id_col": str(getattr(self, "materia_id_col_", "materia_id")),
                "teacher_vocab_size_": int(getattr(self, "teacher_vocab_size_", 0) or 0),
                "materia_vocab_size_": int(getattr(self, "materia_vocab_size_", 0) or 0),
                "embed_dim_": int(getattr(self, "embed_dim_", 16) or 16),
            })

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, in_dir: str, device: Optional[str] = None) -> "RBMGeneral":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        obj = cls()
        obj.device = device

        # meta y vectorizer
        meta_path = os.path.join(in_dir, "meta.json")
        vec_path  = os.path.join(in_dir, "vectorizer.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            obj.labels = list(meta.get("labels", _DEFAULT_LABELS))
            obj.feat_cols_ = list(meta.get("feat_cols_", []))
            obj.feat_cols_ = list(meta.get("feat_cols_", []))
            obj.vec = _Vectorizer.from_dict(meta.get("vectorizer", None))
            obj.scale_mode = str(meta.get("hparams", {}).get("scale_mode", obj.scale_mode))
            obj.text_embed_prefix_ = str(meta.get("hparams", {}).get("text_embed_prefix", obj.text_embed_prefix_))

        # Si vectorizer.json existe, tiene prioridad (por si meta antiguo no lo tenía)
        if os.path.exists(vec_path):
            with open(vec_path, "r", encoding="utf-8") as fh:
                obj.vec = _Vectorizer.from_dict(json.load(fh))

        # fallback feat_cols por si falta meta
        if not obj.feat_cols_:
            obj.feat_cols_ = [f"calif_{i+1}" for i in range(10)]

        # cargar rbm/head
        rbm_ckpt = torch.load(os.path.join(in_dir, "rbm.pt"), map_location=device)
        obj.rbm = _RBM(n_visible=rbm_ckpt["n_visible"], n_hidden=rbm_ckpt["n_hidden"], cd_k=rbm_ckpt.get("cd_k", 1)).to(device)
        obj.rbm.load_state_dict(rbm_ckpt["state_dict"])
        head_ckpt = torch.load(os.path.join(in_dir, "head.pt"), map_location=device)
        task_type = str(meta.get("task_type", "classification")).lower()
        obj.task_type = task_type
        obj.target_col_ = meta.get("target_col")

        if task_type == "regression":
            obj.target_scale_ = float(meta.get("target_scale", 1.0) or 1.0)
            obj.include_teacher_materia_ = bool(meta.get("include_teacher_materia", False))
            obj.teacher_materia_mode_ = str(meta.get("teacher_materia_mode", ""))
            obj.teacher_id_col_ = str(meta.get("teacher_id_col", obj.teacher_id_col_))
            obj.materia_id_col_ = str(meta.get("materia_id_col", obj.materia_id_col_))
            obj.teacher_vocab_size_ = int(meta.get("teacher_vocab_size_", 0) or 0)
            obj.materia_vocab_size_ = int(meta.get("materia_vocab_size_", 0) or 0)
            obj.embed_dim_ = int(meta.get("embed_dim_", obj.embed_dim_) or obj.embed_dim_)

            include_ids = bool(obj.include_teacher_materia_ and obj.teacher_materia_mode_.lower() == "embed")
            obj.head = _RegressionHead(
                n_hidden=int(rbm_ckpt["n_hidden"]),
                n_teachers=max(1, int(obj.teacher_vocab_size_)),
                n_materias=max(1, int(obj.materia_vocab_size_)),
                emb_dim=int(obj.embed_dim_),
                include_ids=include_ids,
            ).to(device)
        else:
            obj.head = nn.Sequential(nn.Linear(int(rbm_ckpt["n_hidden"]), 3)).to(device)

        obj.head.load_state_dict(head_ckpt["state_dict"])


        obj.X = None; obj.y = None; obj._epoch = 0
        return obj

    def fit(self, *args, **kwargs) -> Dict:
        """
        Soporta dos modos:
        A) fit(X_df_o_np, y_np, ...)  -> usado por train_rbm.py
        B) fit(df_completo, ...)      -> autodetecta labels/feats desde el DF (modo antiguo)
        """
        import numpy as _np
        import pandas as _pd
        job_dir = kwargs.get("job_dir") or kwargs.get("out_dir") or kwargs.get("job_dir_path")
        if job_dir is None:
            job_dir = os.path.join("artifacts", "jobs", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(job_dir, exist_ok=True)

        # ------ hparams ------
        get = lambda k, d=None: kwargs.get(k, d)
        self.scale_mode   = str(get("scale_mode", self.scale_mode))
        self.lr_rbm       = float(get("lr_rbm", self.lr_rbm))
        self.lr_head      = float(get("lr_head", self.lr_head))
        self.momentum     = float(get("momentum", self.momentum))
        self.weight_decay = float(get("weight_decay", self.weight_decay))
        self.epochs_rbm   = int(get("epochs_rbm", self.epochs_rbm))
        self.epochs       = int(get("epochs", self.epochs))
        self.cd_k         = int(get("cd_k", self.cd_k))
        self.seed         = int(get("seed", self.seed or 42)); np.random.seed(self.seed); torch.manual_seed(self.seed)

        # ------ Modo A: (X_df|X_np, y) ------
        X_np = None; y_np = None
        if len(args) >= 1 and isinstance(args[0], (_pd.DataFrame, _np.ndarray)):
            Xarg = args[0]

            if isinstance(Xarg, _pd.DataFrame):
                # ------------------------------------------------------------------
                # Caso: fit recibe un DataFrame (modo A)
                #
                # En algunos usos (tests, scripts antiguos) se pasa un DataFrame
                # que incluye tanto las columnas de características numéricas
                # (calif_1..calif_10, etc.) como la etiqueta de salida
                # `sentiment_label_teacher` (string).
                #
                # Para evitar errores del tipo "could not convert string to float",
                # seleccionamos únicamente las columnas numéricas para alimentar
                # al núcleo RBM, manteniendo compatibilidad con:
                #   - Pipelines modernos (que ya pasan X numérico + y aparte).
                #   - Tests que envían el DF completo con la columna target.
                # ------------------------------------------------------------------
                self.X_raw = Xarg  # útil para depuración/documentación

                Xnum = Xarg.select_dtypes(include=[_np.number])
                if Xnum.shape[1] == 0:
                    raise ValueError(
                        "RBMGeneral.fit recibió un DataFrame sin columnas numéricas; "
                        "verifique que las columnas calif_* existan y sean numéricas."
                    )

                self.feat_cols_ = list(Xnum.columns)
                X_np = Xnum.to_numpy(dtype=_np.float32)
            else:
                # Caso: ya pasan directamente una matriz/array numérica
                X_np = _np.asarray(Xarg, dtype=_np.float32)
                if not self.feat_cols_:
                    self.feat_cols_ = [f"f{i}" for i in range(X_np.shape[1])]

            if len(args) >= 2 and args[1] is not None:
                y_np = _np.asarray(args[1], dtype=_np.int64)
        else:
            # ------ Modo B: DF completo (autodetección clásica) ------
            data_ref = get("data") or get("data_ref") or get("dataset")
            df = self._load_df(data_ref) if isinstance(data_ref, str) else (args[0] if len(args)>=1 else None)
            if df is None or not isinstance(df, _pd.DataFrame):
                raise ValueError("fit(...) requiere (X,y) o un DataFrame completo.")
            include_text_embeds = bool(get("use_text_embeds", False))
            include_text_probs  = bool(get("use_text_probs", False))
            self.text_embed_prefix_ = str(get("text_embed_prefix", self.text_embed_prefix_))
            max_calif = int(get("max_calif", 10))
            X_np, y_np, feat_cols = self._prepare_xy(
                df.copy(),
                accept_teacher=bool(get("accept_teacher", False)),
                threshold=float(get("accept_threshold", 0.8)),
                max_calif=max_calif,
                include_text_probs=include_text_probs,
                include_text_embeds=include_text_embeds or any(c.startswith(self.text_embed_prefix_) for c in df.columns),
                text_embed_prefix=self.text_embed_prefix_,
            )
            self.feat_cols_ = list(feat_cols)

        # ------- Filtrado automático de clases inválidas (y fuera de {0,1,2}) -------
        if y_np is not None:
            valid_mask = (y_np >= 0) & (y_np <= 2)
            if valid_mask.sum() < len(y_np):
                X_np = X_np[valid_mask]
                y_np = y_np[valid_mask]
        # ------- chequeos -------
        if X_np is None or X_np.size == 0:
            raise ValueError("X de entrenamiento está vacío; revisa el pipeline de features.")
        self.vec = _Vectorizer().fit(X_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
        Xs = self.vec.transform(X_np); self.X = torch.from_numpy(Xs).to(self.device)
        n_hidden = int(get("n_hidden", self.n_hidden or 32))
        self.rbm  = _RBM(n_visible=self.X.shape[1], n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
        self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)
        self.head = nn.Sequential(nn.Linear(n_hidden, 3)).to(self.device)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        # ------- pretrain RBM -------
        self.rbm.train()
        for _ in range(max(1, self.epochs_rbm)):
            self.opt_rbm.zero_grad()
            vk, hk = self.rbm.contrastive_divergence_step(self.X)
            loss_rbm = self.rbm.free_energy(self.X).mean() - self.rbm.free_energy(vk).mean()
            loss_rbm.backward(); self.opt_rbm.step()

        # ------- head supervised (si y) -------
        if y_np is None:
            f1_macro, acc = 0.0, 0.0
        else:
            self.y = torch.from_numpy(y_np).to(self.device)
            for _ in range(max(1, self.epochs)):
                self.opt_head.zero_grad()
                with torch.no_grad(): H = self.rbm.hidden_probs(self.X)
                logits = self.head(H)
                # Ignora índice 3 si aún quedara alguno por arriba (paranoia-safe)
                loss = F.cross_entropy(logits, self.y, ignore_index=3)
                loss.backward(); self.opt_head.step()
            self.rbm.eval(); self.head.eval()
            with torch.no_grad():
                H = self.rbm.hidden_probs(self.X)
                preds = torch.argmax(self.head(H), dim=1).cpu().numpy()
                y_true = self.y.cpu().numpy()
            acc = float((preds == y_true).mean())
            # f1 macro simple (sin sklearn)
            f1s = []
            for c in [0,1,2]:
                tp = int(((preds==c)&(y_true==c)).sum()); fp = int(((preds==c)&(y_true!=c)).sum()); fn = int(((preds!=c)&(y_true==c)).sum())
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0; rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1s.append(0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec))
            f1_macro = float(np.mean(f1s))

        # ------- persistencia mínima -------
        try:
            torch.save({"state_dict": self.rbm.state_dict(), "n_visible": self.rbm.W.shape[0], "n_hidden": self.rbm.W.shape[1], "cd_k": self.rbm.cd_k}, os.path.join(job_dir,"rbm.pt"))
            torch.save({"state_dict": self.head.state_dict()}, os.path.join(job_dir,"head.pt"))
            with open(os.path.join(job_dir,"vectorizer.json"),"w",encoding="utf-8") as f: json.dump(self.vec.to_dict(), f, indent=2)
            with open(os.path.join(job_dir,"job_meta.json"),"w",encoding="utf-8") as f: json.dump({"f1_macro":float(f1_macro),"accuracy":float(acc),"feat_cols":self.feat_cols_}, f, indent=2)
        except Exception as ex:
            print("Warning(save):", ex)

        return {"f1_macro": float(f1_macro), "accuracy": float(acc), "job_dir": job_dir}


# alias histórico
ModeloRBMGeneral = RBMGeneral
