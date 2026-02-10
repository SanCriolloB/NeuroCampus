# backend/src/neurocampus/models/strategies/modelo_rbm_restringida.py
# RBM Student "restringida" con cabeza supervisada para {neg, neu, pos}.
# Alineado 1:1 en API con RBMGeneral:
# - Mismos métodos/propiedades: setup, fit, train_step, predict_proba_df, predict_df,
#   predict_proba, predict, save, load, feature_columns, feat_cols_.
# - Selección de features: calif_* + (opcional) p_* + (opcional) x_text_*.
#   Por defecto esta variante evita usar text_probs para reducir fuga, pero puede activarse.
#
# Alias exportado: ModeloRBMRestringida = RBMRestringida

from __future__ import annotations
import os, json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..utils.metrics import mae as _mae, rmse as _rmse, r2_score as _r2_score

__all__ = ["RBMRestringida", "ModeloRBMRestringida"]

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

def _strip_localfs(ref: str) -> str:
    s = str(ref or "").strip()
    return s[len("localfs://"):] if s.startswith("localfs://") else s


def _find_repo_root(start: Path) -> Path:
    # sube hasta encontrar una estructura típica de repo
    for p in [start] + list(start.parents):
        if (p / "backend").exists():
            return p
    # fallback razonable
    return start.parents[5]


def _resolve_ref_path(ref: str) -> Path:
    s = _strip_localfs(ref)
    p = Path(s)

    # absoluto
    if p.is_absolute():
        return p

    # relativo a cwd
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p

    # relativo al root del repo
    root = _find_repo_root(Path(__file__).resolve())
    return (root / p).resolve()


def _safe_lower(s) -> str:
    try:
        return str(s).lower()
    except Exception:
        return ""


def _suffix_index(name: str, prefix: str) -> int:
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
    """Orden de columnas de entrada:
       - calif_1..calif_{max_calif} si existen (o numéricas sin metadatos).
       - (opcional) p_neg/p_neu/p_pos
       - (opcional) x_text_* ordenadas por sufijo numérico."""
    cols = list(df.columns)

    # calificaciones
    califs = [c for c in cols if c.startswith("calif_")]
    if califs:
        def _idx(c: str):
            try:
                return int(c.split("_")[1])
            except Exception:
                return 10**9
        califs = sorted(califs, key=_idx)[:max_calif]
    else:
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        califs = [c for c in nums if _safe_lower(c) not in _META_EXCLUDE][:max_calif]

    features: List[str] = list(califs)

    # p_* (evitamos por defecto, pero soportamos si se activa)
    if include_text_probs and all(k in df.columns for k in ["p_neg", "p_neu", "p_pos"]):
        features += ["p_neg", "p_neu", "p_pos"]

    # x_text_* embeds
    if include_text_embeds:
        embed_cols = [c for c in cols if c.startswith(text_embed_prefix)]
        if embed_cols:
            embed_cols = sorted(embed_cols, key=lambda c: _suffix_index(c, text_embed_prefix))
            features += embed_cols

    return features


@dataclass
class _Vectorizer:
    mean_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    mode: str = "minmax"  # "minmax" | "scale_0_5"

    def fit(self, X: np.ndarray, mode: str = "minmax") -> "_Vectorizer":
        self.mode = mode

        # Aseguramos float32
        X = X.astype(np.float32, copy=False)

        # Estadísticos básicos permitiendo NaNs
        self.mean_ = np.nanmean(X, axis=0).astype(np.float32)

        if self.mode == "scale_0_5":
            self.min_ = np.zeros(X.shape[1], dtype=np.float32)
            self.max_ = np.ones(X.shape[1], dtype=np.float32) * 5.0
        else:
            self.min_ = np.nanmin(X, axis=0).astype(np.float32)
            self.max_ = np.nanmax(X, axis=0).astype(np.float32)

        # Columnas problemáticas: mean/min/max no finitos (todo NaN, inf, -inf)
        bad = (
            ~np.isfinite(self.mean_) |
            ~np.isfinite(self.min_) |
            ~np.isfinite(self.max_)
        )
        if np.any(bad):
            # Las tratamos como features sin información: siempre 0
            self.mean_[bad] = 0.0
            self.min_[bad] = 0.0
            self.max_[bad] = 1.0

        # Evitar rango casi 0 para no dividir por 0
        self.max_ = np.where(
            (self.max_ - self.min_) < 1e-9,
            self.min_ + 1.0,
            self.max_
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("Vectorizer no entrenado")
        if X.shape[1] != len(self.mean_):
            raise ValueError(f"Dimensión {X.shape[1]} != {len(self.mean_)} usada en fit()")

        X = X.astype(np.float32, copy=False)

        # Reemplazar inf/-inf por NaN para tratarlos igual
        X = np.where(np.isfinite(X), X, np.nan)

        # Imputar NaNs con la media de la columna
        Xc = np.where(np.isnan(X), self.mean_[None, :], X)

        # Escalado
        if self.mode == "scale_0_5":
            Xs = Xc / 5.0
        else:
            Xs = (Xc - self.min_[None, :]) / (self.max_[None, :] - self.min_[None, :])

        # Asegurar [0,1] y sanear NaNs/inf residuales
        Xs = np.clip(Xs, 0.0, 1.0)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)

        return Xs.astype(np.float32, copy=False)
    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mean_": None if self.mean_ is None else self.mean_.astype(float).tolist(),
            "min_": None if self.min_ is None else self.min_.astype(float).tolist(),
            "max_": None if self.max_ is None else self.max_.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "_Vectorizer":
        obj = cls()
        if not d:
            return obj
        obj.mode = str(d.get("mode", "minmax"))
        mean_ = d.get("mean_")
        min_ = d.get("min_")
        max_ = d.get("max_")
        obj.mean_ = None if mean_ is None else np.asarray(mean_, dtype=np.float32)
        obj.min_  = None if min_  is None else np.asarray(min_, dtype=np.float32)
        obj.max_  = None if max_  is None else np.asarray(max_, dtype=np.float32)
        return obj




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
        # --- AUTOFIX: sincronizar visibles de W/b_v con la cantidad de columnas de v ---
        try:
            nv = v.shape[1]
            vW, hW = self.W.shape
            if nv != vW:
                device = self.W.device
                dtype = self.W.dtype
                std = 0.01
                # mantener como nn.Parameter
                self.W  = nn.Parameter(torch.randn((nv, hW), device=device, dtype=dtype) * std)
                self.b_v = nn.Parameter(torch.zeros((nv,), device=device, dtype=dtype))
        except Exception:
            pass
        # --- fin AUTOFIX ---
        p_h = self._sigmoid(v @ self.W + self.b_h)
        h = torch.bernoulli(p_h)
        return p_h, h

    def sample_v(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        p_v = self._sigmoid(h @ self.W.t() + self.b_v)
        v = torch.bernoulli(p_v)
        return p_v, v

    def cd_step(self, v0: Tensor) -> Dict[str, float]:
        # --- AUTOFIX visibles: sincroniza W/b_v con n_features del batch (v0) ---
        try:
            nv = v0.shape[1]
            vW, hW = self.W.shape
            if nv != vW:
                device = self.W.device
                dtype  = self.W.dtype
                std = 0.01
                self.W  = nn.Parameter(torch.randn((nv, hW), device=device, dtype=dtype) * std)
                self.b_v = nn.Parameter(torch.zeros((nv,), device=device, dtype=dtype))
        except Exception:
            pass
        # --- fin AUTOFIX ---

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

        recon = torch.mean((v0 - pvk) ** 2).item()

        self.W.grad   = -dW
        self.b_v.grad = -dbv
        self.b_h.grad = -dbh

        grad_norm = torch.linalg.vector_norm(dW.detach()).item()
        return {"recon_error": recon, "grad_norm": grad_norm}

    def hidden_probs(self, v: Tensor) -> Tensor:
        return self._sigmoid(v @ self.W + self.b_h)

class _RegressionHead(nn.Module):
    """Head de regresión para score_docente (0–50) con embeddings opcionales.

    Nota: NO es un "modelo nuevo" (no MLP). Es una cabeza lineal supervisada
    encima del embedding del RBM.
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
        return self.linear(x).squeeze(1)


# --------------------------------
# Student: RBM + cabeza supervisada
# --------------------------------
class RBMRestringida:
    """Variante 'restringida' (mismas APIs que RBMGeneral)."""
    def __init__(self,
                 n_visible=None, n_hidden=None, cd_k=None,
                 lr_rbm=None, lr_head=None, momentum=None, weight_decay=None,
                 seed=None, device=None, **extra):
        hp = {}
        for k, v in dict(
            n_visible=n_visible, n_hidden=n_hidden, cd_k=cd_k,
            lr_rbm=lr_rbm, lr_head=lr_head, momentum=momentum,
            weight_decay=weight_decay, seed=seed, device=device
        ).items():
            if v is not None:
                hp[k] = v
        hp.update(extra or {})

        # Estado
        self.device: str = "cpu"
        self.vec: _Vectorizer = _Vectorizer()
        self.rbm: Optional[_RBM] = None
        self.head: Optional[nn.Module] = None
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
        self.scale_mode: str = "minmax"
        self.feat_cols_: List[str] = []
        self.text_embed_prefix_: str = "x_text_"
        self.opt_rbm: Optional[torch.optim.Optimizer] = None
        self.opt_head: Optional[torch.optim.Optimizer] = None
        self._epoch: int = 0
        self.classes_: List[str] = _CLASSES

        self.setup(data_ref=hp.pop("data_ref", None), hparams=hp)

    @property
    def feature_columns(self) -> List[str]:
        return list(self.feat_cols_)

    # ---------- helpers de IO ----------
    def _load_df(self, ref: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Carga df desde ruta (parquet/csv) o retorna copia si ya es DataFrame.

        Resuelve rutas relativas contra CWD y contra la raíz del repo, y soporta
        prefijos ``localfs://``.
        """
        if isinstance(ref, pd.DataFrame):
            return ref.copy()

        p = _resolve_ref_path(str(ref))
        if not p.exists():
            raise FileNotFoundError(str(p))

        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)

        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)

        # fallback por si llega una extensión rara
        try:
            return pd.read_parquet(p)
        except Exception:
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


    def _split_train_val_indices(
        self,
        df: pd.DataFrame,
        *,
        split_mode: str,
        val_ratio: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(len(df))
        if n < 2:
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

        val_ratio = min(max(float(val_ratio), 0.0), 0.9)
        n_val = max(1, int(round(n * val_ratio)))

        idx = np.arange(n, dtype=np.int64)
        sm = str(split_mode or "").lower()

        if sm == "temporal" and ("periodo" in df.columns):
            order = np.argsort(df["periodo"].apply(self._period_key).to_numpy())
            idx = idx[order]
            return idx[: n - n_val], idx[n - n_val :]

        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
        return idx[: n - n_val], idx[n - n_val :]


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

        # entrenar en escala 0..1 para estabilidad
        target_scale = float(getattr(self, "target_scale_", 50.0) or 50.0)
        if np.nanmax(y) <= 1.5 and target_scale > 1.5:
            target_scale = 1.0
        self.target_scale_ = float(target_scale)
        y_scaled = (y / float(target_scale)).astype(np.float32)

        tid = None
        mid = None
        if include_teacher_materia and str(teacher_materia_mode).lower() == "embed":
            if teacher_id_col not in df.columns or materia_id_col not in df.columns:
                raise ValueError("teacher_id/materia_id requeridos para modo embed. Faltan columnas.")

            tid = pd.to_numeric(df[teacher_id_col], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            mid = pd.to_numeric(df[materia_id_col], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

            # Importante: NO permitir que el vocab de embeddings se reduzca cuando
            # _prepare_xy_regression() se llama sobre validación (eso rompe índices de train).
            prev_t_vocab = int(getattr(self, "teacher_vocab_size_", 0) or 0)
            prev_m_vocab = int(getattr(self, "materia_vocab_size_", 0) or 0)

            if prev_t_vocab > 0:
                # vocab congelado (ya calculado en train); UNK es el último índice
                self.teacher_vocab_size_ = prev_t_vocab
                unk_t = self.teacher_vocab_size_ - 1
            else:
                tmax = int(np.nanmax(tid)) if tid.size else -1
                unk_t = tmax + 1
                self.teacher_vocab_size_ = int(unk_t + 1)

            if prev_m_vocab > 0:
                self.materia_vocab_size_ = prev_m_vocab
                unk_m = self.materia_vocab_size_ - 1
            else:
                mmax = int(np.nanmax(mid)) if mid.size else -1
                unk_m = mmax + 1
                self.materia_vocab_size_ = int(unk_m + 1)

            # Mapear inválidos / fuera de rango a UNK (evita index-out-of-range)
            tid = np.where((tid < 0) | (tid >= self.teacher_vocab_size_), unk_t, tid)
            mid = np.where((mid < 0) | (mid >= self.materia_vocab_size_), unk_m, mid)


        drop_cols = {target_col}
        if include_teacher_materia and str(teacher_materia_mode).lower() == "embed":
            drop_cols.update([teacher_id_col, materia_id_col])

        num_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        num_df = num_df.select_dtypes(include=[np.number]).copy()
        feat_cols = list(num_df.columns)
        if not feat_cols:
            raise ValueError("No hay columnas numéricas de features en pair_matrix (excluyendo ids/target).")

        X = num_df.to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        w = None
        if loss_weight_col and (loss_weight_col in df.columns):
            ww = pd.to_numeric(df[loss_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            w = np.log1p(np.clip(ww, 0.0, None)).astype(np.float32)
            if float(np.max(w)) <= 0.0:
                w = None

        return X, y_scaled.astype(np.float32), feat_cols, tid, mid, w


    def _iter_minibatches_reg(
        self,
        X: Tensor,
        y: Tensor,
        tid: Optional[Tensor],
        mid: Optional[Tensor],
        w: Optional[Tensor],
    ):
        n = int(X.shape[0])
        idx = torch.randperm(n, device=X.device)
        bs = int(getattr(self, "batch_size", 64))
        for i in range(0, n, bs):
            j = idx[i : i + bs]
            xb = X[j]
            yb = y[j]
            tb = tid[j] if tid is not None else None
            mb = mid[j] if mid is not None else None
            wb = w[j] if w is not None else None
            yield xb, yb, tb, mb, wb


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

            prev_cols = meta.get("feat_cols") or meta.get("feat_cols_") or []
            if not isinstance(prev_cols, list) or list(prev_cols) != list(getattr(self, "feat_cols_", [])):
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


    # ---------- API pública ----------
    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        self.seed = int(hparams.get("seed", 42) or 42)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() and bool(hparams.get("use_cuda", False)) else "cpu"

        self.batch_size = int(hparams.get("batch_size", 64))
        self.cd_k = int(hparams.get("cd_k", 1))
        self.lr_rbm = float(hparams.get("lr_rbm", 1e-2))
        self.lr_head = float(hparams.get("lr_head", 1e-2))
        self.momentum = float(hparams.get("momentum", 0.5))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))
        self.epochs_rbm = int(hparams.get("epochs_rbm", 1))
        self.scale_mode = str(hparams.get("scale_mode", "minmax"))

        # task selection
        family = str(hparams.get("family", "") or "").lower()
        self.task_type = str(hparams.get("task_type") or ("regression" if family == "score_docente" else "classification")).lower()

        # ---------- Si no hay data_ref ----------
        if not data_ref:
            self.rbm = None
            self.head = None
            self.opt_rbm = None
            self.opt_head = None
            self.X = None
            self.y = None
            # regresión
            self.X_tr = None; self.y_tr = None; self.tid_tr = None; self.mid_tr = None; self.w_tr = None
            self.X_va = None; self.y_va = None; self.tid_va = None; self.mid_va = None; self.w_va = None
            self.target_col_ = None
            self._epoch = 0
            return

        # ============================================================
        # REGRESIÓN (score_docente) -> pair_matrix.parquet
        # ============================================================
        if self.task_type == "regression":
            df = self._load_df(data_ref)
            if df.empty:
                raise ValueError("score_docente requiere data_ref con pair_matrix.parquet")

            # config ids/target
            self.include_teacher_materia_ = bool(hparams.get("include_teacher_materia", True))
            self.teacher_materia_mode_ = str(hparams.get("teacher_materia_mode", "embed")).lower()
            self.teacher_id_col_ = str(hparams.get("teacher_id_col", "teacher_id"))
            self.materia_id_col_ = str(hparams.get("materia_id_col", "materia_id"))
            self.embed_dim_ = int(hparams.get("embed_dim", 16))
            self.target_scale_ = float(hparams.get("target_scale", 50.0) or 50.0)

            # target_col: request > pair_meta.json > heurística
            target_col = str(hparams.get("target_col") or "").strip() or None
            try:
                ref = str(data_ref)
                if ref.startswith("localfs://"):
                    ref = ref.replace("localfs://", "", 1)
                meta_path = Path(ref).with_name("pair_meta.json")
                if meta_path.exists():
                    m = json.loads(meta_path.read_text(encoding="utf-8")) or {}
                    target_col = target_col or m.get("target_col") or m.get("target")
            except Exception:
                pass

            candidates = [
                target_col,
                "target_score",
                "score_total_0_50",
                "mean_score_total_0_50",
                "mean_score_total",
                "mean_score_base_0_50",
                "score_base_0_50",
            ]
            self.target_col_ = None
            for c in candidates:
                if c and (c in df.columns):
                    self.target_col_ = str(c)
                    break
            if not self.target_col_:
                raise ValueError("No se pudo resolver target_col para score_docente desde pair_matrix/pair_meta.")

            # split sobre df (consistente)
            split_mode = str(hparams.get("split_mode", "temporal")).lower()
            val_ratio = float(hparams.get("val_ratio", 0.2))
            idx_tr, idx_va = self._split_train_val_indices(df, split_mode=split_mode, val_ratio=val_ratio, seed=self.seed)

            df_tr = df.iloc[idx_tr].reset_index(drop=True)
            df_va = df.iloc[idx_va].reset_index(drop=True) if idx_va.size else df.iloc[0:0].copy()

            loss_weight_col = hparams.get("loss_weight_col", "n_par")

            X_tr_np, y_tr_np, feat_cols, tid_tr_np, mid_tr_np, w_tr_np = self._prepare_xy_regression(
                df_tr,
                target_col=self.target_col_,
                include_teacher_materia=self.include_teacher_materia_,
                teacher_materia_mode=self.teacher_materia_mode_,
                teacher_id_col=self.teacher_id_col_,
                materia_id_col=self.materia_id_col_,
                loss_weight_col=str(loss_weight_col) if loss_weight_col else None,
            )

            if len(df_va):
                X_va_np, y_va_np, _feat_cols2, tid_va_np, mid_va_np, w_va_np = self._prepare_xy_regression(
                    df_va,
                    target_col=self.target_col_,
                    include_teacher_materia=self.include_teacher_materia_,
                    teacher_materia_mode=self.teacher_materia_mode_,
                    teacher_id_col=self.teacher_id_col_,
                    materia_id_col=self.materia_id_col_,
                    loss_weight_col=str(loss_weight_col) if loss_weight_col else None,
                )
            else:
                X_va_np = np.zeros((0, X_tr_np.shape[1]), dtype=np.float32)
                y_va_np = np.zeros((0,), dtype=np.float32)
                tid_va_np = None; mid_va_np = None; w_va_np = None

            self.feat_cols_ = list(feat_cols)

            # vectorizer
            self.vec = _Vectorizer().fit(X_tr_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
            X_tr_np = self.vec.transform(X_tr_np)
            X_va_np = self.vec.transform(X_va_np) if X_va_np.size else X_va_np

            # tensores
            self.X_tr = torch.from_numpy(X_tr_np).to(self.device)
            self.y_tr = torch.from_numpy(y_tr_np).to(self.device)
            self.X_va = torch.from_numpy(X_va_np).to(self.device)
            self.y_va = torch.from_numpy(y_va_np).to(self.device)

            self.tid_tr = torch.from_numpy(tid_tr_np).to(self.device) if tid_tr_np is not None else None
            self.mid_tr = torch.from_numpy(mid_tr_np).to(self.device) if mid_tr_np is not None else None
            self.tid_va = torch.from_numpy(tid_va_np).to(self.device) if tid_va_np is not None else None
            self.mid_va = torch.from_numpy(mid_va_np).to(self.device) if mid_va_np is not None else None

            self.w_tr = torch.from_numpy(w_tr_np).to(self.device) if w_tr_np is not None else None
            self.w_va = torch.from_numpy(w_va_np).to(self.device) if w_va_np is not None else None

            # compat legacy: evitar evaluación de clasificación
            self.X = self.X_tr
            self.y = None

            # modelo
            n_visible = int(self.X_tr.shape[1])
            n_hidden = int(hparams.get("n_hidden", 32))
            self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
            self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum, weight_decay=self.weight_decay)

            include_ids = bool(self.include_teacher_materia_) and (self.teacher_materia_mode_ == "embed")
            self.head = _RegressionHead(
                n_hidden=n_hidden,
                n_teachers=max(1, int(getattr(self, "teacher_vocab_size_", 1) or 1)),
                n_materias=max(1, int(getattr(self, "materia_vocab_size_", 1) or 1)),
                emb_dim=int(self.embed_dim_),
                include_ids=include_ids,
                dropout=float(hparams.get("dropout", 0.0) or 0.0),
            ).to(self.device)
            self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

            warm_path = hparams.get("warm_start_path")
            if warm_path:
                self._warm_start_info_ = self._try_warm_start(str(warm_path))
            else:
                self._warm_start_info_ = {"warm_start": "none"}

            self._epoch = 0
            return

        # ============================================================
        # CLASIFICACIÓN (sentiment_desempeno) -> como estaba
        # ============================================================
        accept_teacher = bool(hparams.get("accept_teacher", True))
        accept_threshold = float(hparams.get("accept_threshold", 0.80))
        max_calif = int(hparams.get("max_calif", 10))

        include_text_probs = bool(hparams.get("use_text_probs", False))
        include_text_embeds = bool(hparams.get("use_text_embeds", False))
        self.text_embed_prefix_ = str(hparams.get("text_embed_prefix", "x_text_"))

        df = self._load_df(data_ref)
        X_np, y_np, feat_cols = self._prepare_xy(
            df,
            accept_teacher=accept_teacher,
            threshold=accept_threshold,
            max_calif=max_calif,
            include_text_probs=include_text_probs,
            include_text_embeds=include_text_embeds,
            text_embed_prefix=self.text_embed_prefix_,
        )
        self.feat_cols_ = list(feat_cols)

        # ROBUSTEZ: clasificación requiere labels sí o sí
        if y_np is None:
            raise ValueError(
                "Entrenamiento classification requiere labels de sentimiento, pero no se encontraron "
                "columnas como y_sentimiento / y / sentiment_label_teacher en el dataset de entrenamiento. "
                "Solución: asegúrate de construir el feature-pack desde data/labeled/<ds>_beto.parquet "
                "(que trae p_neg/p_neu/p_pos) y reconstruye artifacts/features/<ds> con force=true."
            )

        self.vec = _Vectorizer().fit(X_np, mode=("scale_0_5" if self.scale_mode == "scale_0_5" else "minmax"))
        X_np = self.vec.transform(X_np)

        self.X = torch.from_numpy(X_np).to(self.device)

        n_visible = X_np.shape[1]
        n_hidden = int(hparams.get("n_hidden", 32))
        self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
        self.opt_rbm = torch.optim.SGD(
            self.rbm.parameters(),
            lr=self.lr_rbm,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.head = nn.Linear(n_hidden, len(_CLASSES)).to(self.device)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        self.y = torch.from_numpy(y_np).to(self.device)

        self._epoch = 0


    # ---------- Mini-batches ----------
    def _iter_minibatches(self, X: Tensor, y: Optional[Tensor]):
        idx = torch.randperm(X.shape[0], device=X.device)
        for start in range(0, len(idx), self.batch_size):
            sel = idx[start:start + self.batch_size]
            yield X[sel], (None if y is None else y[sel])

    # ---------- Entrenamiento ----------
    def train_step(self, epoch: int, hparams: Optional[Dict[str, Any]] = None, y: Any = None):
        assert self.rbm is not None and self.X is not None
        self._epoch = epoch
        t0 = time.perf_counter() if "time" in globals() else None  # defensivo

        # -------------------------
        # REGRESIÓN (score_docente)
        # -------------------------
        if getattr(self, "task_type", "classification") == "regression":
            assert self.X_tr is not None and self.y_tr is not None and self.head is not None and self.opt_head is not None

            rbm_losses: List[float] = []
            rbm_grad: List[float] = []
            reg_losses: List[float] = []

            # RBM (CD-k)
            self.rbm.train()
            for _ in range(max(1, int(getattr(self, "epochs_rbm", 1)))):
                for xb, _ in self._iter_minibatches(self.X_tr, None):
                    m = self.rbm.cd_step(xb)
                    rbm_losses.append(float(m["recon_error"]))
                    rbm_grad.append(float(m["grad_norm"]))
                    # update manual (como estaba en clasificación)
                    for p in self.rbm.parameters():
                        if p.grad is not None:
                            p.data -= float(self.lr_rbm) * p.grad

            # Head supervised (MSE)
            self.head.train()
            for xb, yb, tid, mid, wb in self._iter_minibatches_reg(self.X_tr, self.y_tr, self.tid_tr, self.mid_tr, self.w_tr):
                with torch.no_grad():
                    H = self.rbm.hidden_probs(xb)

                self.opt_head.zero_grad(set_to_none=True)

                pred = self.head(H, tid, mid) if isinstance(self.head, _RegressionHead) else self.head(H).squeeze(1)
                diff2 = (pred - yb) ** 2

                if wb is not None:
                    w_norm = wb / (wb.mean() + 1e-8)
                    loss_reg = torch.mean(w_norm * diff2)
                else:
                    loss_reg = torch.mean(diff2)

                loss_reg.backward()
                self.opt_head.step()
                reg_losses.append(float(loss_reg.detach().cpu()))

            # Métricas (train/val) en escala 0..50
            def _predict_scaled(X: Tensor, tid: Optional[Tensor], mid: Optional[Tensor]) -> np.ndarray:
                self.rbm.eval()
                self.head.eval()
                with torch.no_grad():
                    H = self.rbm.hidden_probs(X)
                    yy = self.head(H, tid, mid) if isinstance(self.head, _RegressionHead) else self.head(H).squeeze(1)
                return yy.detach().cpu().numpy().astype(np.float32, copy=False)

            target_scale = float(getattr(self, "target_scale_", 50.0) or 50.0)

            y_tr_true = (self.y_tr.detach().cpu().numpy().astype(np.float32, copy=False) * target_scale)
            y_tr_pred = (_predict_scaled(self.X_tr, self.tid_tr, self.mid_tr) * target_scale)

            y_va_true = (self.y_va.detach().cpu().numpy().astype(np.float32, copy=False) * target_scale) if (self.y_va is not None) else np.zeros((0,), dtype=np.float32)
            y_va_pred = (_predict_scaled(self.X_va, self.tid_va, self.mid_va) * target_scale) if (self.X_va is not None and self.X_va.numel() > 0) else np.zeros((0,), dtype=np.float32)

            train_mae = _mae(y_tr_true, y_tr_pred)
            train_rmse = _rmse(y_tr_true, y_tr_pred)
            train_r2 = _r2_score(y_tr_true, y_tr_pred)

            val_mae = _mae(y_va_true, y_va_pred) if y_va_true.size else None
            val_rmse = _rmse(y_va_true, y_va_pred) if y_va_true.size else None
            val_r2 = _r2_score(y_va_true, y_va_pred) if y_va_true.size else None

            # análisis por evidencia (si w_va ~ log1p(n_par))
            val_by_evidence = None
            if (self.w_va is not None) and y_va_true.size:
                n_par = (np.exp(self.w_va.detach().cpu().numpy().astype(np.float32, copy=False)) - 1.0)
                bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 1_000_000)]
                packs = []
                for lo, hi in bins:
                    m = (n_par >= float(lo)) & (n_par <= float(hi))
                    if not np.any(m):
                        continue
                    packs.append(
                        {
                            "bin": f"{int(lo)}-{int(hi)}" if hi < 999_999 else f"{int(lo)}+",
                            "n": int(np.sum(m)),
                            "mae": _mae(y_va_true[m], y_va_pred[m]),
                            "rmse": _rmse(y_va_true[m], y_va_pred[m]),
                        }
                    )
                val_by_evidence = packs

            time_epoch_ms = 0.0
            if t0 is not None:
                time_epoch_ms = float((time.perf_counter() - t0) * 1000.0)

            metrics: Dict[str, Any] = {
                "epoch": float(epoch),
                "task_type": "regression",
                "target_col": getattr(self, "target_col_", None),
                "recon_error": float(np.mean(rbm_losses)) if rbm_losses else 0.0,
                "rbm_grad_norm": float(np.mean(rbm_grad)) if rbm_grad else 0.0,
                "reg_loss": float(np.mean(reg_losses)) if reg_losses else 0.0,
                "train_mae": float(train_mae),
                "train_rmse": float(train_rmse),
                "train_r2": float(train_r2),
                "val_mae": None if val_mae is None else float(val_mae),
                "val_rmse": None if val_rmse is None else float(val_rmse),
                "val_r2": None if val_r2 is None else float(val_r2),
                "n_train": int(len(y_tr_true)),
                "n_val": int(len(y_va_true)),
                "pred_min": float(np.min(y_va_pred)) if y_va_pred.size else float(np.min(y_tr_pred)),
                "pred_max": float(np.max(y_va_pred)) if y_va_pred.size else float(np.max(y_tr_pred)),
                "time_epoch_ms": time_epoch_ms,
                "loss": float((np.mean(rbm_losses) if rbm_losses else 0.0) + (np.mean(reg_losses) if reg_losses else 0.0)),
            }
            if val_by_evidence is not None:
                metrics["val_by_evidence"] = val_by_evidence
            if hasattr(self, "_warm_start_info_"):
                metrics["warm_start"] = getattr(self, "_warm_start_info_", None)

            return float(metrics.get("loss", 0.0)), metrics

        # -------------------------
        # CLASIFICACIÓN (existente)
        # -------------------------
        assert self.opt_rbm is not None and self.head is not None and self.opt_head is not None

        rbm_losses, rbm_grad = [], []
        cls_losses = []

        self.rbm.train()
        for _ in range(max(1, self.epochs_rbm)):
            for xb, _ in self._iter_minibatches(self.X, self.y):
                self.opt_rbm.zero_grad(set_to_none=True)
                m = self.rbm.cd_step(xb)
                rbm_losses.append(m["recon_error"])
                rbm_grad.append(m["grad_norm"])
                for p in self.rbm.parameters():
                    if p.grad is not None:
                        p.data -= self.lr_rbm * p.grad

        if self.y is not None:
            n_classes = self.head.out_features
            counts = torch.stack([(self.y == i).sum() for i in range(n_classes)]).float()
            if counts.sum() <= 0:
                weights = torch.ones(n_classes, device=self.device)
            else:
                weights = (counts.sum() / (counts + 1e-9))
                weights = (weights / weights.sum()) * n_classes
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
            "task_type": "classification",
            "recon_error": float(np.mean(rbm_losses)) if rbm_losses else 0.0,
            "rbm_grad_norm": float(np.mean(rbm_grad)) if rbm_grad else 0.0,
            "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "time_epoch_ms": 0.0
        }
        return metrics["recon_error"] + metrics["cls_loss"], metrics



    def fit(self,
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y: Optional[Union[np.ndarray, List[int]]] = None,
            epochs: int = 1,
            log_every: int = 1,
            **_):
        if X is not None:
            if isinstance(X, pd.DataFrame):
                if not self.feat_cols_:
                    cals = [c for c in X.columns if c.startswith("calif_")]
                    embeds = [c for c in X.columns if c.startswith(self.text_embed_prefix_)]
                    self.feat_cols_ = cals + embeds
                X_np = X[self.feat_cols_].to_numpy(dtype=np.float32)
                if self.vec.mean_ is None:
                    self.vec.fit(X_np, mode=self.scale_mode)
                try:
                    Xs = self.vec.transform(X_np)
                except Exception:
                    mn, mx = np.nanmin(X_np, axis=0), np.nanmax(X_np, axis=0)
                    mx = np.where((mx - mn) < 1e-9, mn + 1.0, mx)
                    Xs = np.clip((X_np - mn) / (mx - mn), 0, 1).astype(np.float32)
                self.X = torch.from_numpy(Xs).to(self.device)
            else:
                # X viene como np.ndarray desde run_kfold_audit
                X_np = np.asarray(X, dtype=np.float32)

                # limpiar NaN / inf antes de cualquier cosa
                X_np = np.nan_to_num(X_np, nan=0.0, posinf=1.0, neginf=0.0)

                if self.vec.mean_ is not None and X_np.shape[1] == len(self.vec.mean_):
                    try:
                        Xs = self.vec.transform(X_np)
                    except Exception:
                        # Fallback robusto: ya no hay NaNs gracias a nan_to_num
                        Xs = np.clip(X_np, 0.0, 1.0)
                else:
                    # Primera vez: todavía no hay vectorizador entrenado
                    Xs = np.clip(X_np, 0.0, 1.0)

                self.X = torch.from_numpy(Xs).to(self.device)

        if y is not None:
            y_np = np.asarray(y, dtype=np.int64)
            n_classes = 3 if (self.head is None) else self.head.out_features
            bad = (y_np < 0) | (y_np >= n_classes)
            if np.any(bad):
                y_np = np.clip(y_np, 0, n_classes - 1)
            self.y = torch.from_numpy(y_np).to(self.device)

        # **Clave**: crear (o re-crear) la RBM con el número correcto de visibles
        if self.rbm is None or self.head is None:
            if self.X is None:
                raise RuntimeError("No hay datos para entrenar. Llama setup(data_ref=...) o fit(X=...).")
            n_visible = self.X.shape[1]
            n_hidden = int(os.environ.get("RBM_N_HIDDEN", "32"))
            self.rbm = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
            self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum, weight_decay=self.weight_decay)
            self.head = nn.Linear(n_hidden, len(_CLASSES)).to(self.device)
            self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)
        else:
            # Si ya existía una RBM previa (p.ej. creada en setup() sin data_ref), ajustarla al tamaño real
            if self.X is not None:
                nv_need = int(self.X.shape[1])
                nv_curr, n_hidden = int(self.rbm.W.shape[0]), int(self.rbm.W.shape[1])
                if nv_need != nv_curr:
                    device = self.rbm.W.device
                    dtype  = self.rbm.W.dtype
                    g = torch.Generator().manual_seed(int(self.seed))
                    self.rbm = _RBM(n_visible=nv_need, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(device)
                    self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum, weight_decay=self.weight_decay)
                    # head no depende de visibles; no es necesario reconstruirla

        for ep in range(1, int(epochs) + 1):
            _, _ = self.train_step(ep)
            if log_every and (ep % int(log_every) == 0):
                pass
        return self

    # ---------- Transformaciones / Inferencia ----------
    def _df_to_X(self, df: pd.DataFrame) -> np.ndarray:
        assert len(self.feat_cols_) > 0, "El modelo no tiene feat_cols_ configuradas."
        missing = [c for c in self.feat_cols_ if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0
        return df[self.feat_cols_].to_numpy(dtype=np.float32)

    def _transform_np(self, X_np: np.ndarray) -> Tensor:
        if self.vec.mean_ is not None and X_np.shape[1] == len(self.vec.mean_):
            Xs = self.vec.transform(X_np)
        else:
            mn, mx = np.nanmin(X_np, axis=0), np.nanmax(X_np, axis=0)
            mx = np.where((mx - mn) < 1e-9, mn + 1.0, mx)
            Xs = np.clip((X_np - mn) / (mx - mn), 0, 1).astype(np.float32)
        Xt = torch.from_numpy(Xs.astype(np.float32, copy=False)).to(self.device)
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
        return H

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "task_type", "classification") == "regression":
            raise ValueError("Este run es de REGRESIÓN. Usa predict_score_df() para score_docente.")
        X_np = self._df_to_X(df.copy())
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            proba = F.softmax(self.head(H), dim=1).cpu().numpy()
        return proba

    def predict_df(self, df: pd.DataFrame) -> List[str]:
        if getattr(self, "task_type", "classification") == "regression":
            raise ValueError("Este run es de REGRESIÓN. Usa predict_score_df() para score_docente.")
        idx = self.predict_proba_df(df).argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    def predict_proba(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if getattr(self, "task_type", "classification") == "regression":
            raise ValueError("Este run es de REGRESIÓN. Usa predict_score_df() para score_docente.")
        if isinstance(X_or_df, pd.DataFrame):
            return self.predict_proba_df(X_or_df)
        X_np = np.asarray(X_or_df, dtype=np.float32)
        self.rbm.eval(); self.head.eval()
        H = self._transform_np(X_np)
        with torch.no_grad():
            return F.softmax(self.head(H), dim=1).cpu().numpy()

    def predict(self, X_or_df: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        if getattr(self, "task_type", "classification") == "regression":
            raise ValueError("Este run es de REGRESIÓN. Usa predict_score_df() para score_docente.")
        proba = self.predict_proba(X_or_df)
        idx = proba.argmax(axis=1)
        return [_INV_LABEL_MAP[i] for i in idx]

    # --------- REGRESIÓN: score ---------
    def predict_score_df(self, df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "task_type", "classification") != "regression":
            raise ValueError("Este run es de CLASIFICACIÓN. Usa predict_proba_df() para sentiment.")
        assert isinstance(self.head, _RegressionHead), "Head de regresión no inicializada."
        assert self.rbm is not None, "RBM no inicializada."

        df = df.copy()

        # ---------- X: usar feat_cols_ entrenadas (no requiere target_col) ----------
        assert len(self.feat_cols_) > 0, "El modelo no tiene feat_cols_ configuradas."
        for c in self.feat_cols_:
            if c not in df.columns:
                df[c] = 0.0

        X_np = df[self.feat_cols_].to_numpy(dtype=np.float32)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        if self.vec.mean_ is not None and X_np.shape[1] == len(self.vec.mean_):
            Xs = self.vec.transform(X_np)
        else:
            Xs = np.clip(X_np, 0.0, 1.0).astype(np.float32, copy=False)

        Xt = torch.from_numpy(Xs.astype(np.float32, copy=False)).to(self.device)

        # ---------- IDs (opcionales): teacher/materia embeddings ----------
        tid = None
        mid = None
        include_ids = bool(getattr(self, "include_teacher_materia_", True)) and (str(getattr(self, "teacher_materia_mode_", "embed")).lower() == "embed")

        if include_ids:
            tcol = str(getattr(self, "teacher_id_col_", "teacher_id"))
            mcol = str(getattr(self, "materia_id_col_", "materia_id"))

            if tcol not in df.columns:
                df[tcol] = -1
            if mcol not in df.columns:
                df[mcol] = -1

            tid_np = pd.to_numeric(df[tcol], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            mid_np = pd.to_numeric(df[mcol], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

            # mapear ids fuera de rango a UNK (último índice)
            tv = int(getattr(self, "teacher_vocab_size_", 0) or 0)
            mv = int(getattr(self, "materia_vocab_size_", 0) or 0)
            unk_t = max(0, tv - 1)
            unk_m = max(0, mv - 1)

            if tv > 0:
                tid_np = np.where((tid_np < 0) | (tid_np >= tv), unk_t, tid_np)
            else:
                tid_np = np.where(tid_np < 0, 0, tid_np)

            if mv > 0:
                mid_np = np.where((mid_np < 0) | (mid_np >= mv), unk_m, mid_np)
            else:
                mid_np = np.where(mid_np < 0, 0, mid_np)

            tid = torch.from_numpy(tid_np).to(self.device)
            mid = torch.from_numpy(mid_np).to(self.device)

        # ---------- forward ----------
        self.rbm.eval()
        self.head.eval()
        with torch.no_grad():
            H = self.rbm.hidden_probs(Xt)
            y_hat = self.head(H, tid, mid).detach().cpu().numpy().astype(np.float32, copy=False)

        target_scale = float(getattr(self, "target_scale_", 50.0) or 50.0)
        return y_hat * target_scale



    # ---------- Persistencia ----------
    def save(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(out_dir, "vectorizer.json"), "w", encoding="utf-8") as f:
            json.dump(self.vec.to_dict(), f, ensure_ascii=False, indent=2)

        torch.save(
            {
                "state_dict": self.rbm.state_dict(),
                "n_visible": int(self.rbm.W.shape[0]),
                "n_hidden": int(self.rbm.W.shape[1]),
                "cd_k": int(self.rbm.cd_k),
            },
            os.path.join(out_dir, "rbm.pt"),
        )

        # head puede ser clasificación (Linear) o regresión (_RegressionHead)
        torch.save({"state_dict": self.head.state_dict()}, os.path.join(out_dir, "head.pt"))

        meta = {
            "task_type": getattr(self, "task_type", "classification"),
            "feat_cols": self.feat_cols_,
            "scale_mode": self.scale_mode,
            "classes": getattr(self, "classes_", _CLASSES),
            "text_embed_prefix": getattr(self, "text_embed_prefix_", "x_text_"),

            # campos de regresión (si aplica)
            "target_col": getattr(self, "target_col_", None),
            "target_scale": getattr(self, "target_scale_", 50.0),
            "include_teacher_materia": getattr(self, "include_teacher_materia_", True),
            "teacher_materia_mode": getattr(self, "teacher_materia_mode_", "embed"),
            "teacher_id_col": getattr(self, "teacher_id_col_", "teacher_id"),
            "materia_id_col": getattr(self, "materia_id_col_", "materia_id"),
            "embed_dim": getattr(self, "embed_dim_", 16),
            "teacher_vocab_size": getattr(self, "teacher_vocab_size_", None),
            "materia_vocab_size": getattr(self, "materia_vocab_size_", None),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: str, device: Optional[str] = None) -> "RBMRestringida":
        obj = cls()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.device = device

        with open(os.path.join(in_dir, "vectorizer.json"), "r", encoding="utf-8") as f:
            obj.vec = _Vectorizer.from_dict(json.load(f))

        rbm_ckpt = torch.load(os.path.join(in_dir, "rbm.pt"), map_location=device)
        obj.rbm = _RBM(
            n_visible=int(rbm_ckpt["n_visible"]),
            n_hidden=int(rbm_ckpt["n_hidden"]),
            cd_k=int(rbm_ckpt.get("cd_k", 1)),
        )
        obj.rbm.load_state_dict(rbm_ckpt["state_dict"])
        obj.rbm.to(device)
        obj.opt_rbm = torch.optim.SGD(obj.rbm.parameters(), lr=1e-6)

        meta = {}
        meta_path = os.path.join(in_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}

        obj.task_type = str(meta.get("task_type", "classification")).lower()

        head_ckpt = torch.load(os.path.join(in_dir, "head.pt"), map_location=device)

        if obj.task_type == "regression":
            obj.target_col_ = meta.get("target_col")
            obj.target_scale_ = float(meta.get("target_scale", 50.0) or 50.0)

            obj.include_teacher_materia_ = bool(meta.get("include_teacher_materia", True))
            obj.teacher_materia_mode_ = str(meta.get("teacher_materia_mode", "embed")).lower()
            obj.teacher_id_col_ = str(meta.get("teacher_id_col", "teacher_id"))
            obj.materia_id_col_ = str(meta.get("materia_id_col", "materia_id"))
            obj.embed_dim_ = int(meta.get("embed_dim", 16) or 16)

            obj.teacher_vocab_size_ = meta.get("teacher_vocab_size")
            obj.materia_vocab_size_ = meta.get("materia_vocab_size")

            include_ids = bool(obj.include_teacher_materia_) and (obj.teacher_materia_mode_ == "embed")
            obj.head = _RegressionHead(
                n_hidden=int(rbm_ckpt["n_hidden"]),
                n_teachers=max(1, int(obj.teacher_vocab_size_ or 1)),
                n_materias=max(1, int(obj.materia_vocab_size_ or 1)),
                emb_dim=int(obj.embed_dim_),
                include_ids=include_ids,
                dropout=0.0,
            ).to(device)
        else:
            obj.head = nn.Linear(int(rbm_ckpt["n_hidden"]), len(_CLASSES)).to(device)
            obj.classes_ = meta.get("classes", _CLASSES)

        obj.head.load_state_dict(head_ckpt["state_dict"])
        obj.opt_head = torch.optim.Adam(obj.head.parameters(), lr=1e-6)

        obj.feat_cols_ = list(meta.get("feat_cols") or meta.get("feature_cols") or meta.get("feature_columns") or [])
        obj.scale_mode = str(meta.get("scale_mode", obj.vec.mode))
        obj.text_embed_prefix_ = str(meta.get("text_embed_prefix", "x_text_"))

        obj.X = None
        obj.y = None
        obj._epoch = 0
        return obj



# Alias para compatibilidad
ModeloRBMRestringida = RBMRestringida
