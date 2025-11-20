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

# ============================
# Mapeos y utilidades
# ============================

_LABEL_MAP = {"neg": 0, "neu": 1, "pos": 2}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}

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
            y_raw = pd.Series([""] * len(df))

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
        return X, y_np, feat_cols

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

        include_text_probs   = bool(hparams.get("use_text_probs", False))
        include_text_embeds  = bool(hparams.get("use_text_embeds", False))
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
        n_hidden  = int(hparams.get("n_hidden", self.n_hidden or 32))
        self.rbm  = _RBM(n_visible=n_visible, n_hidden=n_hidden, cd_k=self.cd_k, seed=self.seed).to(self.device)
        self.opt_rbm = torch.optim.SGD(self.rbm.parameters(), lr=self.lr_rbm, momentum=self.momentum)

        self.head = nn.Sequential(nn.Linear(n_hidden, 3)).to(self.device)
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)

        self.y = torch.from_numpy(y_np).to(self.device) if y_np is not None else None
        self._epoch = 0

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
            "hparams": {"scale_mode": self.scale_mode, "text_embed_prefix": self.text_embed_prefix_, "cd_k": self.cd_k},
        }
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
        obj.head = nn.Sequential(nn.Linear(rbm_ckpt["n_hidden"], 3)).to(device)
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
