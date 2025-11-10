# backend/src/neurocampus/features/tfidf_lsa.py
import json
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .base import TextFeaturizer


def _sanitize_min_df(val: Optional[float | int], default: int = 3) -> float | int:
    # None -> default
    if val is None:
        return default
    # enteros >=1 se aceptan tal cual
    if isinstance(val, int):
        return max(1, val)
    # floats:
    if isinstance(val, float):
        # 0 < val < 1.0 => fracción válida
        if 0.0 < val < 1.0:
            return val
        # val >= 1.0 => trátalo como entero (round)
        if val >= 1.0:
            return int(round(val))
    return default


def _sanitize_max_df(val: Optional[float | int]) -> Optional[float | int]:
    if val is None:
        return None
    if isinstance(val, int):
        # entero <=1 => ignóralo, no tiene sentido
        return None if val <= 1 else val
    if isinstance(val, float):
        # 0<val<=1.0 válido como fracción; >1.0 no tiene sentido => None
        return val if 0.0 < val <= 1.0 else None
    return None


class TfidfLSAFeaturizer(TextFeaturizer):
    """
    TF-IDF + LSA robusto frente a corpora pequeños o vacíos:
    - Soporta max_df opcional
    - Reintenta con min_df=1 y max_df=None si sklearn prunea todo
    - Si aún no hay vocabulario, retorna features cero
    """

    def __init__(
        self,
        max_features: int = 20000,
        n_components: int = 64,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: float | int = 3,
        max_df: Optional[float | int] = None,
    ):
        self.max_features = max_features
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.min_df = _sanitize_min_df(min_df, default=3)
        self.max_df = _sanitize_max_df(max_df)

        vec_kwargs: Dict[str, Any] = dict(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        if self.max_df is not None:
            vec_kwargs["max_df"] = self.max_df

        self.vec: Optional[TfidfVectorizer] = TfidfVectorizer(**vec_kwargs)
        self.svd: Optional[TruncatedSVD] = TruncatedSVD(n_components=self.n_components, random_state=42)
        self._degenerate = False  # marca “sin vocabulario”: produce ceros

    # ---------- helpers internos ----------
    def _zeros(self, n_rows: int) -> np.ndarray:
        return np.zeros((n_rows, self.n_components), dtype=np.float32)

    def _retry_relaxed(self, texts: List[str]) -> np.ndarray:
        """Reintento relajando pruning (min_df=1, max_df=None)."""
        self.min_df = 1
        self.max_df = None
        self.vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        # Si aún no hay vocabulario, marcamos degenerado
        try:
            X = self.vec.fit_transform(texts)
            if X.shape[1] == 0:
                self._degenerate = True
                return self._zeros(len(texts))
            Z = self.svd.fit_transform(X)
            return Z.astype(np.float32)
        except Exception:
            self._degenerate = True
            return self._zeros(len(texts))

    # ---------- API principal ----------
    def fit(self, texts: List[str]) -> "TfidfLSAFeaturizer":
        self._degenerate = False
        if self.vec is None or self.svd is None:
            # reconstruir por si carga externa dejó None
            self.vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                **({"max_df": self.max_df} if self.max_df is not None else {}),
            )
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        try:
            X = self.vec.fit_transform(texts)
            if X.shape[1] == 0:
                # sin vocabulario; marcar y no romper
                self._degenerate = True
                return self
            self.svd.fit(X)
            return self
        except ValueError:
            # pruning dejó sin términos => relajamos
            self._retry_relaxed(texts)
            return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self._degenerate or self.vec is None or self.svd is None:
            return self._zeros(len(texts))
        try:
            X = self.vec.transform(texts)
            if X.shape[1] == 0:
                return self._zeros(len(texts))
            Z = self.svd.transform(X)
            return Z.astype(np.float32)
        except Exception:
            return self._zeros(len(texts))

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self._degenerate = False
        if self.vec is None or self.svd is None:
            self.vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                **({"max_df": self.max_df} if self.max_df is not None else {}),
            )
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        try:
            X = self.vec.fit_transform(texts)
            if X.shape[1] == 0:
                self._degenerate = True
                return self._zeros(len(texts))
            Z = self.svd.fit_transform(X)
            return Z.astype(np.float32)
        except ValueError:
            # pruning sin términos: reintenta relajado
            return self._retry_relaxed(texts)

    # ---------- Persistencia ----------
    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, p / "tfidf.joblib")
        joblib.dump(self.svd, p / "svd.joblib")
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TfidfLSAFeaturizer":
        p = Path(path)
        # intenta leer hiperparámetros
        max_features = 20000
        n_components = 64
        ngram_range: Tuple[int, int] = (1, 2)
        min_df: float | int = 3
        max_df: Optional[float | int] = None

        meta_file = p / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                max_features = meta.get("max_features", max_features)
                n_components = meta.get("n_components", n_components)
                ngram_range = tuple(meta.get("ngram_range", list(ngram_range)))  # type: ignore
                min_df = meta.get("min_df", min_df)
                max_df = meta.get("max_df", max_df)
            except Exception:
                pass

        obj = cls(
            max_features=max_features,
            n_components=n_components,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        try:
            obj.vec = joblib.load(p / "tfidf.joblib")
            obj.svd = joblib.load(p / "svd.joblib")
        except Exception:
            # si no existen, quedará en modo degenerado
            obj.vec = None
            obj.svd = None
            obj._degenerate = True
        return obj

    # ---------- Metadatos ----------
    @property
    def meta(self) -> Dict[str, Any]:
        vocab_size = len(getattr(self.vec, "vocabulary_", {}) or {})
        return {
            "type": "tfidf_lsa",
            "vocab_size": vocab_size,
            "n_components": getattr(self.svd, "n_components", self.n_components),
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "degenerate": bool(self._degenerate),
        }
