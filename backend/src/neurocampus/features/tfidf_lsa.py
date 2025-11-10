# backend/src/neurocampus/features/tfidf_lsa.py
import json
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .base import TextFeaturizer


class TfidfLSAFeaturizer(TextFeaturizer):
    """
    Featurizer TF-IDF + LSA (SVD truncado).

    Parámetros:
        max_features: tamaño máximo del vocabulario TF-IDF.
        n_components: dimensiones de la proyección LSA.
        ngram_range: rango de n-gramas para TF-IDF.
        min_df: frecuencia mínima (absoluta o fracción) para incluir un término.
        max_df: frecuencia máxima (absoluta o fracción) para incluir un término (opcional).

    Notas:
        - Devuelve arrays float32.
        - Provee fit(), transform() y fit_transform().
        - save(path) / load(path) guardan y restauran vectorizador y SVD, además de un meta.json.
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
        self.min_df = min_df
        self.max_df = max_df

        vec_kwargs: Dict[str, Any] = dict(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        if max_df is not None:
            vec_kwargs["max_df"] = max_df

        self.vec: TfidfVectorizer = TfidfVectorizer(**vec_kwargs)
        self.svd: TruncatedSVD = TruncatedSVD(n_components=n_components, random_state=42)

    # -------------------------
    # API principal
    # -------------------------
    def fit(self, texts: List[str]) -> "TfidfLSAFeaturizer":
        X = self.vec.fit_transform(texts)
        self.svd.fit(X)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        Z = self.svd.transform(X)
        return Z.astype(np.float32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        X = self.vec.fit_transform(texts)
        Z = self.svd.fit_transform(X)
        return Z.astype(np.float32)

    # -------------------------
    # Persistencia
    # -------------------------
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

        # Carga objetos entrenados
        vec: TfidfVectorizer = joblib.load(p / "tfidf.joblib")
        svd: TruncatedSVD = joblib.load(p / "svd.joblib")

        # Intenta recuperar hiperparámetros desde meta.json si existe
        max_features = 20000
        n_components = getattr(svd, "n_components", 64)
        ngram_range = (1, 2)
        min_df = 3
        max_df: Optional[float | int] = None

        meta_file = p / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                max_features = meta.get("max_features", max_features)
                n_components = meta.get("n_components", n_components)
                ngram_range = tuple(meta.get("ngram_range", ngram_range))  # type: ignore[arg-type]
                min_df = meta.get("min_df", min_df)
                max_df = meta.get("max_df", max_df)
            except Exception:
                pass

        # Instancia con los mismos hiperparámetros, pero sustituye vec y svd por los cargados
        obj = cls(
            max_features=max_features,
            n_components=n_components,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        obj.vec = vec
        obj.svd = svd
        return obj

    # -------------------------
    # Metadatos
    # -------------------------
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
        }
