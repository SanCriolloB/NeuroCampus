import json, joblib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .base import TextFeaturizer

class TfidfLSAFeaturizer(TextFeaturizer):
    def __init__(self, max_features=20000, n_components=64, ngram_range=(1,2), min_df=3):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, texts: List[str]):
        X = self.vec.fit_transform(texts)
        self.svd.fit(X)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        Z = self.svd.transform(X)
        return Z.astype(np.float32)

    def save(self, path: str) -> None:
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, p / "tfidf.joblib")
        joblib.dump(self.svd, p / "svd.joblib")
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TfidfLSAFeaturizer":
        p = Path(path)
        obj = cls()
        obj.vec = joblib.load(p / "tfidf.joblib")
        obj.svd = joblib.load(p / "svd.joblib")
        return obj

    @property
    def meta(self) -> Dict[str, Any]:
        return {"type":"tfidf_lsa", "n_components": self.svd.n_components, "vocab_size": len(self.vec.vocabulary_)}
