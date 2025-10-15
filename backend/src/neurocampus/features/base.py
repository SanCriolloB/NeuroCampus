from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class TextFeaturizer(ABC):
    @abstractmethod
    def fit(self, texts: List[str]) -> "TextFeaturizer": ...
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "TextFeaturizer": ...
    @property
    @abstractmethod
    def meta(self) -> Dict[str, Any]: ...
