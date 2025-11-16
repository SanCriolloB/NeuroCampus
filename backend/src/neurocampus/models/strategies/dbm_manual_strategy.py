# backend/src/neurocampus/models/strategies/dbm_manual_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from neurocampus.models.dbm_manual import DBMManual


class DBMManualStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.model: DBMManual | None = None

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convierte el DataFrame en una matriz numpy de float32 usando solo columnas numéricas.
        """
        X = (
            df.select_dtypes(include=[np.number])
              .fillna(0.0)
              .to_numpy(dtype=np.float32)
        )

        if X.shape[1] == 0:
            raise ValueError(
                "DBMManualStrategy: el DataFrame no contiene columnas numéricas para entrenar."
            )

        return X

    def fit(self, df: pd.DataFrame) -> DBMManualStrategy:
        X = self._numeric_matrix(df)

        n_visible = X.shape[1]
        n_hidden1 = self.config.get("n_hidden1", 64)
        n_hidden2 = self.config.get("n_hidden2", 32)
        lr = self.config.get("lr", 0.01)
        cd_k = self.config.get("cd_k", 1)
        epochs = self.config.get("epochs", 10)
        batch_size = self.config.get("batch_size", 64)

        self.model = DBMManual(
            n_visible=n_visible,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            lr=lr,
            cd_k=cd_k,
        )
        self.model.pretrain(X, epochs=epochs, batch_size=batch_size)

        return self

    def transform(self, df: pd.DataFrame):
        assert self.model is not None, "Modelo DBM no entrenado"
        X = self._numeric_matrix(df)
        H = self.model.transform(X)
        return H
