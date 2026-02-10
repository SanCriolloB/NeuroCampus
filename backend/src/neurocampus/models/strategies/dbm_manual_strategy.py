# backend/src/neurocampus/models/strategies/dbm_manual_strategy.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

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

    def fit(self, df: pd.DataFrame) -> "DBMManualStrategy":
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


class DBMManualPlantillaStrategy:
    """
    Strategy DBM compatible con PlantillaEntrenamiento (setup/train_step).

    - Entrenamiento greedy: 1 epoch de rbm_v_h1 + 1 epoch de rbm_h1_h2 por train_step.
    - Reporta recon_error (loss) para graficación UI.
    """

    def __init__(self) -> None:
        self.model: Optional[DBMManual] = None
        self.X: Optional[np.ndarray] = None
        self.batch_size: int = 64
        self.eval_rows: int = 2048
        self._rng = np.random.default_rng(42)

    def reset(self) -> None:
        self.model = None
        self.X = None

    def _load_df(self, data_ref: str) -> pd.DataFrame:
        if not data_ref:
            raise ValueError("DBMManualPlantillaStrategy: data_ref vacío")
        if not os.path.exists(data_ref):
            raise FileNotFoundError(data_ref)

        ext = os.path.splitext(data_ref)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(data_ref)
        if ext in (".csv", ".txt"):
            return pd.read_csv(data_ref)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(data_ref)
        raise ValueError(f"DBMManualPlantillaStrategy: formato no soportado: {ext}")

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = (
            df.select_dtypes(include=[np.number])
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
              .to_numpy(dtype=np.float32)
        )
        if X.shape[1] == 0:
            raise ValueError("DBMManualPlantillaStrategy: no hay columnas numéricas para entrenar.")
        return X

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        df = self._load_df(str(data_ref))
        X = self._numeric_matrix(df)

        # Hparams
        n_hidden1 = int(hparams.get("n_hidden1", 64) or 64)
        n_hidden2 = int(hparams.get("n_hidden2", 32) or 32)
        lr = float(hparams.get("lr", 0.01) or 0.01)
        cd_k = int(hparams.get("cd_k", 1) or 1)
        self.batch_size = int(hparams.get("batch_size", 64) or 64)

        seed = int(hparams.get("seed", 42) or 42)
        l2 = float(hparams.get("l2", 0.0) or 0.0)
        clip_grad = hparams.get("clip_grad", 1.0)
        clip_grad = None if clip_grad is None else float(clip_grad)

        binarize_input = bool(hparams.get("binarize_input", False))
        input_bin_threshold = float(hparams.get("input_bin_threshold", 0.5) or 0.5)
        use_pcd = bool(hparams.get("use_pcd", False))

        self.eval_rows = int(hparams.get("eval_rows", 2048) or 2048)
        self._rng = np.random.default_rng(seed)

        self.model = DBMManual(
            n_visible=X.shape[1],
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            lr=lr,
            cd_k=cd_k,
            seed=seed,
            l2=l2,
            clip_grad=clip_grad,
            binarize_input=binarize_input,
            input_bin_threshold=input_bin_threshold,
            use_pcd=use_pcd,
        )
        self.X = X

    def train_step(self, epoch: int, hparams: Dict[str, Any], y: Any = None) -> Dict[str, Any]:
        if self.model is None or self.X is None:
            raise RuntimeError("DBMManualPlantillaStrategy: falta setup(data_ref, hparams)")

        # 1 epoch layer1
        self.model.rbm_v_h1.fit(self.X, epochs=1, batch_size=self.batch_size, verbose=0)
        H1 = self.model.rbm_v_h1.transform(self.X)

        # 1 epoch layer2
        self.model.rbm_h1_h2.fit(H1, epochs=1, batch_size=self.batch_size, verbose=0)

        # Recon error (muestra para no encarecer)
        n = self.X.shape[0]
        m = min(self.eval_rows, n)
        if m <= 0:
            m = min(256, n)

        idx = self._rng.choice(n, size=m, replace=False) if n > m else np.arange(n)
        Xs = self.X[idx]

        v1_rec = self.model.rbm_v_h1.reconstruct(Xs)
        mse1 = float(np.mean((Xs - v1_rec) ** 2))

        H1s = self.model.rbm_v_h1.transform(Xs)
        h1_rec = self.model.rbm_h1_h2.reconstruct(H1s)
        mse2 = float(np.mean((H1s - h1_rec) ** 2))

        recon = float((mse1 + mse2) / 2.0)

        return {
            "epoch": float(epoch),
            "loss": recon,
            "recon_error": recon,
            "recon_error_layer1": mse1,
            "recon_error_layer2": mse2,
        }
