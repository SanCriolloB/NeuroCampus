# backend/src/neurocampus/models/strategies/dbm_manual_strategy.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from ..utils.metrics import accuracy as _accuracy, f1_macro as _f1_macro, confusion_matrix as _confusion_matrix
import pandas as pd

from neurocampus.models.dbm_manual import DBMManual

logger = logging.getLogger(__name__)


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
    - Para regression: eval supervisada con ridge sobre embeddings latentes (sin leakage).
    """

    def __init__(self) -> None:
        self.model: Optional[DBMManual] = None

        # X usado para entrenar DBM (para regression será X_tr; para otros, X_all)
        self.X: Optional[np.ndarray] = None

        # Para debugging / UI / consistencia
        self.feat_cols_: List[str] = []

        # Para regression
        self.task_type_: str = "unsupervised"
        self.target_col_: Optional[str] = None
        self.target_scale_: float = 50.0
        self.split_mode_: str = "random"
        self.val_ratio_: float = 0.2
        self.seed_: int = 42
        self.ridge_l2_: float = 1e-3

        self.X_tr: Optional[np.ndarray] = None
        self.X_va: Optional[np.ndarray] = None
        self.y_tr: Optional[np.ndarray] = None  # escalada 0..1
        self.y_va: Optional[np.ndarray] = None  # escalada 0..1

        self.batch_size: int = 64
        self.eval_rows: int = 2048
        self._rng = np.random.default_rng(42)

        # Trazabilidad warm start (se llena en setup / _try_warm_start)
        self._warm_start_info_: Dict[str, Any] = {"warm_start": "skipped"}

    def reset(self) -> None:
        self.model = None
        self.X = None
        self.feat_cols_ = []

        self.task_type_ = "unsupervised"
        self.target_col_ = None
        self.target_scale_ = 50.0
        self.split_mode_ = "random"
        self.val_ratio_ = 0.2
        self.seed_ = 42
        self.ridge_l2_ = 1e-3

        self.X_tr = None
        self.X_va = None
        self.y_tr = None
        self.y_va = None
        self._warm_start_info_ = {"warm_start": "skipped"}


    # ------------------------------------------------------------------
    # Persistencia y warm start
    # ------------------------------------------------------------------

    def _try_warm_start(self, warm_start_path: str) -> Dict[str, Any]:
        """
        Intenta cargar pesos desde un directorio previo y copiarlos al modelo actual.

        Debe llamarse DESPUÉS de que self.model ya fue instanciado en setup().

        Retorna un dict de trazabilidad:
          - {"warm_start": "ok"}        si los pesos se cargaron correctamente.
          - {"warm_start": "skipped"}   si warm_start_path está vacío.
          - {"warm_start": "error"}     si ocurrió un error (incompatibilidad / missing files).
        """
        info: Dict[str, Any] = {
            "warm_start": "skipped",
            "warm_start_dir": str(warm_start_path),
        }
        if not warm_start_path:
            return info

        try:
            prev = DBMManual.load(str(warm_start_path))
            if self.model is None:
                info["warm_start"] = "error"
                info["error"] = "self.model es None; setup() debe llamarse antes"
                return info

            self.model.copy_weights_from(prev)
            info["warm_start"] = "ok"
            info["n_visible"] = self.model.n_visible
            info["n_hidden1"] = self.model.n_hidden1
            info["n_hidden2"] = self.model.n_hidden2
            logger.info(
                "DBMManualPlantillaStrategy: warm start OK desde %s", warm_start_path
            )
        except (ValueError, FileNotFoundError) as exc:
            # Dimensiones incompatibles o archivos faltantes → marcar error
            info["warm_start"] = "error"
            info["error"] = str(exc)
            logger.warning(
                "DBMManualPlantillaStrategy: warm start falló (%s) — %s",
                type(exc).__name__,
                exc,
            )
            raise  # Re-raise para que _run_training lo capture si fue explícito
        except Exception as exc:
            info["warm_start"] = "error"
            info["error"] = str(exc)
            logger.exception(
                "DBMManualPlantillaStrategy: warm start error inesperado desde %s", warm_start_path
            )
            raise

        return info

    def save(self, out_dir: str) -> None:
        """
        Persiste el estado del DBM en out_dir/.

        Escribe:
        - dbm_state.npz  — pesos W/bv/bh de las dos RBMs.
        - meta.json      — dimensiones, feat_cols_, task/target y hparams.

        Llamado automáticamente por _try_write_predictor_bundle al final del run.
        """
        if self.model is None:
            raise RuntimeError(
                "DBMManualPlantillaStrategy.save: modelo no entrenado (self.model es None)"
            )

        extra: Dict[str, Any] = {
            "feat_cols_": list(self.feat_cols_),
            "task_type": self.task_type_,
            "target_col": self.target_col_,
            "target_scale": float(self.target_scale_),
        }
        if self._warm_start_info_:
            extra["warm_start_info"] = self._warm_start_info_

        self.model.save(out_dir, extra_meta=extra)

        # Defensa: si por alguna razón no quedó meta.json, lo escribimos mínimo
        out_path = Path(out_dir)
        meta_path = out_path / "meta.json"
        if not meta_path.exists():
            meta = {
                "schema_version": 1,
                "n_visible": int(getattr(self.model, "n_visible", 0) or 0),
                "n_hidden1": int(getattr(self.model, "n_hidden1", 0) or 0),
                "n_hidden2": int(getattr(self.model, "n_hidden2", 0) or 0),
                "hparams": {},
                "legacy_repaired": True,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        logger.info("DBMManualPlantillaStrategy: modelo guardado en %s", out_dir)


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

    def _numeric_matrix(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> np.ndarray:
        exclude = set(exclude_cols or [])
        num_df = df.select_dtypes(include=[np.number])

        cols = [c for c in num_df.columns if c not in exclude]
        X = (
            num_df[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        if X.shape[1] == 0:
            raise ValueError("DBMManualPlantillaStrategy: no hay columnas numéricas para entrenar.")

        self.feat_cols_ = cols
        return X

    def _split_train_val_indices(
        self,
        df: pd.DataFrame,
        split_mode: str,
        val_ratio: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(len(df))
        if n <= 1:
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)

        val_ratio = float(val_ratio)
        if val_ratio <= 0.0:
            return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)
        if val_ratio >= 0.9:
            val_ratio = 0.9

        cut = int(np.floor(n * (1.0 - val_ratio)))
        cut = max(1, min(cut, n - 1))  # garantiza >=1 train y >=1 val

        split_mode = (split_mode or "random").lower()
        if split_mode == "temporal" and "periodo" in df.columns:
            # Orden estable por periodo (string). Suficiente para ventanas tipo 2025-1, 2025-2, etc.
            order = np.argsort(df["periodo"].astype(str).to_numpy())
        else:
            rng = np.random.default_rng(int(seed))
            order = rng.permutation(n)

        tr_idx = order[:cut].astype(np.int64, copy=False)
        va_idx = order[cut:].astype(np.int64, copy=False)
        return tr_idx, va_idx


    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        df = self._load_df(str(data_ref))

        # ---- Task meta ----
        self.task_type_ = str(hparams.get("task_type") or "").lower() or "unsupervised"
        self.target_col_ = hparams.get("target_col")
        self.target_scale_ = float(hparams.get("target_scale", 50.0) or 50.0)

        self.split_mode_ = str(hparams.get("split_mode", "random") or "random").lower()
        self.val_ratio_ = float(hparams.get("val_ratio", 0.2) or 0.2)
        self.seed_ = int(hparams.get("seed", 42) or 42)
        self.ridge_l2_ = float(hparams.get("ridge_l2", 1e-3) or 1e-3)

        is_regression = (self.task_type_ == "regression")

        # ---- Para regression: validar target, filtrar nans, split ----
        if is_regression:
            if not self.target_col_:
                raise ValueError("DBMManualPlantillaStrategy(regression): falta target_col en hparams")
            if self.target_col_ not in df.columns:
                raise ValueError(f"DBMManualPlantillaStrategy(regression): target_col no existe en data: {self.target_col_}")

            y_raw = pd.to_numeric(df[self.target_col_], errors="coerce")
            mask = y_raw.notna()
            df = df[mask].reset_index(drop=True)
            y = (y_raw[mask].to_numpy(dtype=np.float32) / float(self.target_scale_))

            tr_idx, va_idx = self._split_train_val_indices(
                df=df,
                split_mode=self.split_mode_,
                val_ratio=self.val_ratio_,
                seed=self.seed_,
            )

            X_all = self._numeric_matrix(df, exclude_cols=[str(self.target_col_)])
            self.X_tr = X_all[tr_idx]
            self.X_va = X_all[va_idx]
            self.y_tr = y[tr_idx]
            self.y_va = y[va_idx]

            # Entrena DBM solo con train (evita leakage temporal)
            self.X = self.X_tr
        elif self.task_type_ == "classification":
            # Clasificación: buscar columna de labels ("label", "sentimiento", "clase")
            label_col = hparams.get("label_col") or next(
                (c for c in ("label", "sentimiento", "clase", "target") if c in df.columns), None
            )
            if label_col and label_col in df.columns:
                from ..utils.metrics import confusion_matrix as _cm_init
                # Construir etiquetas y codificación
                self.labels_ = sorted(df[label_col].dropna().unique().tolist())
                lbl2idx = {l: i for i, l in enumerate(self.labels_)}
                y_cls = np.array([lbl2idx.get(v, -1) for v in df[label_col]], dtype=np.int64)
                mask = y_cls >= 0
                df = df[mask].reset_index(drop=True)
                y_cls = y_cls[mask]
                X_all = self._numeric_matrix(df, exclude_cols=[label_col])
                tr_idx, va_idx = self._split_train_val_indices(
                    df=df, split_mode=self.split_mode_, val_ratio=self.val_ratio_, seed=self.seed_
                )
                self.X = X_all[tr_idx]
                self.X_tr = X_all[tr_idx]
                self.X_va = X_all[va_idx]
                self.y_tr = y_cls[tr_idx]
                self.y_va = y_cls[va_idx]
            else:
                X_all = self._numeric_matrix(df, exclude_cols=None)
                self.X = X_all
                self.X_tr = self.X_va = self.y_tr = self.y_va = None
                self.labels_ = []
        else:
            X_all = self._numeric_matrix(df, exclude_cols=None)
            self.X = X_all
            self.X_tr = None
            self.X_va = None
            self.y_tr = None
            self.y_va = None

        # ---- Hparams DBM ----
        n_hidden1 = int(hparams.get("n_hidden1", 64) or 64)
        n_hidden2 = int(hparams.get("n_hidden2", 32) or 32)
        lr = float(hparams.get("lr", 0.01) or 0.01)
        cd_k = int(hparams.get("cd_k", 1) or 1)
        self.batch_size = int(hparams.get("batch_size", 64) or 64)

        l2 = float(hparams.get("l2", 0.0) or 0.0)
        clip_grad = hparams.get("clip_grad", 1.0)
        clip_grad = None if clip_grad is None else float(clip_grad)

        binarize_input = bool(hparams.get("binarize_input", False))
        input_bin_threshold = float(hparams.get("input_bin_threshold", 0.5) or 0.5)
        use_pcd = bool(hparams.get("use_pcd", False))

        self.eval_rows = int(hparams.get("eval_rows", 2048) or 2048)
        self._rng = np.random.default_rng(self.seed_)

        if self.X is None:
            raise RuntimeError("DBMManualPlantillaStrategy: X no inicializado en setup()")

        self.model = DBMManual(
            n_visible=int(self.X.shape[1]),
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            lr=lr,
            cd_k=cd_k,
            seed=int(self.seed_),
            l2=l2,
            clip_grad=clip_grad,
            binarize_input=binarize_input,
            input_bin_threshold=input_bin_threshold,
            use_pcd=use_pcd,
        )

        # ---- Warm start (si se proveyó warm_start_path) ----
        warm_dir = str(hparams.get("warm_start_path") or "").strip()
        self._warm_start_info_ = {"warm_start": "skipped", "warm_start_dir": warm_dir}
        if warm_dir:
            ws_mode = str(hparams.get("warm_start_from") or "run_id").lower()
            try:
                self._warm_start_info_ = self._try_warm_start(warm_dir)
            except Exception as exc:
                # Si el warm start fue explícito (run_id), re-raise para fallar el job.
                # Si fue "champion" (podría no existir), dejamos skipped+error.
                self._warm_start_info_ = {
                    "warm_start": "error",
                    "warm_start_dir": warm_dir,
                    "error": str(exc),
                }
                if ws_mode == "run_id":
                    raise


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

        out: Dict[str, Any] = {
            "epoch": float(epoch),
            "loss": recon,
            "recon_error": recon,
            "recon_error_layer1": mse1,
            "recon_error_layer2": mse2,
        }

        # Incluir trazabilidad de warm start en la primera época
        if epoch == 1 and hasattr(self, "_warm_start_info_"):
            out["warm_start"] = dict(self._warm_start_info_)

        # ---- Regression eval: ridge head sobre embeddings latentes (solo si hay split/y) ----
        if self.task_type_ == "regression" and self.X_tr is not None and self.X_va is not None and self.y_tr is not None and self.y_va is not None:
            # Latentes
            def _latent(Xa: np.ndarray) -> np.ndarray:
                try:
                    Z = self.model.transform(Xa)
                    return np.asarray(Z, dtype=np.float32)
                except Exception:
                    H1a = self.model.rbm_v_h1.transform(Xa)
                    try:
                        Z2 = self.model.rbm_h1_h2.transform(H1a)
                        return np.asarray(Z2, dtype=np.float32)
                    except Exception:
                        return np.asarray(H1a, dtype=np.float32)

            Ztr = _latent(self.X_tr)
            Zva = _latent(self.X_va)

            # Ridge cerrado: (A^T A + l2 I)w = A^T y
            l2 = float(self.ridge_l2_)
            ytr = np.asarray(self.y_tr, dtype=np.float32).reshape(-1, 1)
            yva = np.asarray(self.y_va, dtype=np.float32).reshape(-1, 1)

            def _ridge_predict(Z_train: np.ndarray, y_train: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
                A_tr = np.concatenate([np.ones((Z_train.shape[0], 1), dtype=np.float32), Z_train], axis=1)
                A_ev = np.concatenate([np.ones((Z_eval.shape[0], 1), dtype=np.float32), Z_eval], axis=1)
                I = np.eye(A_tr.shape[1], dtype=np.float32)
                I[0, 0] = 0.0  # no regularizar bias
                w = np.linalg.solve((A_tr.T @ A_tr) + (l2 * I), (A_tr.T @ y_train))
                return (A_ev @ w).reshape(-1)

            p_tr = _ridge_predict(Ztr, ytr, Ztr)
            p_va = _ridge_predict(Ztr, ytr, Zva)

            # Unscale a escala original (0..50)
            scale = float(self.target_scale_ or 1.0)
            ytr_u = (ytr.reshape(-1) * scale).astype(np.float32)
            yva_u = (yva.reshape(-1) * scale).astype(np.float32)
            p_tr_u = (p_tr * scale).astype(np.float32)
            p_va_u = (p_va * scale).astype(np.float32)

            def _mae(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.mean(np.abs(a - b)))

            def _rmse(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.sqrt(np.mean((a - b) ** 2)))

            def _r2(a: np.ndarray, b: np.ndarray) -> float:
                denom = float(np.sum((a - float(np.mean(a))) ** 2))
                if denom <= 1e-12:
                    return 0.0
                return float(1.0 - (np.sum((a - b) ** 2) / denom))

            out.update({
                "task_type": "regression",
                "target_col": self.target_col_,
                "train_mae": _mae(ytr_u, p_tr_u),
                "train_rmse": _rmse(ytr_u, p_tr_u),
                "train_r2": _r2(ytr_u, p_tr_u),
                "val_mae": _mae(yva_u, p_va_u),
                "val_rmse": _rmse(yva_u, p_va_u),
                "val_r2": _r2(yva_u, p_va_u),
                "n_train": int(len(ytr_u)),
                "n_val": int(len(yva_u)),
                "pred_min": float(np.min(p_va_u)) if p_va_u.size else None,
                "pred_max": float(np.max(p_va_u)) if p_va_u.size else None,
            })

        # ---- Clasificación eval: softmax lineal + accuracy/f1_macro ----
        if self.task_type_ == "classification" and self.X_tr is not None and self.y_tr is not None:
            labels_list = list(getattr(self, "labels_", []))
            n_classes = max(int(np.max(self.y_tr)) + 1 if len(self.y_tr) > 0 else 2, len(labels_list))

            def _latent_np(Xa: np.ndarray) -> np.ndarray:
                H1a = self.model.rbm_v_h1.transform(Xa)
                return np.asarray(self.model.rbm_h1_h2.transform(H1a), dtype=np.float32)

            Ztr = _latent_np(self.X_tr)
            A_tr = np.concatenate([np.ones((Ztr.shape[0], 1), dtype=np.float32), Ztr], axis=1)
            l2c = float(getattr(self, "ridge_l2_", 1e-3))

            # One-vs-rest: una columna de pesos por clase
            W = np.zeros((A_tr.shape[1], n_classes), dtype=np.float32)
            I = np.eye(A_tr.shape[1], dtype=np.float32); I[0,0] = 0.0
            ATA = (A_tr.T @ A_tr) + l2c * I
            for c in range(n_classes):
                yc = (self.y_tr == c).astype(np.float32).reshape(-1, 1)
                W[:, c] = np.linalg.solve(ATA, A_tr.T @ yc).reshape(-1)

            logits_tr = A_tr @ W
            preds_tr = logits_tr.argmax(axis=1).astype(int)
            y_tr_int = self.y_tr.astype(int) if isinstance(self.y_tr, np.ndarray) else np.asarray(self.y_tr, int)
            out.update({
                "task_type": "classification",
                "labels": labels_list,
                "n_classes": n_classes,
                "n_train": int(len(y_tr_int)),
                "accuracy": float(_accuracy(y_tr_int, preds_tr)),
                "f1_macro": float(_f1_macro(y_tr_int, preds_tr, n_classes)),
            })
            if self.X_va is not None and self.y_va is not None:
                Zva = _latent_np(self.X_va)
                A_va = np.concatenate([np.ones((Zva.shape[0], 1), dtype=np.float32), Zva], axis=1)
                logits_va = A_va @ W
                preds_va = logits_va.argmax(axis=1).astype(int)
                y_va_int = self.y_va.astype(int) if isinstance(self.y_va, np.ndarray) else np.asarray(self.y_va, int)
                out.update({
                    "n_val": int(len(y_va_int)),
                    "val_accuracy": float(_accuracy(y_va_int, preds_va)),
                    "val_f1_macro": float(_f1_macro(y_va_int, preds_va, n_classes)),
                    "confusion_matrix": _confusion_matrix(y_va_int, preds_va, n_classes),
                })

        return out

