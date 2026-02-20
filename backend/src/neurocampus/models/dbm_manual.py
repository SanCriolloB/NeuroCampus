# backend/src/neurocampus/models/dbm_manual.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .rbm_manual import RBMManual


class DBMManual:
    """
    DBM simple con 2 capas ocultas, usando pre-entrenamiento
    greedy de RBMs encadenadas.

    Persistencia
    ------------
    - ``save(out_dir)``  →  escribe ``dbm_state.npz`` + ``meta.json``
    - ``load(in_dir)``   →  reconstruye instancia desde esos archivos
    """

    # Versión del schema de persistencia (bump si cambias el formato)
    _SCHEMA_VERSION = 1

    def __init__(
        self,
        n_visible: int,
        n_hidden1: int,
        n_hidden2: int,
        lr: float = 0.01,
        cd_k: int = 1,
        *,
        seed: int = 42,
        l2: float = 0.0,
        clip_grad: float | None = 1.0,
        binarize_input: bool = False,
        input_bin_threshold: float = 0.5,
        use_pcd: bool = False,
    ):
        self.n_visible = int(n_visible)
        self.n_hidden1 = int(n_hidden1)
        self.n_hidden2 = int(n_hidden2)

        # Guardamos por si luego se usan en fases de fine-tuning
        self.lr = float(lr)
        self.cd_k = int(cd_k)

        self.seed = int(seed)
        self.l2 = float(l2)
        self.clip_grad = None if clip_grad is None else float(clip_grad)
        self.binarize_input = bool(binarize_input)
        self.input_bin_threshold = float(input_bin_threshold)
        self.use_pcd = bool(use_pcd)

        # RBMManual (RestrictedBoltzmannMachine) soporta cd_k/use_pcd/seed/etc.
        self.rbm_v_h1 = RBMManual(
            n_visible=self.n_visible,
            n_hidden=self.n_hidden1,
            learning_rate=self.lr,
            seed=self.seed,
            l2=self.l2,
            clip_grad=self.clip_grad,
            binarize_input=self.binarize_input,
            input_bin_threshold=self.input_bin_threshold,
            cd_k=self.cd_k,
            use_pcd=self.use_pcd,
        )

        # Para h1->h2: los "visibles" son probs en [0,1], normalmente NO binarizamos.
        self.rbm_h1_h2 = RBMManual(
            n_visible=self.n_hidden1,
            n_hidden=self.n_hidden2,
            learning_rate=self.lr,
            seed=self.seed + 1,
            l2=self.l2,
            clip_grad=self.clip_grad,
            binarize_input=False,
            input_bin_threshold=0.5,
            cd_k=self.cd_k,
            use_pcd=self.use_pcd,
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, out_dir: str, *, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Guarda el estado del DBM en ``out_dir``.

        Archivos creados:
        - ``dbm_state.npz``: arrays W/bv/bh de las dos RBMs.
        - ``meta.json``: dimensiones, hparams y metadata adicional.

        Parámetros
        ----------
        out_dir:
            Directorio destino (se crea si no existe).
        extra_meta:
            Dict adicional que se mezcla en ``meta.json`` (ej. feat_cols_,
            task_type, target_col, etc.).
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Arrays numpy de ambas RBMs
        np.savez(
            out_path / "dbm_state.npz",
            W1=self.rbm_v_h1.W,
            bv1=self.rbm_v_h1.bv,
            bh1=self.rbm_v_h1.bh,
            W2=self.rbm_h1_h2.W,
            bv2=self.rbm_h1_h2.bv,
            bh2=self.rbm_h1_h2.bh,
        )

        meta: Dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "n_visible": self.n_visible,
            "n_hidden1": self.n_hidden1,
            "n_hidden2": self.n_hidden2,
            "hparams": {
                "lr": self.lr,
                "cd_k": self.cd_k,
                "seed": self.seed,
                "l2": self.l2,
                "clip_grad": self.clip_grad,
                "binarize_input": self.binarize_input,
                "input_bin_threshold": self.input_bin_threshold,
                "use_pcd": self.use_pcd,
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(out_path / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    @classmethod
    def load(cls, in_dir: str) -> "DBMManual":
        """
        Carga un DBMManual desde un directorio guardado con ``save()``.

        Lanza
        -----
        FileNotFoundError:
            Si falta ``dbm_state.npz`` o ``meta.json``.
        ValueError:
            Si el schema es incompatible.
        """
        in_path = Path(in_dir)
        npz_path = in_path / "dbm_state.npz"
        meta_path = in_path / "meta.json"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"DBMManual.load: no se encontró dbm_state.npz en '{in_dir}'"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"DBMManual.load: no se encontró meta.json en '{in_dir}'"
            )

        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        hp = meta.get("hparams", {})
        obj = cls(
            n_visible=int(meta["n_visible"]),
            n_hidden1=int(meta["n_hidden1"]),
            n_hidden2=int(meta["n_hidden2"]),
            lr=float(hp.get("lr", 0.01)),
            cd_k=int(hp.get("cd_k", 1)),
            seed=int(hp.get("seed", 42)),
            l2=float(hp.get("l2", 0.0)),
            clip_grad=hp.get("clip_grad", 1.0),
            binarize_input=bool(hp.get("binarize_input", False)),
            input_bin_threshold=float(hp.get("input_bin_threshold", 0.5)),
            use_pcd=bool(hp.get("use_pcd", False)),
        )

        arrays = np.load(npz_path)
        obj.rbm_v_h1.W  = arrays["W1"].astype(np.float32)
        obj.rbm_v_h1.bv = arrays["bv1"].astype(np.float32)
        obj.rbm_v_h1.bh = arrays["bh1"].astype(np.float32)
        obj.rbm_h1_h2.W  = arrays["W2"].astype(np.float32)
        obj.rbm_h1_h2.bv = arrays["bv2"].astype(np.float32)
        obj.rbm_h1_h2.bh = arrays["bh2"].astype(np.float32)

        return obj

    def copy_weights_from(self, other: "DBMManual") -> None:
        """
        Copia los pesos de ``other`` a ``self`` (para warm start in-place).

        Solo se aplica si las dimensiones coinciden; lanza ``ValueError`` si no.
        """
        if (
            self.rbm_v_h1.W.shape != other.rbm_v_h1.W.shape
            or self.rbm_h1_h2.W.shape != other.rbm_h1_h2.W.shape
        ):
            raise ValueError(
                f"DBMManual.copy_weights_from: dimensiones incompatibles. "
                f"self=({self.rbm_v_h1.W.shape}, {self.rbm_h1_h2.W.shape}) "
                f"other=({other.rbm_v_h1.W.shape}, {other.rbm_h1_h2.W.shape})"
            )
        self.rbm_v_h1.W  = other.rbm_v_h1.W.copy()
        self.rbm_v_h1.bv = other.rbm_v_h1.bv.copy()
        self.rbm_v_h1.bh = other.rbm_v_h1.bh.copy()
        self.rbm_h1_h2.W  = other.rbm_h1_h2.W.copy()
        self.rbm_h1_h2.bv = other.rbm_h1_h2.bv.copy()
        self.rbm_h1_h2.bh = other.rbm_h1_h2.bh.copy()

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def pretrain(self, X: np.ndarray, epochs: int = 10, batch_size: int = 64):
        # Preentrenar primera RBM
        from neurocampus.trainers.rbm_trainer import RBMTrainer

        trainer1 = RBMTrainer(
            self.rbm_v_h1,
            out_dir="reports/dbm_layer1",
            max_epochs=epochs,
            batch_size=batch_size,
        )
        trainer1.fit(X)

        # Transformar datos a espacio de h1
        H1 = self.rbm_v_h1.transform(X)

        trainer2 = RBMTrainer(
            self.rbm_h1_h2,
            out_dir="reports/dbm_layer2",
            max_epochs=epochs,
            batch_size=batch_size,
        )
        trainer2.fit(H1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        h1 = self.rbm_v_h1.transform(X)
        h2 = self.rbm_h1_h2.transform(h1)
        return h2
