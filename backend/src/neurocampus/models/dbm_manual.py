# backend/src/neurocampus/models/dbm_manual.py
from __future__ import annotations

import numpy as np

from .rbm_manual import RBMManual


class DBMManual:
    """
    DBM simple con 2 capas ocultas, usando pre-entrenamiento
    greedy de RBMs encadenadas.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden1: int,
        n_hidden2: int,
        lr: float = 0.01,
        cd_k: int = 1,
    ):
        self.n_visible = n_visible
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

        # Guardamos por si luego se usan en fases de fine-tuning
        self.lr = lr
        self.cd_k = cd_k

        # RBMManual es un alias de RestrictedBoltzmannMachine,
        # cuya firma es:
        # __init__(n_visible, n_hidden=64, learning_rate=0.05, seed=42,
        #          l2=0.0, clip_grad=1.0, binarize_input=False, input_bin_threshold=0.5)
        self.rbm_v_h1 = RBMManual(
            n_visible=n_visible,
            n_hidden=n_hidden1,
            learning_rate=lr,
        )

        self.rbm_h1_h2 = RBMManual(
            n_visible=n_hidden1,
            n_hidden=n_hidden2,
            learning_rate=lr,
        )

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
