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
