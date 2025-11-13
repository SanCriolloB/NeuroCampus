# backend/src/neurocampus/trainers/rbm_trainer.py
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from neurocampus.models.rbm_manual import RBMManual

Callback = Callable[[int, Dict[str, float]], None]

class RBMTrainer:
    def __init__(
        self,
        model: RBMManual,
        out_dir: str,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 5,
    ):
        self.model = model
        self.out_dir = Path(out_dir)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.callbacks: List[Callback] = []
        self.best_metric: float | None = None
        self.best_epoch: int = -1
        self.history: List[Dict[str, float]] = []

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_callback(self, cb: Callback) -> None:
        self.callbacks.append(cb)

    def _run_callbacks(self, epoch: int, metrics: Dict[str, float]):
        for cb in self.callbacks:
            cb(epoch, metrics)

    def _save_metrics(self):
        metrics_path = self.out_dir / "metrics_rbm.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            # shuffle
            idx = np.random.permutation(n_samples)
            X_shuff = X[idx]

            mse_epoch = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch = X_shuff[start:end]
                if batch.shape[0] == 0:
                    continue
                metrics = self.model.train_batch(batch)
                mse_epoch += metrics["mse_recon"]
                n_batches += 1

            mse_epoch /= max(1, n_batches)
            elapsed = time.time() - t0

            record = {"epoch": epoch, "mse_recon": mse_epoch, "time_sec": elapsed}
            self.history.append(record)
            self._run_callbacks(epoch, record)
            self._save_metrics()

            # Early stopping (minimizar MSE)
            if self.best_metric is None or mse_epoch < self.best_metric:
                self.best_metric = mse_epoch
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
