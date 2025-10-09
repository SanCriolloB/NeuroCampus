from typing import Dict, Any, Tuple
import random

class RBMRestringida:
    """RBM 'restringida' (visible↔oculta bipartita). TODO: implementar real.
    Mantener el mismo contrato que RBMGeneral para intercambiabilidad."""
    def __init__(self) -> None:
        self.hparams: Dict[str, Any] = {}

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        self.hparams = hparams
        # TODO: inicializar pesos/sesgos y preprocesamiento

    def train_step(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        loss = max(0.01, 0.8 / epoch) * random.uniform(0.9, 1.1)  # simulado
        metrics = {"recon_error": loss, "epoch": float(epoch)}
        return loss, metrics