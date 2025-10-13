# backend/src/neurocampus/models/strategies/modelo_rbm_restringida.py
# Reusa la implementación "general" cambiando defaults (p.ej., más unidades ocultas / más pasos CD).
# Mantiene la misma API: setup(..), train_step(..), predict_proba/predict, save/load.

from __future__ import annotations
from typing import Dict, Tuple, Optional, List

from .modelo_rbm_general import RBMGeneral


class RBMRestringida(RBMGeneral):
    """
    RBM 'restringida' (visible<->oculta bipartita) para el Student.
    Hereda de RBMGeneral y ajusta hiperparámetros por defecto para un
    entrenamiento un poco más "profundo" (más hidden/steps).
    """
    def __init__(self) -> None:
        super().__init__()

    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        # Ajustes por defecto (pueden sobreescribirse en hparams):
        defaults = {
            "n_hidden": hparams.get("n_hidden", 64),
            "cd_k": hparams.get("cd_k", 2),
            "epochs_rbm": hparams.get("epochs_rbm", 2),
            "lr_rbm": hparams.get("lr_rbm", 5e-3),   # ligeramente más bajo
            "lr_head": hparams.get("lr_head", 1e-2),
        }
        merged = {**hparams, **defaults}
        super().setup(data_ref=data_ref, hparams=merged)