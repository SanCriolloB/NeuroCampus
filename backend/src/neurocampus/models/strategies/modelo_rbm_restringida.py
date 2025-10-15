# backend/src/neurocampus/models/strategies/modelo_rbm_restringida.py
# Subclase fina sobre RBMGeneral que ajusta hiperparámetros por defecto.
# Mantiene la misma API/contrato: setup(..), train_step(..),
# predict_proba(..), predict(..), save(..), load(..).

from __future__ import annotations

from typing import Dict, Optional

from .modelo_rbm_general import RBMGeneral


class RBMRestringida(RBMGeneral):
    """
    Variante 'restringida' (visible<->oculta bipartita) que reutiliza la
    implementación de RBMGeneral, pero con defaults algo más 'profundos':
      - más unidades ocultas,
      - más pasos de Contrastive Divergence,
      - más épocas base para el bloque RBM,
      - y un lr_rbm por defecto un poco más conservador.

    Cualquier hiperparámetro explícito en `hparams` **sobrescribe** a los
    defaults definidos aquí.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        # Defaults de esta variante. OJO: estos son valores base que se
        # utilizarán solo si NO vienen en `hparams`. Si el llamador pasa
        # un valor (vía CLI o llamada directa), ese valor tendrá prioridad.
        defaults = {
            "n_hidden": 96,       # más grande que el general (p.ej., 64)
            "cd_k": 2,            # pasos de CD > 1
            "epochs_rbm": 2,      # algunas épocas extra
            "lr_rbm": 5e-3,       # ligeramente conservador
            "lr_head": 1e-2,
            # El resto (batch_size, momentum, weight_decay, etc.) lo hereda
            # RBMGeneral con sus propios defaults / lo que venga en hparams.
        }

        # Merge correcto: primero defaults, luego hparams para que hparams
        # tenga prioridad sobre los defaults.
        merged = {**defaults, **(hparams or {})}

        # Delegamos toda la inicialización en el padre.
        super().setup(data_ref=data_ref, hparams=merged)


# ---------------------------------------------------------------------------
# Alias de compatibilidad (por si quedaran importaciones antiguas).
#    from neurocampus.models.strategies.modelo_rbm_restringida import ModeloRBMRestringida
# ---------------------------------------------------------------------------
class ModeloRBMRestringida(RBMRestringida):
    """Alias legacy para compatibilidad retrocompatible."""
    pass


__all__ = ["RBMRestringida", "ModeloRBMRestringida"]
