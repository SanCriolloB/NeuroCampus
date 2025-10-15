# backend/src/neurocampus/models/strategies/modelo_rbm_restringida.py
# Variante "restringida": subclase del general con defaults distintos.
# Compatibilidad hacia atrás: __init__(**kwargs) acepta los argumentos
# que pasan train_rbm/cmd_autoretrain (n_visible, n_hidden, cd_k, etc.)
# y los manda a setup(...). Mantiene alias ModeloRBMRestringida.

from __future__ import annotations

from typing import Dict, Optional, Any

from .modelo_rbm_general import RBMGeneral


class RBMRestringida(RBMGeneral):
    """
    RBM bipartita (visible<->oculta) con defaults algo más "profundos"
    que la general. Cualquier hiperparámetro pasado por el llamador
    (CLI u otro) SOBRESCRIBE los defaults de abajo.
    """

    # --- N O T A ---
    # Para compatibilidad con código que invoca el constructor con kwargs
    # (n_visible, n_hidden, cd_k, lr_rbm, lr_head, momentum, weight_decay, seed, device, ...),
    # aceptamos kwargs y llamamos a setup(...) aquí.
    def __init__(
        self,
        n_visible: Optional[int] = None,
        n_hidden: Optional[int] = None,
        cd_k: Optional[int] = None,
        lr_rbm: Optional[float] = None,
        lr_head: Optional[float] = None,
        momentum: Optional[float] = None,
        weight_decay: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        **extra: Any,
    ) -> None:
        # Construimos dict desde los kwargs no-nulos:
        incoming: Dict[str, Any] = {}
        for k, v in dict(
            n_visible=n_visible, n_hidden=n_hidden, cd_k=cd_k,
            lr_rbm=lr_rbm, lr_head=lr_head,
            momentum=momentum, weight_decay=weight_decay,
            seed=seed, device=device,
        ).items():
            if v is not None:
                incoming[k] = v
        incoming.update(extra or {})

        # Defaults propios de la variante restringida:
        defaults: Dict[str, Any] = {
            "n_hidden": 96,
            "cd_k": 2,
            "epochs_rbm": 2,
            "lr_rbm": 5e-3,
            "lr_head": 1e-2,
        }

        # Merge correcto: hparams del llamador pisan defaults:
        merged = {**defaults, **incoming}

        # Llamamos al setup del padre (no usamos super().__init__ para
        # evitar un setup prematuro si el padre también lo hiciera).
        self.setup(data_ref=None, hparams=merged)

    def setup(self, data_ref: Optional[str], hparams: Dict) -> None:
        # Aseguramos defaults si setup se invoca externamente después
        # del __init__: mismo merge que arriba.
        defaults = {
            "n_hidden": 96,
            "cd_k": 2,
            "epochs_rbm": 2,
            "lr_rbm": 5e-3,
            "lr_head": 1e-2,
        }
        merged = {**defaults, **(hparams or {})}
        super().setup(data_ref=data_ref, hparams=merged)


# Alias legacy para compatibilidad (importaciones antiguas):
class ModeloRBMRestringida(RBMRestringida):
    pass


__all__ = ["RBMRestringida", "ModeloRBMRestringida"]
