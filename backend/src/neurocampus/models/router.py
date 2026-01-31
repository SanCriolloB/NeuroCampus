"""neurocampus.models.router

Registry de estrategias/modelos usados por el backend.

El backend (FastAPI) usa ``MODELOS_REGISTRY`` para instanciar el strategy correcto
cuando llega un request de entrenamiento/predicción.

Se usa un import lazy y defensivo para evitar que un fallo de import tumbe el
arranque completo en entornos parciales (p. ej. algunos unit tests).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


def _build_registry() -> Dict[str, Type[Any]]:
    # Imports locales (evitan ciclos al importarse neurocampus.models.__init__)
    from .strategies.modelo_rbm_general import RBMGeneral
    from .strategies.modelo_rbm_restringida import RBMRestringida

    return {
        "rbm_general": RBMGeneral,
        "rbm_restringida": RBMRestringida,
    }


try:
    MODELOS_REGISTRY: Dict[str, Type[Any]] = _build_registry()
except Exception as e:  # pragma: no cover
    logger.exception("No se pudo construir MODELOS_REGISTRY: %s", e)
    MODELOS_REGISTRY = {}
