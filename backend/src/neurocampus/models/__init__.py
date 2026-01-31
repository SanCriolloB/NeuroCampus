"""neurocampus.models

API pública del paquete de modelos.

Este paquete es consumido por:
- el backend (FastAPI) a través de ``MODELOS_REGISTRY`` (ver ``neurocampus.models.router``)
- notebooks / scripts (re-exports de estrategias y modelos manuales)

Notas:
- Los re-exports se protegen con try/except para evitar fallos en importaciones parciales.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registry requerido por el backend
# ---------------------------------------------------------------------------

try:
    # Requerido por backend/src/neurocampus/app/routers/modelos.py (y otros).
    from .router import MODELOS_REGISTRY  # type: ignore
except Exception:  # pragma: no cover
    # Fallback defensivo: el backend debería fallar de forma explícita si necesita el registry
    # y no puede importarse; pero para entornos parciales (unit tests / lint) preferimos no romper.
    MODELOS_REGISTRY = {}  # type: ignore


# ---------------------------------------------------------------------------
# Modelos manuales (legacy / utilitarios)
# ---------------------------------------------------------------------------

try:  # pragma: no cover
    from .rbm_manual import RBMManualStrategy
    from .bm_manual import BMManualStrategy
    from .dbm_manual import DBMManualStrategy
except Exception:  # pragma: no cover
    RBMManualStrategy = None  # type: ignore
    BMManualStrategy = None  # type: ignore
    DBMManualStrategy = None  # type: ignore


# ---------------------------------------------------------------------------
# Re-exports de estrategias principales (RBM General / Restringida)
# ---------------------------------------------------------------------------

try:  # pragma: no cover
    from .strategies.modelo_rbm_general import RBMGeneral, ModeloRBMGeneral
    from .strategies.modelo_rbm_restringida import RBMRestringida, ModeloRBMRestringida
except Exception:  # pragma: no cover
    RBMGeneral = None  # type: ignore
    ModeloRBMGeneral = None  # type: ignore
    RBMRestringida = None  # type: ignore
    ModeloRBMRestringida = None  # type: ignore


__all__ = [
    "MODELOS_REGISTRY",
    "RBMManualStrategy",
    "BMManualStrategy",
    "DBMManualStrategy",
    "RBMGeneral",
    "ModeloRBMGeneral",
    "RBMRestringida",
    "ModeloRBMRestringida",
]
