# backend/src/neurocampus/observability/__init__.py
"""
Paquete de observabilidad.
Exporta helpers y constantes de eventos de predicción.
"""

from .eventos_prediccion import (
    EV_PRED_REQUESTED,
    EV_PRED_COMPLETED,
    EV_PRED_FAILED,
    emit_requested,
    emit_completed,
    emit_failed,
)

__all__ = [
    "EV_PRED_REQUESTED",
    "EV_PRED_COMPLETED",
    "EV_PRED_FAILED",
    "emit_requested",
    "emit_completed",
    "emit_failed",
]
