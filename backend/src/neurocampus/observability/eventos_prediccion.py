# backend/src/neurocampus/observability/eventos_prediccion.py
"""
Constantes y helpers para eventos de predicción.
No implementa predicción; solo estandariza nombres y payloads.
"""

from typing import Dict, Any
from .bus_eventos import publicador  # ya existe y registra en log_handler (Día 4)

EV_PRED_REQUESTED = "prediction.requested"
EV_PRED_COMPLETED = "prediction.completed"
EV_PRED_FAILED    = "prediction.failed"

def emit_requested(correlation_id: str, family: str, mode: str, n_items: int) -> None:
    publicador(EV_PRED_REQUESTED, {
        "correlation_id": correlation_id,
        "family": family,
        "mode": mode,            # "online" | "batch"
        "n_items": n_items
    })

def emit_completed(correlation_id: str, latencia_ms: int, n_items: int,
                   distribucion_labels: Dict[str, int] | None = None,
                   distribucion_sentiment: Dict[str, float] | None = None) -> None:
    publicador(EV_PRED_COMPLETED, {
        "correlation_id": correlation_id,
        "latencia_ms": latencia_ms,
        "n_items": n_items,
        "distribucion_labels": distribucion_labels or {},
        "distribucion_sentiment": distribucion_sentiment or {}
    })

def emit_failed(correlation_id: str, error: str, stage: str | None = None) -> None:
    publicador(EV_PRED_FAILED, {
        "correlation_id": correlation_id,
        "error": error,
        "stage": stage
    })
