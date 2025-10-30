# backend/src/neurocampus/observability/eventos_prediccion.py
"""
Constantes y helpers para eventos de predicción.
No implementa predicción; solo estandariza nombres y payloads, y valida su forma.
"""

from __future__ import annotations

from typing import Dict, Optional
from datetime import datetime, timezone

from .bus_eventos import publicador  # ya existe y registra en log_handler (Día 4)
from .payloads_prediccion import (
    PredRequested,
    PredCompleted,
    PredFailed,
)

# Nombres canónicos de eventos
EV_PRED_REQUESTED = "prediction.requested"
EV_PRED_COMPLETED = "prediction.completed"
EV_PRED_FAILED    = "prediction.failed"


def _now_utc() -> datetime:
    """Datetime actual en UTC con tzinfo."""
    return datetime.now(timezone.utc)


def emit_requested(correlation_id: str, family: str, mode: str, n_items: int) -> None:
    """
    Emite `prediction.requested` validando el payload con Pydantic.
    - correlation_id: UUID v4 de la solicitud (string).
    - family: nombre de familia/ensamble de modelo (string).
    - mode: "online" | "batch".
    - n_items: número de registros (>=1).
    """
    payload = PredRequested(
        correlation_id=correlation_id,
        family=family,
        mode=mode,              # validado: "online" | "batch"
        n_items=n_items,
        ts=_now_utc(),          # timestamp de emisión (UTC)
    ).model_dump(mode="json", exclude_none=True)

    publicador(EV_PRED_REQUESTED, payload)


def emit_completed(
    correlation_id: str,
    latencia_ms: int,
    n_items: int,
    distribucion_labels: Dict[str, int] | None = None,
    distribucion_sentiment: Dict[str, float] | None = None,
) -> None:
    """
    Emite `prediction.completed` validando el payload con Pydantic.
    - correlation_id: mismo del ciclo.
    - latencia_ms: latencia total del request/lote en servidor.
    - n_items: cantidad de registros procesados (>=1).
    - distribucion_labels: conteo por etiqueta (opcional).
    - distribucion_sentiment: promedios {pos, neu, neg} (opcional).
    """
    payload = PredCompleted(
        correlation_id=correlation_id,
        latencia_ms=latencia_ms,
        n_items=n_items,
        distribucion_labels=distribucion_labels,
        distribucion_sentiment=distribucion_sentiment,
        ts=_now_utc(),
    ).model_dump(mode="json", exclude_none=True)

    publicador(EV_PRED_COMPLETED, payload)


def emit_failed(
    correlation_id: str,
    error: str,
    stage: Optional[str] = None,
    *,
    error_code: Optional[str] = None,
) -> None:
    """
    Emite `prediction.failed` validando el payload con Pydantic.
    - correlation_id: mismo del ciclo.
    - error: mensaje legible y corto (sin PII).
    - stage: etapa opcional ("vectorize" | "predict" | "postprocess" | "io").
    - error_code: código de error (p.ej. "MODEL_NOT_AVAILABLE"). Si no se provee,
      se usa "INTERNAL_ERROR" para compatibilidad.
    """
    payload = PredFailed(
        correlation_id=correlation_id,
        error=error,
        error_code=error_code or "INTERNAL_ERROR",
        stage=stage,
        ts=_now_utc(),
    ).model_dump(mode="json", exclude_none=True)

    publicador(EV_PRED_FAILED, payload)


__all__ = [
    "EV_PRED_REQUESTED",
    "EV_PRED_COMPLETED",
    "EV_PRED_FAILED",
    "emit_requested",
    "emit_completed",
    "emit_failed",
]
