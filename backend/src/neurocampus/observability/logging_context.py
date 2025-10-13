# backend/src/neurocampus/observability/logging_context.py
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

# Contexto global para correlation_id
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(cid: Optional[str]) -> None:
    """
    Establece el correlation_id en el contexto (ContextVar).
    Usar en middleware al inicio de cada request.
    """
    correlation_id_var.set((cid or "").strip() or "-")


def get_correlation_id() -> str:
    """Obtiene el correlation_id actual del contexto."""
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Restablece el correlation_id a '-' (útil en tareas background)."""
    correlation_id_var.set("-")


def install_logrecord_factory() -> None:
    """
    Instala una LogRecordFactory que inyecta 'correlation_id' en cada LogRecord
    **sin sobrescribir** si ya existe. Solo lo añade cuando:
      - No está presente en record.__dict__, o
      - Está vacío/falsy o es '-'.
    """
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)

        # Valor actual (si fue pasado vía extra=..., otro factory, etc.)
        current = record.__dict__.get("correlation_id")

        # Inyectar solo cuando falte o sea vacío/'-'
        if not current or current == "-":
            cid = correlation_id_var.get()
            record.__dict__["correlation_id"] = cid or "-"

        # Asegurar que siempre exista para el formatter %(correlation_id)s
        if "correlation_id" not in record.__dict__:
            record.__dict__["correlation_id"] = "-"

        return record

    logging.setLogRecordFactory(record_factory)
