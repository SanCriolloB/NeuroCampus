# backend/src/neurocampus/observability/logging_context.py
import logging
from contextvars import ContextVar

# Contexto global para correlation_id
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

def set_correlation_id(cid: str):
    """
    Establece el correlation_id en el contexto (ContextVar).
    Usar en middleware al inicio de cada request.
    """
    return correlation_id_var.set(cid)

def get_correlation_id() -> str:
    """Obtiene el correlation_id actual del contexto."""
    return correlation_id_var.get()

def install_logrecord_factory() -> None:
    """
    Reemplaza la LogRecordFactory para inyectar correlation_id
    automáticamente en cada LogRecord.
    """
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        # Asegurar que el registro tenga correlation_id
        if not hasattr(record, "correlation_id"):
            record.correlation_id = correlation_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)
