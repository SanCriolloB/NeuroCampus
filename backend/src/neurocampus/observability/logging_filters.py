# backend/src/neurocampus/observability/logging_filters.py
import logging

class CorrelationIdLogFilter(logging.Filter):
    """
    Inyecta 'correlation_id' en el LogRecord si no está presente,
    para que el formateador siempre pueda usar %(correlation_id)s.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "-"
        return True
