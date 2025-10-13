# backend/src/neurocampus/app/logging_config.py
import logging
import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,  # mantiene loggers de uvicorn/fastapi
    "filters": {
        "cid": {
            "()": "neurocampus.observability.logging_filters.CorrelationIdLogFilter"
        }
    },
    "formatters": {
        "default": {
            # Incluimos cid en todas las líneas
            "format": "%(asctime)s %(levelname)s [cid=%(correlation_id)s] %(name)s: %(message)s"
        },
        "uvicorn": {
            "format": "%(asctime)s %(levelname)s [cid=%(correlation_id)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "filters": ["cid"],
            "formatter": "default",
        },
        # Puedes añadir más handlers (file, syslog, etc.) con el mismo filtro "cid"
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    },
    # Opcional: afinar loggers de uvicorn para que usen el mismo handler/formatter
    "loggers": {
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        # Logger de tu app
        "neurocampus": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING)
