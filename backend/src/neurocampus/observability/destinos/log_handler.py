# backend/src/neurocampus/observability/destinos/log_handler.py
"""
Destino de observabilidad vía logging.

- Se suscribe al bus in-memory para eventos:
  - training.started | training.epoch_end | training.completed | training.failed
  - prediction.requested | prediction.completed | prediction.failed
- Imprime cada evento a un logger específico, agregando correlation_id al LogRecord.

Idempotente con _WIRED para evitar duplicar suscripciones en entornos con --reload.
"""

import logging
from typing import Tuple
from ..bus_eventos import BUS, Evento  # import relativo al paquete neurocampus.observability

# Logger de eventos (puedes unificar con tu config dictConfig)
logger = logging.getLogger("neurocampus.events")
logger.setLevel(logging.INFO)

# Fallback: si no hay ningún handler configurado (p. ej., sin dictConfig),
# adjuntamos uno de consola mínimo para que se vea algo durante el desarrollo.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] [cid=%(correlation_id)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# Evita doble impresión si el root logger también maneja INFO (opcional)
logger.propagate = False

# Bandera de idempotencia (evita múltiples suscripciones con --reload)
_WIRED = False

# Tópicos a suscribir
_TRAINING_TOPICS: Tuple[str, ...] = (
    "training.started",
    "training.epoch_end",
    "training.completed",
    "training.failed",
)

_PREDICTION_TOPICS: Tuple[str, ...] = (
    "prediction.requested",
    "prediction.completed",
    "prediction.failed",
)


def _to_log(evt: Evento) -> None:
    """
    Handler que escribe el evento en logs.
    - Usa 'extra={"correlation_id": ...}' para integrarse con formatters que
      incluyan %(correlation_id)s (compatibles con middleware/contexto CID).
    - El mensaje incluye el nombre del evento y el payload serializado.
    """
    logger.info("%s %s", evt.name, evt.payload, extra={"correlation_id": evt.correlation_id})


def wire_logging_destination() -> None:
    """
    Conecta una única vez el log handler al bus de eventos existentes.
    Si se llama más de una vez (p. ej. por --reload), no duplica suscripciones.
    """
    global _WIRED
    if _WIRED:
        return

    for topic in _TRAINING_TOPICS + _PREDICTION_TOPICS:
        BUS.subscribe(topic, _to_log)

    _WIRED = True
    logger.info("Observability logging wired for topics: %s", _TRAINING_TOPICS + _PREDICTION_TOPICS)
