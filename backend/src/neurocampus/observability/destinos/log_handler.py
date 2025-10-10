# backend/src/neurocampus/observability/destinos/log_handler.py
import logging
from ..bus_eventos import BUS, Evento  # <- subir un nivel (..), no "."

logger = logging.getLogger("neurocampus.training")
logger.setLevel(logging.INFO)

# Adjuntar un handler si no hay ninguno (asegura salida en consola)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s %(message)s"))
    logger.addHandler(_h)

# Evita doble impresión si el root logger también maneja INFO (opcional)
logger.propagate = False

# Bandera de idempotencia (evita múltiples suscripciones con --reload)
_WIRED = False

# Handler: imprime cada evento training.*
def _to_log(evt: Evento):
    # usa formato parametrizado para no construir strings si el nivel no aplica
    logger.info("%s cid=%s payload=%s", evt.name, evt.correlation_id, evt.payload)

def wire_logging_destination():
    """Conecta una única vez el log handler al bus de eventos."""
    global _WIRED
    if _WIRED:
        return
    for topic in (
        "training.started",
        "training.epoch_end",
        "training.completed",
        "training.failed",
    ):
        BUS.subscribe(topic, _to_log)
    _WIRED = True
