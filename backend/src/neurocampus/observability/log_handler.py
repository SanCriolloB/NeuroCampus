import logging
from .bus_eventos import BUS, Evento

logger = logging.getLogger("neurocampus.training")
logger.setLevel(logging.INFO)

# Handler de ejemplo: imprime en logs cada evento training.*
def _to_log(evt: Evento):
    logger.info(f"{evt.name} cid={evt.correlation_id} payload={evt.payload}")

def wire_logging_destination():
    # Suscripciones de interés para Día 4 (entrenamiento)
    for topic in [
        "training.started",
        "training.epoch_end",
        "training.completed",
        "training.failed",
    ]:
        BUS.subscribe(topic, _to_log)