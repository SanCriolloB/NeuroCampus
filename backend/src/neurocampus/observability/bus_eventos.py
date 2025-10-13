# backend/src/neurocampus/observability/bus_eventos.py
from __future__ import annotations
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
import time, uuid, logging

# Evento base para training.*, prediction.*, data.* (extensible)
@dataclass
class Evento:
    name: str              # e.g., "training.started"
    ts: float              # epoch seconds
    correlation_id: str    # job_id o run_id
    payload: Dict[str, Any]

class EventBus:
    """Bus simple in-memory (pub/sub) con entrega sin garantías.
    Sustituible por Kafka/Rabbit en despliegues futuros.
    """
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Evento], None]]] = {}
        self._log = logging.getLogger("neurocampus.events.bus")

    def subscribe(self, topic: str, handler: Callable[[Evento], None]) -> None:
        """Registra un handler para un tópico concreto (e.g., 'prediction.completed')."""
        self._subs.setdefault(topic, []).append(handler)
        self._log.info("Suscrito handler a topic=%s: %s",
                       topic, getattr(handler, "__name__", repr(handler)))

    def publish(self, topic: str, payload: Dict[str, Any]) -> Evento:
        """Publica un evento al tópico dado y entrega a todos los suscriptores."""
        evt = Evento(
            name=topic,
            ts=time.time(),
            correlation_id=payload.get("correlation_id") or str(uuid.uuid4()),
            payload=payload
        )
        handlers = self._subs.get(topic, [])
        if not handlers:
            # Fallback en desarrollo: si no hay suscriptores, al menos dejar rastro en logs
            self._log.info("event=%s payload=%s", evt.name, evt.payload)
            return evt

        for handler in handlers:
            try:
                handler(evt)
            except Exception as e:
                # Nunca romper el flujo de la app por el destino de observabilidad
                self._log.warning("Handler error for topic=%s: %s", topic, e)
        return evt

# Singleton (simple) para importar en otras capas
BUS = EventBus()

# --- API de módulo esperada por otras capas ---

def publicador(event: str, payload: Dict[str, Any]) -> Evento:
    """Punto único para publicar eventos desde el dominio."""
    return BUS.publish(event, payload)

def suscribir(topic: str, handler: Callable[[Evento], None]) -> None:
    """Atajo para registrar suscriptores al bus in-memory."""
    BUS.subscribe(topic, handler)

__all__ = ["Evento", "EventBus", "BUS", "publicador", "suscribir"]
