from typing import Callable, Dict, List, Any
from dataclasses import dataclass, asdict
import time, uuid

# Evento base para training.*, prediction.*, data.* (extensible)
@dataclass
class Evento:
    name: str              # e.g., "training.started"
    ts: float              # epoch seconds
    correlation_id: str    # job_id o run_id
    payload: Dict[str, Any]

class EventBus:
    """Bus simple in-memory (pub/sub) con entrega sin garantías.
    Sustituible por Kafka/Rabbit en despliegues futuros."""
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Evento], None]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Evento], None]) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Dict[str, Any]) -> Evento:
        evt = Evento(name=topic, ts=time.time(), correlation_id=payload.get("correlation_id", str(uuid.uuid4())), payload=payload)
        for handler in self._subs.get(topic, []):
            try:
                handler(evt)
            except Exception as e:
                # No interrumpe el flujo de entrenamiento
                print(f"[obs] handler error for {topic}: {e}")
        return evt

# Singleton (simple) para importar en otras capas
BUS = EventBus()