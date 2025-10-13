# backend/tests/observability/test_eventos_prediccion.py
import types
import builtins
from neurocampus.observability import (
    EV_PRED_REQUESTED, EV_PRED_COMPLETED, EV_PRED_FAILED,
    emit_requested, emit_completed, emit_failed
)

class Collector:
    def __init__(self):
        self.calls = []
    def __call__(self, event, payload):
        self.calls.append((event, payload))

def test_emit_requested(monkeypatch):
    from neurocampus.observability import eventos_prediccion as ev
    fake = Collector()
    monkeypatch.setattr(ev, "publicador", fake, raising=True)

    emit_requested("cid-1", "sentiment_desempeno", "online", 1)

    assert fake.calls, "No se llamó publicador"
    event, payload = fake.calls[0]
    assert event == EV_PRED_REQUESTED
    assert payload["family"] == "sentiment_desempeno"
    assert payload["mode"] == "online"
    assert payload["n_items"] == 1
    assert "ts" in payload

def test_emit_completed(monkeypatch):
    from neurocampus.observability import eventos_prediccion as ev
    fake = Collector()
    monkeypatch.setattr(ev, "publicador", fake, raising=True)

    emit_completed("cid-2", latencia_ms=10, n_items=2,
                   distribucion_labels={"Álgebra":1,"Geometría":1})

    event, payload = fake.calls[0]
    assert event == EV_PRED_COMPLETED
    assert payload["n_items"] == 2
    assert "latencia_ms" in payload
    assert "distribucion_labels" in payload

def test_emit_failed(monkeypatch):
    from neurocampus.observability import eventos_prediccion as ev
    fake = Collector()
    monkeypatch.setattr(ev, "publicador", fake, raising=True)

    emit_failed("cid-3", error="modelo no disponible", stage="predict", error_code="MODEL_NOT_AVAILABLE")

    event, payload = fake.calls[0]
    assert event == EV_PRED_FAILED
    assert payload["error_code"] == "MODEL_NOT_AVAILABLE"
    assert payload.get("stage") == "predict"
