# backend/src/neurocampus/app/routers/modelos.py

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from ..schemas.modelos import EntrenarRequest, EntrenarResponse, EstadoResponse
from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida
from ...observability.bus_eventos import BUS  # capturamos eventos training.*
from typing import Dict, Any
import uuid

router = APIRouter()

# Registro in-memory de estados
# Estructura:
# _ESTADOS[job_id] = {
#   "job_id": str,
#   "status": "running" | "completed" | "failed" | "unknown",
#   "metrics": Dict[str, float],         # último snapshot
#   "history": list[dict[str, Any]],     # [{epoch, loss, ...metrics}]
#   "model": str,                        # "rbm_general" | "rbm_restringida" (si disponible)
#   "params": Dict[str, Any],            # hparams, etc. (si disponible)
#   "error": str | None
# }
_ESTADOS: Dict[str, Dict[str, Any]] = {}

# Para evitar suscribir múltiples veces los handlers de un mismo job con --reload
_OBS_WIRED_JOBS: set[str] = set()


def _normalize_hparams(hparams: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normaliza claves a minúsculas y retorna dict seguro (no None)."""
    if not hparams:
        return {}
    return {str(k).lower(): v for k, v in hparams.items()}


def _flatten_metrics_from_payload(payload: Dict[str, Any], allow_loss: bool = True) -> Dict[str, float]:
    """
    Cuando no venga `metrics` como dict explícito en el payload del evento,
    intenta aplanar los pares numéricos (excepto campos de control).
    """
    if not payload:
        return {}
    ctrl = {"correlation_id", "epoch", "loss", "event", "model", "params", "final_metrics"}
    out: Dict[str, float] = {}
    for k, v in payload.items():
        if k in ctrl:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # opcionalmente incluir loss
    if allow_loss and "loss" in payload and isinstance(payload["loss"], (int, float)):
        out.setdefault("loss", float(payload["loss"]))
    return out


def _wire_job_observers(job_id: str) -> None:
    """
    Se suscribe al BUS para capturar:
    - training.started: inicializa metadatos (modelo/params) si vienen en payload
    - training.epoch_end: agrega un punto a history[] y actualiza metrics
    - training.completed: marca estado y métricas finales
    - training.failed: marca error
    Idempotente por job_id (no re-suscribe en recargas).
    """
    if job_id in _OBS_WIRED_JOBS:
        return

    def _match(evt) -> bool:
        # Coincidir por correlation_id == job_id
        try:
            return evt.payload.get("correlation_id") == job_id
        except Exception:
            return False

    def _on_started(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])
        st["status"] = "running"
        st["model"] = evt.payload.get("model", st.get("model"))
        # Si llegan params desde el evento, preferirlos sobre los iniciales
        params_evt = evt.payload.get("params")
        if isinstance(params_evt, dict):
            st["params"] = _normalize_hparams(params_evt) or st.get("params", {})

    def _on_epoch_end(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])

        payload = evt.payload or {}
        epoch = payload.get("epoch")
        loss = payload.get("loss")

        # Métricas pueden venir como bloque `metrics` o aplanadas
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = _flatten_metrics_from_payload(payload, allow_loss=True)

        # Guardar punto de la curva (epoch/loss + métricas)
        point = {"epoch": epoch}
        if isinstance(loss, (int, float)):
            point["loss"] = float(loss)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("epoch",):
                    point[k] = float(v)

        st["history"].append(point)
        # Snapshot de últimas métricas
        st["metrics"] = {k: v for k, v in point.items() if k not in ("epoch",)}

    def _on_completed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        payload = evt.payload or {}
        final_metrics = payload.get("final_metrics")

        if not isinstance(final_metrics, dict):
            # fallback: aplanar payload (sin epoch) o usar último snapshot
            final_metrics = _flatten_metrics_from_payload(payload, allow_loss=True)
            if not final_metrics:
                final_metrics = st.get("metrics", {})

        st["metrics"] = final_metrics
        st["status"] = "completed"

    def _on_failed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st["status"] = "failed"
        st["error"] = evt.payload.get("error", "unknown error")

    # Suscripciones (best-effort; el BUS no implementa unsubscribe)
    BUS.subscribe("training.started", _on_started)
    BUS.subscribe("training.epoch_end", _on_epoch_end)
    BUS.subscribe("training.completed", _on_completed)
    BUS.subscribe("training.failed", _on_failed)

    _OBS_WIRED_JOBS.add(job_id)


def _run_training(job_id: str, req: EntrenarRequest):
    # Elige estrategia
    estrategia = RBMGeneral() if req.modelo == "rbm_general" else RBMRestringida()
    tpl = PlantillaEntrenamiento(estrategia)

    # Asegurar wiring de observabilidad para este job antes de correr
    _wire_job_observers(job_id)

    # Ejecuta entrenamiento (emite training.* que recogerán los handlers)
    out = tpl.run(
        req.data_ref,
        req.epochs,
        {**(_normalize_hparams(req.hparams)), "job_id": job_id},
        model_name=req.modelo,
    )

    # Consolidar estado final (por si el template devolvió info adicional)
    # Nota: out = {"job_id", "status", "metrics", "history?"} según plantilla
    st = _ESTADOS.get(job_id, {})
    st.update(out)
    _ESTADOS[job_id] = st


@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # Normaliza hparams para mantener contrato consistente a lo largo del flujo
    hp_norm = _normalize_hparams(req.hparams)

    # Estado inicial visible para la UI
    _ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "metrics": {},
        "history": [],  # acumularemos aquí cada epoch_end
        "model": req.modelo,
        "params": {"epochs": req.epochs, **hp_norm},
        "error": None,
    }

    # Lanza en background con hparams normalizados
    req_norm = EntrenarRequest(
        modelo=req.modelo,
        data_ref=req.data_ref,
        epochs=req.epochs,
        hparams=hp_norm
    )
    bg.add_task(_run_training, job_id, req_norm)

    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str):
    # Mantener compat con esquema actual: si no existe, devolver "unknown"
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}}
    # OJO: si tu EstadoResponse no define "history", FastAPI filtrará ese campo.
    return EstadoResponse(**st)
