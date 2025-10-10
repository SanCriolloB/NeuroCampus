from fastapi import APIRouter, BackgroundTasks
from ..schemas.modelos import EntrenarRequest, EntrenarResponse, EstadoResponse
from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida
from ...observability.bus_eventos import BUS  # capturamos eventos training.*
from typing import Dict, Any, List
import uuid

router = APIRouter()

# Registro in-memory de estados
# Estructura:
# _ESTADOS[job_id] = {
#   "job_id": str,
#   "status": "running" | "completed" | "failed" | "unknown",
#   "metrics": Dict[str, float],         # último snapshot
#   "history": List[Dict[str, Any]],     # [{epoch, loss, ...metrics}]
#   "model": str,                        # "rbm_general" | "rbm_restringida" (si disponible)
#   "params": Dict[str, Any],            # hparams, etc. (si disponible)
#   "error": str | None
# }
_ESTADOS: Dict[str, Dict[str, Any]] = {}

# Para evitar suscribir múltiples veces los handlers de un mismo job con --reload
_OBS_WIRED_JOBS: set[str] = set()


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
        st["params"] = evt.payload.get("params", st.get("params", {}))

    def _on_epoch_end(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])
        epoch = evt.payload.get("epoch")
        loss = evt.payload.get("loss")
        metrics = evt.payload.get("metrics", {}) or {}
        # Guardar punto de la curva
        point = {"epoch": epoch, "loss": loss, **metrics}
        st["history"].append(point)
        # Snapshot de últimas métricas
        st["metrics"] = metrics if isinstance(metrics, dict) else {"loss": loss}

    def _on_completed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        final_metrics = evt.payload.get("final_metrics", {}) or {}
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
        {**req.hparams, "job_id": job_id},
        model_name=req.modelo,
    )

    # Consolidar estado final (por si el template devolvió info adicional)
    # Nota: out = {"job_id", "status", "metrics"} según plantilla actual
    st = _ESTADOS.get(job_id, {})
    st.update(out)
    _ESTADOS[job_id] = st


@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # Estado inicial visible para la UI
    _ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "metrics": {},
        "history": [],  # acumularemos aquí cada epoch_end
        "model": req.modelo,
        "params": {"epochs": req.epochs, **(req.hparams or {})},
        "error": None,
    }
    # Lanza en background
    bg.add_task(_run_training, job_id, req)
    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str):
    # Mantener compat con esquema actual: si no existe, devolver "unknown"
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}}
    # OJO: si tu EstadoResponse aún no define "history", FastAPI filtrará ese campo.
    # Para exponerlo en la respuesta, añade en schemas/modelos.py:
    #   class EstadoResponse(BaseModel):
    #       job_id: str
    #       status: str
    #       metrics: Dict[str, float] = {}
    #       history: List[Dict[str, Any]] = []
    return EstadoResponse(**st)
