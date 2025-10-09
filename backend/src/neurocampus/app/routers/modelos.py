from fastapi import APIRouter, BackgroundTasks
from ..schemas.modelos import EntrenarRequest, EntrenarResponse, EstadoResponse
from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida
from typing import Dict
import uuid

router = APIRouter(prefix="/modelos", tags=["modelos"])

# Registro in-memory de estados
_ESTADOS: Dict[str, Dict] = {}

def _run_training(job_id: str, req: EntrenarRequest):
    estrategia = RBMGeneral() if req.modelo == "rbm_general" else RBMRestringida()
    tpl = PlantillaEntrenamiento(estrategia)
    out = tpl.run(req.data_ref, req.epochs, {**req.hparams, "job_id": job_id}, model_name=req.modelo)
    _ESTADOS[job_id] = out

@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _ESTADOS[job_id] = {"job_id": job_id, "status": "running", "metrics": {}}
    bg.add_task(_run_training, job_id, req)
    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")

@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str):
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}}
    return EstadoResponse(**st)