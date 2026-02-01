"""
Router de Jobs (procesos asíncronos/lanzables desde la UI).

Este módulo expone endpoints bajo el prefijo ``/jobs`` (montado en ``main.py``).

Incluye:

- Preprocesamiento BETO:
  - POST ``/preproc/beto/run``
  - GET  ``/preproc/beto/estado``

- Unificación de histórico:
  - POST ``/data/unify/run``  (mode=acumulado | acumulado_labeled)
  - GET  ``/data/unify/estado``

- Feature-pack (train_matrix.parquet y JSON auxiliares):
  - POST ``/data/features/prepare/run``
  - GET  ``/data/features/prepare/estado``

.. important::
   Las rutas **NO** deben iniciar con ``/jobs`` dentro de este router, porque el prefijo
   ya se aplica desde ``main.py``. Si aquí se declarara ``/jobs/...`` el resultado sería
   el bug ``/jobs/jobs/...``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ---------------------------------------------------------------------------
# Base dir del repo (ajústalo si tu estructura cambia)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
JOBS_DIR = BASE_DIR / "data" / ".jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JobStatus = Literal["queued", "running", "done", "error"]


def resolve_local_path(ref: str) -> str:
    """Resuelve una referencia localfs:// o relativa a path absoluto.

    :param ref: Ruta relativa o URI ``localfs://...``.
    :return: Ruta absoluta como string.
    """
    if ref.startswith("localfs://"):
        ref = ref.replace("localfs://", "")
    p = Path(ref)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return str(p)


# ---------------------------------------------------------------------------
# Modelo de estado genérico
# ---------------------------------------------------------------------------
class JobState(BaseModel):
    """Estado de un job para polling desde frontend.

    :param job_id: Identificador único.
    :param status: Estado (queued/running/done/error).
    :param message: Mensaje breve para UI.
    :param output_uri: Artefacto principal producido (si aplica).
    :param error: Error detallado si status=error.
    :param params: Parámetros del job (eco para depuración/UI).
    """

    job_id: str
    status: JobStatus = "queued"
    message: str = ""
    output_uri: Optional[str] = None
    error: Optional[str] = None
    params: dict = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Persistir estado en disco (JSON)."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JobState":
        """Cargar estado desde disco (JSON)."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# BETO
# ---------------------------------------------------------------------------
class BetoRunRequest(BaseModel):
    """Request para lanzar preprocesamiento BETO.

    :param dataset_id: ID de dataset/periodo (ej. ``2025-1``).
    :param treat_empty_as_no_text: Si True, convierte comentarios vacíos a NO_TEXT.
    :param force: Si True, recalcula aunque exista salida previa.
    """

    dataset_id: str
    treat_empty_as_no_text: bool = True
    force: bool = False


def _beto_job_path(job_id: str) -> Path:
    """Path del estado del job BETO."""
    return JOBS_DIR / f"beto_{job_id}.json"


def _run_beto_job(job_id: str, req: BetoRunRequest) -> None:
    """Ejecuta el job BETO en background.

    Implementación: llama al módulo cmd_preprocesar_beto usando **sys.executable**
    para asegurar que use el mismo intérprete/venv que está corriendo Uvicorn.
    """
    p = _beto_job_path(job_id)
    job = JobState.load(p)
    job.status = "running"
    job.message = "Ejecutando BETO..."
    job.save(p)

    try:
        input_uri = f"datasets/{req.dataset_id}.parquet"
        output_uri = f"data/labeled/{req.dataset_id}_beto.parquet"

        cmd = [
            sys.executable,
            "-m",
            "neurocampus.app.jobs.cmd_preprocesar_beto",
            "--dataset",
            req.dataset_id,
            "--in",
            input_uri,
            "--out",
            output_uri,
        ]

        if req.treat_empty_as_no_text:
            cmd.append("--treat-empty-as-no-text")
        if req.force:
            cmd.append("--force")

        res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or res.stdout or "Fallo ejecutando cmd_preprocesar_beto")

        job.status = "done"
        job.message = "BETO finalizado."
        job.output_uri = output_uri
        job.save(p)

    except Exception as e:
        job.status = "error"
        job.message = "BETO falló."
        job.error = str(e)
        job.save(p)


@router.post("/preproc/beto/run")
@router.post("/data/preprocesar_beto/run")  # alias legacy (sin /jobs en el path)
def beto_run(req: BetoRunRequest):
    """Lanza el job de preprocesamiento BETO.

    Retorna un ``job_id`` para hacer polling con ``/preproc/beto/estado``.
    """
    job_id = str(uuid4())
    p = _beto_job_path(job_id)
    job = JobState(
        job_id=job_id,
        status="queued",
        message="Job BETO en cola.",
        params=req.model_dump(),
        output_uri=f"data/labeled/{req.dataset_id}_beto.parquet",
    )
    job.save(p)

    # BackgroundTask: en esta base simple usamos subprocess sin cola formal.
    # FastAPI ejecutará esto en el threadpool si lo llamas como background task
    # desde la UI; aquí lo dejamos listo para ser invocado como tarea simple.
    # Nota: si ya usas BackgroundTasks en tu repo, puedes integrarlo ahí.
    from threading import Thread

    Thread(target=_run_beto_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": job.status}


@router.get("/preproc/beto/estado")
@router.get("/data/preprocesar_beto/estado")  # alias legacy
def beto_estado(job_id: str):
    """Consulta el estado del job BETO."""
    p = _beto_job_path(job_id)
    if not p.exists():
        return {"job_id": job_id, "status": "error", "message": "Job no encontrado.", "error": "not_found"}
    return JobState.load(p).model_dump()


# ---------------------------------------------------------------------------
# UNIFY (historico/unificado*.parquet)
# ---------------------------------------------------------------------------
class UnifyRunRequest(BaseModel):
    """Request para unificar histórico.

    :param mode: ``acumulado`` (history) o ``acumulado_labeled`` (labeled).
    """

    mode: Literal["acumulado", "acumulado_labeled"] = "acumulado"


def _unify_job_path(job_id: str) -> Path:
    """Path del estado del job UNIFY."""
    return JOBS_DIR / f"unify_{job_id}.json"


def _run_unify_job(job_id: str, req: UnifyRunRequest) -> None:
    """Ejecuta unificación en background."""
    p = _unify_job_path(job_id)
    job = JobState.load(p)
    job.status = "running"
    job.message = f"Unificando histórico ({req.mode})..."
    job.save(p)

    try:
        try:
            from neurocampus.data.strategies.unificacion import UnificacionStrategy
        except Exception as e:
            raise RuntimeError(
                "No se pudo importar UnificacionStrategy. "
                "Revisa neurocampus.data.strategies.unificacion."
            ) from e

        strat = UnificacionStrategy(base_uri=f"localfs://{BASE_DIR.as_posix()}")

        method_name = "acumulado_labeled" if req.mode == "acumulado_labeled" else "acumulado"
        fn = getattr(strat, method_name, None)
        if fn is None:
            raise RuntimeError(f"UnificacionStrategy no tiene el método '{method_name}'")

        fn()

        out = "historico/unificado_labeled.parquet" if req.mode == "acumulado_labeled" else "historico/unificado.parquet"
        job.status = "done"
        job.message = "Unificación finalizada."
        job.output_uri = out
        job.save(p)

    except Exception as e:
        job.status = "error"
        job.message = "Unificación falló."
        job.error = str(e)
        job.save(p)


@router.post("/data/unify/run")
def unify_run(req: UnifyRunRequest):
    """Lanza la unificación del histórico.

    Retorna ``job_id`` para polling en ``/data/unify/estado``.
    """
    job_id = str(uuid4())
    p = _unify_job_path(job_id)

    out = "historico/unificado_labeled.parquet" if req.mode == "acumulado_labeled" else "historico/unificado.parquet"
    job = JobState(
        job_id=job_id,
        status="queued",
        message="Job unificación en cola.",
        params=req.model_dump(),
        output_uri=out,
    )
    job.save(p)

    from threading import Thread

    Thread(target=_run_unify_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": job.status}


@router.get("/data/unify/estado")
def unify_estado(job_id: str):
    """Consulta el estado del job UNIFY."""
    p = _unify_job_path(job_id)
    if not p.exists():
        return {"job_id": job_id, "status": "error", "message": "Job no encontrado.", "error": "not_found"}
    return JobState.load(p).model_dump()


# ---------------------------------------------------------------------------
# FEATURE PACK
# ---------------------------------------------------------------------------
class FeaturePackRunRequest(BaseModel):
    """Request para construir feature-pack.

    :param dataset_id: ID de dataset/periodo (ej. ``2025-1``).
    :param input_uri: Dataset fuente (normalmente labeled o processed).
    """

    dataset_id: str
    input_uri: str = ""


def _features_job_path(job_id: str) -> Path:
    """Path del estado del job feature-pack."""
    return JOBS_DIR / f"feature_pack_{job_id}.json"


def _run_feature_pack_job(job_id: str, req: FeaturePackRunRequest) -> None:
    """Construye artifacts/features/<dataset_id>/* en background."""
    p = _features_job_path(job_id)
    job = JobState.load(p)
    job.status = "running"
    job.message = "Preparando feature-pack..."
    job.save(p)

    try:
        # Preferimos construir desde labeled si existe, si no desde processed.
        input_uri = req.input_uri.strip()
        if not input_uri:
            labeled = BASE_DIR / "data" / "labeled" / f"{req.dataset_id}_beto.parquet"
            processed = BASE_DIR / "data" / "processed" / f"{req.dataset_id}.parquet"
            if labeled.exists():
                input_uri = str(labeled.relative_to(BASE_DIR)).replace("\\", "/")
            elif processed.exists():
                input_uri = str(processed.relative_to(BASE_DIR)).replace("\\", "/")
            else:
                raise FileNotFoundError(
                    f"No existe input_uri y no se encontró labeled/processed para dataset={req.dataset_id}"
                )

        try:
            from neurocampus.data.features_prepare import prepare_feature_pack
        except Exception as e:
            raise RuntimeError("No se pudo importar prepare_feature_pack") from e

        out_dir = str((BASE_DIR / "artifacts" / "features" / req.dataset_id).resolve())

        prepare_feature_pack(
            base_dir=BASE_DIR,
            dataset_id=req.dataset_id,
            input_uri=input_uri,
            output_dir=out_dir,
        )

        job.status = "done"
        job.message = "Feature-pack listo."
        job.output_uri = f"artifacts/features/{req.dataset_id}/train_matrix.parquet"
        job.save(p)

    except Exception as e:
        job.status = "error"
        job.message = "Feature-pack falló."
        job.error = str(e)
        job.save(p)


@router.post("/data/features/prepare/run")
def features_prepare_run(req: FeaturePackRunRequest):
    """Lanza construcción de feature-pack.

    Retorna ``job_id`` para polling con ``/data/features/prepare/estado``.
    """
    job_id = str(uuid4())
    p = _features_job_path(job_id)

    job = JobState(
        job_id=job_id,
        status="queued",
        message="Job feature-pack en cola.",
        params=req.model_dump(),
        output_uri=f"artifacts/features/{req.dataset_id}/train_matrix.parquet",
    )
    job.save(p)

    from threading import Thread

    Thread(target=_run_feature_pack_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": job.status}


@router.get("/data/features/prepare/estado")
def features_prepare_estado(job_id: str):
    """Consulta el estado del job feature-pack."""
    p = _features_job_path(job_id)
    if not p.exists():
        return {"job_id": job_id, "status": "error", "message": "Job no encontrado.", "error": "not_found"}
    return JobState.load(p).model_dump()
