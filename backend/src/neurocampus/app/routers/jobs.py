from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os
import time
import uuid
import subprocess
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Base path (raíz del repo)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Encuentra la raíz del repo de NeuroCampus de forma robusta.

    Criterio:
    - Un directorio que contenga `data/` y `datasets/`.

    Esto evita errores cuando el servidor se lanza desde `backend/` u otra ruta.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data").exists() and (p / "datasets").exists():
            return p
    # Fallback defensivo.
    return here.parents[5]


BASE_DIR: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Estado de jobs en disco (artifacts/jobs)
# ---------------------------------------------------------------------------

JOBS_DIR = (BASE_DIR / "artifacts" / "jobs").resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    """Ruta al archivo JSON de un job."""
    return (JOBS_DIR / f"{job_id}.json").resolve()


def _save_job(job: Dict[str, Any]) -> None:
    """Guarda el job en disco."""
    job_id = str(job.get("job_id") or "")
    if not job_id:
        return
    p = _job_path(job_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _load_job(job_id: str) -> Dict[str, Any]:
    """Carga el job desde disco."""
    p = _job_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BetoPreprocRequest(BaseModel):
    """
    Request para preprocesamiento BETO.

    Parámetros típicos:
    - dataset_id: identificador del dataset (ej. 2025-1)
    - src: ruta al parquet origen (datasets/<id>.parquet o data/processed/<id>.parquet)
    - dst: ruta al parquet labeled de salida (data/labeled/<id>_beto.parquet)
    - device, batch_size, etc: parámetros de inferencia BETO
    """
    dataset_id: str
    src: Optional[str] = None
    dst: Optional[str] = None
    batch_size: int = 64
    device: str = "cpu"


class BetoPreprocJob(BaseModel):
    """
    Estado persistido del job BETO.

    .. important::
       Este job se usa para "polling" desde UI.
    """
    job_id: str
    kind: str = "beto_preproc"
    status: str = "queued"     # queued|running|done|failed
    dataset: str
    src: str
    dst: str
    created_at: float
    updated_at: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Implementación job BETO
# ---------------------------------------------------------------------------

def _run_beto_job(job_id: str) -> None:
    """
    Ejecuta un job de preprocesamiento BETO.

    Flujo:

    1. Normaliza/convierte el dataset de entrada a ``data/processed/<dataset_id>.parquet`` si aplica.
    2. Ejecuta ``cmd_preprocesar_beto`` para producir el parquet *labeled* (por defecto
       ``data/labeled/<dataset_id>_beto.parquet``).
    3. (Mejora) Intenta construir el **Feature Pack** para el entrenamiento de modelos RBM
       (``artifacts/features/<dataset_id>/train_matrix.parquet``).

    El paso (3) es *best-effort*: si la construcción del feature pack falla, el job de BETO
    se marca como ``done`` igualmente, y el entrenamiento puede resolverlo de forma **lazy**
    mediante ``/modelos/entrenar`` con ``auto_prepare=True``.

    :param job_id: identificador del job (archivo JSON dentro de ``artifacts/jobs``).
    """
    job = _load_job(job_id)
    dataset_id = str(job.get('dataset') or '')
    # Ruta de salida del parquet labeled generado por BETO.
    dst = str(job.get('dst') or '')

    job["status"] = "running"
    job["updated_at"] = time.time()
    _save_job(job)

    try:
        # 1) Resolver paths
        dataset_id = str(job.get("dataset") or "")
        src = str(job.get("src") or f"datasets/{dataset_id}.parquet")
        dst = str(job.get("dst") or f"data/labeled/{dataset_id}_beto.parquet")

        # 2) Si src es datasets/<id>.parquet, convertir a processed primero
        #    para mantener consistencia con el pipeline nuevo.
        processed_path = BASE_DIR / "data" / "processed" / f"{dataset_id}.parquet"
        in_path = (BASE_DIR / src).resolve()
        if in_path.exists() and str(in_path).endswith(".parquet") and ("datasets" in src.replace("\\", "/")):
            # Ejecuta cmd_cargar_dataset (normaliza columnas/rating/calif_*)
            cmd = [
                "python", "-m", "neurocampus.app.jobs.cmd_cargar_dataset",
                "--in", str(in_path),
                "--out", str(processed_path),
            ]
            subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
            src = str(processed_path.relative_to(BASE_DIR).as_posix())

        # 3) Ejecuta cmd_preprocesar_beto
        cmd = [
            "python", "-m", "neurocampus.app.jobs.cmd_preprocesar_beto",
            "--in", str((BASE_DIR / src).resolve()),
            "--out", str((BASE_DIR / dst).resolve()),
            "--batch-size", str(int(job.get("batch_size") or 64)),
            "--device", str(job.get("device") or "cpu"),
        ]
        subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

        # Cargar meta (para UI/depuración si existe)
        meta_path = (BASE_DIR / dst).with_suffix((BASE_DIR / dst).suffix + ".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    _ = json.load(f)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Auto-build del Feature Pack (train_matrix.parquet)
        # ---------------------------------------------------------------------
        #
        # La RBM Restringida consume por defecto `data_source="feature_pack"`,
        # que apunta a:
        #
        #   artifacts/features/<dataset_id>/train_matrix.parquet
        #
        # Este feature-pack se construye a partir del parquet *labeled* generado
        # por BETO (dst). Lo ejecutamos en modo best-effort:
        #   - Si falla, NO marcamos el job como failed (para no romper el flujo de Datos).
        #   - El endpoint /modelos/entrenar también puede generarlo en modo lazy
        #     usando `auto_prepare=True`.
        #
        # Permite desactivar el auto-build desde entorno (útil en dev/CI).
        auto_fp = str(os.getenv("NC_AUTO_FEATURE_PACK", "1")).strip().lower() not in ("0", "false", "no")
        if auto_fp:
            try:
                from neurocampus.data.feature_pack_auto import ensure_feature_pack  # noqa: WPS433
                ensure_feature_pack(
                    base_dir=BASE_DIR,
                    dataset_id=dataset_id,
                    input_uri=dst,
                    output_dir=str((BASE_DIR / "artifacts" / "features" / dataset_id).resolve()),
                )
            except Exception as e:
                # Log en stdout/stderr; la UI no depende de este detalle.
                print(f"[jobs] feature_pack auto-build failed for {dataset_id}: {e}")

        job["status"] = "done"
        job["updated_at"] = time.time()
        _save_job(job)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = time.time()
        _save_job(job)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/jobs/data/preprocesar_beto/run",
    summary="Lanza un job de preprocesamiento BETO (genera data/labeled/<dataset>_beto.parquet).",
)
def launch_beto_preproc(req: BetoPreprocRequest, background: BackgroundTasks) -> Dict[str, Any]:
    """
    Lanza job BETO en background.

    - Crea un job_id
    - Persiste el job en artifacts/jobs/<job_id>.json
    - Ejecuta `_run_beto_job(job_id)` en BackgroundTasks

    :param req: request con dataset_id y rutas src/dst opcionales.
    :param background: background tasks.
    :return: dict con job_id.
    """
    dataset_id = str(req.dataset_id).strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id vacío")

    job_id = f"beto_{dataset_id}_{uuid.uuid4().hex[:8]}"
    now = time.time()

    src = str(req.src or f"datasets/{dataset_id}.parquet")
    dst = str(req.dst or f"data/labeled/{dataset_id}_beto.parquet")

    job = BetoPreprocJob(
        job_id=job_id,
        dataset=dataset_id,
        src=src,
        dst=dst,
        created_at=now,
        updated_at=now,
        batch_size=int(req.batch_size),
        device=str(req.device),
    ).model_dump()

    _save_job(job)
    background.add_task(_run_beto_job, job_id)
    return {"job_id": job_id}


@router.get(
    "/jobs/data/preprocesar_beto/estado",
    summary="Estado de un job BETO.",
)
def estado_beto_preproc(job_id: str) -> BetoPreprocJob:
    """
    Devuelve el estado actual del job.

    :param job_id: id del job.
    :return: modelo pydantic con el estado.
    """
    job = _load_job(job_id)
    return BetoPreprocJob(**job)


# ---------------------------------------------------------------------------
# Job: features_prepare (ya existía en tu archivo)
# ---------------------------------------------------------------------------

class FeaturePackPrepareRequest(BaseModel):
    dataset_id: str
    input_uri: str
    output_dir: Optional[str] = None


def _run_feature_pack_prepare(job_id: str, req: FeaturePackPrepareRequest) -> None:
    job = _load_job(job_id)
    job["status"] = "running"
    job["updated_at"] = time.time()
    _save_job(job)

    try:
        from neurocampus.data.features_prepare import prepare_feature_pack  # noqa: WPS433

        dataset_id = str(req.dataset_id).strip()
        if not dataset_id:
            raise ValueError("dataset_id vacío")

        out_dir = req.output_dir or str((BASE_DIR / "artifacts" / "features" / dataset_id).resolve())

        artifacts = prepare_feature_pack(
            base_dir=BASE_DIR,
            dataset_id=dataset_id,
            input_uri=str(req.input_uri),
            output_dir=str(out_dir),
        )

        job["status"] = "done"
        job["updated_at"] = time.time()
        job["artifacts"] = artifacts or {}
        _save_job(job)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = time.time()
        _save_job(job)


@router.post(
    "/jobs/data/features/prepare/run",
    summary="Lanza un job para crear artifacts/features/<dataset_id>/train_matrix.parquet",
)
def launch_feature_pack_prepare(req: FeaturePackPrepareRequest, background: BackgroundTasks) -> Dict[str, Any]:
    dataset_id = str(req.dataset_id).strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id vacío")

    job_id = f"featpack_{dataset_id}_{uuid.uuid4().hex[:8]}"
    now = time.time()

    job = {
        "job_id": job_id,
        "kind": "feature_pack_prepare",
        "status": "queued",
        "dataset": dataset_id,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    _save_job(job)
    background.add_task(_run_feature_pack_prepare, job_id, req)
    return {"job_id": job_id}


@router.get(
    "/jobs/data/features/prepare/estado",
    summary="Estado de un job features/prepare.",
)
def estado_feature_pack_prepare(job_id: str) -> Dict[str, Any]:
    return _load_job(job_id)
