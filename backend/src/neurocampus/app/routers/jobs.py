# backend/src/neurocampus/app/routers/jobs.py
"""
Router del contexto 'jobs'.

Uso:
- Operaciones relacionadas con ejecución y estado de jobs en background.
- Proporciona puntos de extensión para colas, schedulers, etc.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Literal, Optional
import json
import os
import subprocess
import sys
import time
import uuid

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuración básica de rutas/paths
# ---------------------------------------------------------------------------

# __file__ = backend/src/neurocampus/app/routers/jobs.py
# parents[0] = routers
# parents[1] = app
# parents[2] = neurocampus
# parents[3] = src
# parents[4] = backend
# parents[5] = raíz del proyecto (NeuroCampus)
BASE_DIR = Path(__file__).resolve().parents[5]

# Directorios del pipeline de datos
DATASETS_DIR = BASE_DIR / "datasets"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_LABELED_DIR = BASE_DIR / "data" / "labeled"

# Directorio de jobs (compatible con tools/cleanup.py → BASE_DIR / "jobs")
JOBS_ROOT = Path(os.getenv("NC_JOBS_DIR", BASE_DIR / "jobs"))
BETO_JOBS_DIR = JOBS_ROOT / "preproc_beto"
BETO_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    """Devuelve timestamp ISO básico en UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _job_path(job_id: str) -> Path:
    """Ruta al archivo JSON de un job BETO concreto."""
    return BETO_JOBS_DIR / f"{job_id}.json"


def _load_job(job_id: str) -> dict:
    """Carga un job desde disco; lanza 404 si no existe."""
    path = _job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_job(job: dict) -> None:
    """Persiste un job en disco (sobrescribe)."""
    path = _job_path(job["id"])
    BETO_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _list_jobs() -> list[dict]:
    """Lista jobs ordenados por fecha de creación descendente."""
    if not BETO_JOBS_DIR.exists():
        return []
    jobs: list[dict] = []
    for p in sorted(BETO_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8") as f:
                jobs.append(json.load(f))
        except Exception:
            continue
    return jobs


# ---------------------------------------------------------------------------
# Modelos Pydantic para requests/responses
# ---------------------------------------------------------------------------

JobStatus = Literal["created", "running", "done", "failed"]


class BetoPreprocRequest(BaseModel):
    """
    Request mínimo para lanzar el preprocesamiento BETO.

    - dataset: nombre base del dataset (sin extensión).
      El job intentará:
        1) Usar data/processed/{dataset}.parquet si existe.
        2) Si no existe, normalizar desde datasets/{dataset}.parquet o .csv
           usando cmd_cargar_dataset, y luego ejecutar BETO.
    """
    dataset: str
    text_col: Optional[str] = None   # Ej: "Sugerencias" o None → auto
    keep_empty_text: bool = True     # Mantener filas sin texto como neutrales


class BetoPreprocMeta(BaseModel):
    """Subset de campos interesantes del .meta.json generado por el job CLI."""
    model: str
    created_at: str
    n_rows: int
    accepted_count: int
    threshold: float
    margin: float
    neu_min: float
    text_col: str
    text_coverage: float
    keep_empty_text: bool
    text_feats: str | None = None


class BetoPreprocJob(BaseModel):
    """Estado de un job BETO expuesto al frontend."""
    id: str
    dataset: str
    src: str
    dst: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    meta: Optional[BetoPreprocMeta] = None
    error: Optional[str] = None

    # Información opcional sobre el puente datasets/ → data/processed/
    raw_src: Optional[str] = None          # Ruta del dataset crudo (datasets/)
    needs_cargar_dataset: bool = False     # True si hubo que normalizar antes


# ---------------------------------------------------------------------------
# Tarea de background: ejecuta cmd_cargar_dataset (si hace falta) y luego cmd_preprocesar_beto.py
# ---------------------------------------------------------------------------

def _run_beto_job(job_id: str) -> None:
    """
    Ejecuta el job en background.

    Flujo:
    - Si needs_cargar_dataset=True y existe raw_src:
        - Llama a cmd_cargar_dataset para generar el parquet normalizado en `src`.
    - Llama al módulo CLI cmd_preprocesar_beto.py con subprocess.run.
    - Actualiza el JSON del job con status, meta y posible error.
    """
    job = _load_job(job_id)
    job["status"] = "running"
    job["started_at"] = _now_iso()
    _save_job(job)

    # Rutas de entrada/salida para BETO (src siempre debe ser el parquet normalizado)
    src = job["src"]
    dst = job["dst"]
    text_col = job.get("text_col") or "auto"
    keep_empty_text = bool(job.get("keep_empty_text", True))

    try:
        # 1) Si hace falta, normalizar primero: datasets/ -> data/processed/
        if job.get("needs_cargar_dataset") and job.get("raw_src"):
            cmd_norm = [
                sys.executable,
                "-m",
                "neurocampus.app.jobs.cmd_cargar_dataset",
                "--in",
                job["raw_src"],
                "--out",
                src,
            ]
            subprocess.run(cmd_norm, check=True)

        # 2) Ejecutar BETO sobre el parquet normalizado
        cmd_beto = [
            sys.executable,
            "-m",
            "neurocampus.app.jobs.cmd_preprocesar_beto",
            "--in",
            src,
            "--out",
            dst,
            "--text-col",
            text_col,
            "--beto-mode",
            "probs",
        ]
        if keep_empty_text:
            cmd_beto.append("--keep-empty-text")

        subprocess.run(cmd_beto, check=True)

        # 3) Intentar leer el .meta.json generado por el script
        meta_path = Path(dst + ".meta.json")
        meta: dict | None = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        job["status"] = "done"
        job["finished_at"] = _now_iso()
        if meta:
            # Guardamos solo los campos que nos interesan para la UI
            job["meta"] = {
                "model": meta.get("model", ""),
                "created_at": meta.get("created_at", ""),
                "n_rows": meta.get("n_rows", 0),
                "accepted_count": meta.get("accepted_count", 0),
                "threshold": meta.get("threshold", 0.0),
                "margin": meta.get("margin", 0.0),
                "neu_min": meta.get("neu_min", 0.0),
                "text_col": meta.get("text_col", ""),
                "text_coverage": meta.get("text_coverage", 0.0),
                "keep_empty_text": meta.get("keep_empty_text", False),
                "text_feats": meta.get("text_feats"),
            }
    except Exception as e:
        job["status"] = "failed"
        job["finished_at"] = _now_iso()
        job["error"] = str(e)

    _save_job(job)


# ---------------------------------------------------------------------------
# Endpoints públicos del router /jobs
# ---------------------------------------------------------------------------

@router.get("/ping")
def ping() -> dict:
    """
    Comprobación rápida de vida del router /jobs.
    Ayuda a verificar que el prefijo y el registro en main.py funcionan.
    """
    return {"jobs": "pong"}


@router.post(
    "/preproc/beto/run",
    response_model=BetoPreprocJob,
    summary=(
        "Lanza un job de preprocesamiento BETO sobre data/processed/*.parquet "
        "(si no existe, intenta normalizar desde datasets/ primero)."
    ),
)
def launch_beto_preproc(req: BetoPreprocRequest, background: BackgroundTasks) -> BetoPreprocJob:
    """
    Crea un job BETO y lo ejecuta en background.

    Flujo:
    - Si ya existe `data/processed/{dataset}.parquet` → usarlo como entrada de BETO.
    - Si NO existe:
        - Buscar `datasets/{dataset}.parquet` o `datasets/{dataset}.csv`.
        - Si encuentra alguno, marca el job con needs_cargar_dataset=True y raw_src=<ruta>.
        - La tarea de background ejecutará cmd_cargar_dataset antes de cmd_preprocesar_beto.
    - Si tampoco existe en datasets/ → 400.
    """
    # Ruta NORMALIZADA esperada por el pipeline
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED_DIR / f"{req.dataset}.parquet"

    needs_cargar_dataset = False
    raw_src: Optional[Path] = None

    if not processed_path.exists():
        # No hay dataset normalizado todavía → intentamos hacer el puente desde datasets/
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        raw_parquet = DATASETS_DIR / f"{req.dataset}.parquet"
        raw_csv = DATASETS_DIR / f"{req.dataset}.csv"

        if raw_parquet.exists():
            raw_src = raw_parquet
        elif raw_csv.exists():
            raw_src = raw_csv
        else:
            # No hay ni procesado ni crudo → no podemos seguir
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No existe dataset procesado en {processed_path} ni dataset crudo en "
                    f"{raw_parquet} / {raw_csv}. Sube un dataset desde /datos o "
                    f"genera el procesado manualmente."
                ),
            )

        needs_cargar_dataset = True

    # Salida etiquetada por BETO
    DATA_LABELED_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_LABELED_DIR / f"{req.dataset}_beto.parquet"

    job_id = f"beto-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    job_dict = {
        "id": job_id,
        "dataset": req.dataset,
        "src": str(processed_path),       # SIEMPRE el parquet normalizado
        "dst": str(dst),
        "status": "created",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "meta": None,
        "error": None,
        "text_col": req.text_col,
        "keep_empty_text": req.keep_empty_text,
        # Campos del puente datasets/ → data/processed/
        "raw_src": str(raw_src) if raw_src is not None else None,
        "needs_cargar_dataset": needs_cargar_dataset,
    }
    _save_job(job_dict)

    # Programar ejecución en background
    background.add_task(_run_beto_job, job_id)

    return BetoPreprocJob(**job_dict)


@router.get(
    "/preproc/beto/{job_id}",
    response_model=BetoPreprocJob,
    summary="Devuelve el estado de un job BETO concreto",
)
def get_beto_job(job_id: str) -> BetoPreprocJob:
    job_dict = _load_job(job_id)
    return BetoPreprocJob(**job_dict)


@router.get(
    "/preproc/beto",
    response_model=list[BetoPreprocJob],
    summary="Lista jobs BETO recientes (últimos primero)",
)
def list_beto_jobs(limit: int = 20) -> list[BetoPreprocJob]:
    jobs = _list_jobs()
    jobs = jobs[:limit]
    return [BetoPreprocJob(**j) for j in jobs]


class RbmSearchJob(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    config_path: str
    last_run_id: Optional[str] = None  # id del run principal generado


RBM_JOBS_DIR = JOBS_ROOT / "rbm_search"
RBM_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _rbm_job_path(job_id: str) -> Path:
    return RBM_JOBS_DIR / f"{job_id}.json"


def _save_rbm_job(job: dict) -> None:
    path = _rbm_job_path(job["id"])
    RBM_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _load_rbm_job(job_id: str) -> dict:
    path = _rbm_job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job training {job_id} no encontrado")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_rbm_search_job(job_id: str) -> None:
    """
    Ejecuta búsqueda de hiperparámetros de RBM.
    Asume que hparam_search guarda sus resultados en artifacts/runs y
    eventualmente actualiza champions.
    """
    job = _load_rbm_job(job_id)
    job["status"] = "running"
    job["started_at"] = _now_iso()
    _save_rbm_job(job)

    config_path = job["config_path"]

    try:
        cmd = [
            sys.executable,
            "-m",
            "neurocampus.models.hparam_search",
            "--config",
            config_path,
        ]
        subprocess.run(cmd, check=True)

        # Si hparam_search deja algún indicador de "last_run_id", podrías leerlo aquí.
        # Por ahora, lo dejamos en None o podríamos inferir el último run creado leyendo artifacts/runs.
        job["status"] = "done"
        job["finished_at"] = _now_iso()
    except Exception as e:
        job["status"] = "failed"
        job["finished_at"] = _now_iso()
        job["error"] = str(e)

    _save_rbm_job(job)

@router.post(
    "/training/rbm-search",
    response_model=RbmSearchJob,
    summary="Lanza un job de búsqueda de hiperparámetros para RBM",
)
def launch_rbm_search(background: BackgroundTasks, config: str | None = None) -> RbmSearchJob:
    """
    Si no se pasa config, usa configs/rbm_search.yaml por defecto.
    """
    base_dir = BASE_DIR  # ya definido arriba
    config_path = Path(config) if config else (base_dir / "configs" / "rbm_search.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=400, detail=f"No existe config en {config_path}")

    job_id = f"rbm-search-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    job_dict = {
        "id": job_id,
        "status": "created",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "config_path": str(config_path),
        "last_run_id": None,
    }
    _save_rbm_job(job_dict)

    background.add_task(_run_rbm_search_job, job_id)

    return RbmSearchJob(**job_dict)


@router.get(
    "/training/rbm-search/{job_id}",
    response_model=RbmSearchJob,
    summary="Estado de un job de búsqueda de hiperparámetros RBM",
)
def get_rbm_search_job(job_id: str) -> RbmSearchJob:
    job = _load_rbm_job(job_id)
    return RbmSearchJob(**job)


@router.get(
    "/training/rbm-search",
    response_model=list[RbmSearchJob],
    summary="Lista jobs de búsqueda RBM recientes",
)
def list_rbm_search_jobs(limit: int = 20) -> list[RbmSearchJob]:
    if not RBM_JOBS_DIR.exists():
        return []
    jobs: list[dict] = []
    for p in sorted(RBM_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8") as f:
                jobs.append(json.load(f))
        except Exception:
            continue
    jobs = jobs[:limit]
    return [RbmSearchJob(**j) for j in jobs]
