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

# Jobs del dominio Datos (unificación y feature-pack)
DATA_UNIFY_JOBS_DIR = JOBS_ROOT / "data_unify"
DATA_UNIFY_JOBS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PREP_JOBS_DIR = JOBS_ROOT / "features_prepare"
FEATURES_PREP_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Meta columnas a conservar si existen en el dataset crudo (datasets/)
DEFAULT_META_LIST = (
    "id,profesor,docente,teacher,"
    "materia,asignatura,subject,"
    "codigo_materia,grupo,cedula_profesor"
)

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


def _processed_missing_docente_cols(processed_path: Path) -> bool:
    """
    True si el parquet procesado NO tiene columnas para agrupar por docente.
    Usamos una lectura liviana de schema si hay pyarrow.
    """
    try:
        import pyarrow.parquet as pq
        cols = set(pq.ParquetFile(processed_path).schema.names)
    except Exception:
        import pandas as pd
        cols = set(pd.read_parquet(processed_path, engine="auto").columns)

    # Candidatos mínimos para poder agrupar por docente
    return not any(c in cols for c in ("profesor", "docente", "teacher"))

# ---------------------------------------------------------------------------
# Modelos Pydantic para requests/responses
# ---------------------------------------------------------------------------

JobStatus = Literal["created", "running", "done", "failed"]


class BetoPreprocRequest(BaseModel):
    """Request para lanzar el preprocesamiento BETO desde la pestaña **Datos**.

    Compatibilidad
    --------------
    Este request mantiene `keep_empty_text` para no romper clientes existentes.
    Sin embargo, el nuevo pipeline recomienda controlar el manejo de texto vacío
    con `empty_text_policy`:

    - ``neutral``: trata texto vacío como neutral (comportamiento legacy).
    - ``zero``: marca NO_TEXT y setea p_neg=p_neu=p_pos=0 (evita sesgo neutral).

    Flags nuevos
    -----------
    - `text_feats="tfidf_lsa"`: genera embeddings `feat_t_1..feat_t_64`.
    - `text_feats_out_dir`: carpeta destino de artefactos TF-IDF+LSA (opcional).
    """

    dataset: str
    text_col: Optional[str] = None   # Ej: "Sugerencias" o None → auto

    # Legacy: mantener filas sin texto (antes se contaban como neutrales).
    keep_empty_text: bool = True

    # Filtrado mínimo: comentarios con menos tokens se consideran sin texto.
    min_tokens: int = 1

    # Activar embeddings TF-IDF+LSA (64 dims). Si es None/"none": no se generan.
    text_feats: Optional[Literal["none", "tfidf_lsa"]] = None

    # Directorio donde se guardan artefactos del embedding (vocab, lsa, etc.).
    # Si no se envía y `text_feats="tfidf_lsa"`, se usa:
    #   artifacts/textfeats/<dataset>
    text_feats_out_dir: Optional[str] = None

    # Política para comentarios vacíos (NO_TEXT).
    # Si es None, se infiere:
    #   - neutral si keep_empty_text=True
    #   - zero si keep_empty_text=False
    empty_text_policy: Optional[Literal["neutral", "zero"]] = None


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

    # Nuevos (opcionales)
    text_feats: str | None = None
    text_feats_out_dir: str | None = None
    empty_text_policy: str | None = None


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
                "--meta-list",
                DEFAULT_META_LIST,
            ]
            subprocess.run(cmd_norm, check=True)

        # 2) Ejecutar BETO sobre el parquet normalizado
        min_tokens = int(job.get("min_tokens", 1))

        # Flags nuevos (opcionales) para embeddings y política NO_TEXT
        text_feats = (job.get("text_feats") or None)
        if text_feats in ("none", "", "null"):
            text_feats = None

        text_feats_out_dir = job.get("text_feats_out_dir")
        if text_feats == "tfidf_lsa" and not text_feats_out_dir:
            # Default reproducible
            text_feats_out_dir = str(BASE_DIR / "artifacts" / "textfeats" / job["dataset"])

        empty_text_policy = job.get("empty_text_policy")
        if empty_text_policy is None:
            empty_text_policy = "neutral" if keep_empty_text else "zero"

        cmd_beto = [
            sys.executable,
            "-m",
            "neurocampus.app.jobs.cmd_preprocesar_beto",
            "--in", src,
            "--out", dst,
            "--text-col", text_col,
            "--beto-mode", "probs",
            "--min-tokens", str(min_tokens),
        ]

        if text_feats:
            cmd_beto += ["--text-feats", str(text_feats)]
        if text_feats_out_dir:
            cmd_beto += ["--text-feats-out-dir", str(text_feats_out_dir)]
        if empty_text_policy:
            cmd_beto += ["--empty-text-policy", str(empty_text_policy)]

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
                "text_feats_out_dir": meta.get("text_feats_out_dir"),
                "empty_text_policy": meta.get("empty_text_policy"),
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

    # Defaults reproducibles para flags nuevos (no rompen compatibilidad)
    text_feats_out_dir = req.text_feats_out_dir
    if req.text_feats == "tfidf_lsa" and not text_feats_out_dir:
        text_feats_out_dir = str(BASE_DIR / "artifacts" / "textfeats" / req.dataset)

    empty_text_policy = req.empty_text_policy
    if empty_text_policy is None:
        empty_text_policy = "neutral" if req.keep_empty_text else "zero"

    job_id = f"beto-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    needs_cargar_dataset = False
    raw_src: Optional[Path] = None

    # Intentamos ubicar el dataset crudo SIEMPRE (sirve para "upgrade" del processed)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    raw_parquet = DATASETS_DIR / f"{req.dataset}.parquet"
    raw_csv = DATASETS_DIR / f"{req.dataset}.csv"

    if raw_parquet.exists():
        raw_src = raw_parquet
    elif raw_csv.exists():
        raw_src = raw_csv

    if not processed_path.exists():
        # No hay dataset normalizado todavía → hay que normalizar sí o sí
        if raw_src is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No existe dataset procesado en {processed_path} ni dataset crudo en "
                    f"{raw_parquet} / {raw_csv}. Sube un dataset desde /datos o "
                    f"genera el procesado manualmente."
                ),
            )
        needs_cargar_dataset = True
    else:
        # Hay processed, pero puede estar incompleto (sin profesor/docente/teacher)
        # Si existe raw, lo podemos regenerar correctamente.
        if raw_src is not None and _processed_missing_docente_cols(processed_path):
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
        "min_tokens": req.min_tokens,
        "text_feats": req.text_feats,
        "text_feats_out_dir": text_feats_out_dir,
        "empty_text_policy": empty_text_policy,
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

# ---------------------------------------------------------------------------
# Jobs del dominio Datos: Unificación histórica + Feature-pack
# ---------------------------------------------------------------------------

class DataUnifyRequest(BaseModel):
    """Request para lanzar un job de unificación histórica desde **Datos**.

    Modos soportados
    ----------------
    - ``acumulado``: genera historico/unificado.parquet
    - ``acumulado_labeled``: genera historico/unificado_labeled.parquet
    - ``periodo_actual``: genera historico/periodo_actual/<periodo>.parquet
    - ``ventana``: genera historico/ventanas/unificado_<tag>.parquet

    Para ``ventana``:
    - Usa `ultimos=N` o bien (`desde` y `hasta`).
    """
    mode: Literal["acumulado", "acumulado_labeled", "periodo_actual", "ventana"] = "acumulado"
    ultimos: Optional[int] = None
    desde: Optional[str] = None
    hasta: Optional[str] = None


class DataUnifyJob(BaseModel):
    """Estado de un job de unificación histórica."""
    id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    mode: str
    out_uri: Optional[str] = None
    meta: Optional[dict] = None
    error: Optional[str] = None


def _data_unify_job_path(job_id: str) -> Path:
    """Ruta al archivo JSON de un job de unificación."""
    return DATA_UNIFY_JOBS_DIR / f"{job_id}.json"


def _load_data_unify_job(job_id: str) -> dict:
    """Carga un job de unificación desde disco."""
    path = _data_unify_job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_data_unify_job(job: dict) -> None:
    """Guarda un job de unificación en disco."""
    DATA_UNIFY_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _data_unify_job_path(job["id"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _list_data_unify_jobs(limit: int = 50) -> list[dict]:
    """Lista jobs de unificación recientes."""
    if not DATA_UNIFY_JOBS_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(DATA_UNIFY_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out[:limit]


def _run_data_unify_job(job_id: str) -> None:
    """Ejecuta un job de unificación histórica en background."""
    job = _load_data_unify_job(job_id)
    job["status"] = "running"
    job["started_at"] = _now_iso()
    _save_data_unify_job(job)

    try:
        # Import lazy para no penalizar startup
        from neurocampus.data.strategies.unificacion import UnificacionStrategy

        strat = UnificacionStrategy(base_uri="localfs://.")
        mode = str(job.get("mode", "acumulado"))

        if mode == "acumulado":
            out_uri, meta = strat.acumulado()
        elif mode == "acumulado_labeled":
            out_uri, meta = strat.acumulado_labeled()
        elif mode == "periodo_actual":
            out_uri, meta = strat.periodo_actual()
        elif mode == "ventana":
            out_uri, meta = strat.ventana(
                ultimos=job.get("ultimos"),
                desde=job.get("desde"),
                hasta=job.get("hasta"),
            )
        else:
            raise ValueError(f"mode inválido: {mode}")

        job["status"] = "done"
        job["finished_at"] = _now_iso()
        job["out_uri"] = out_uri
        job["meta"] = meta

    except Exception as e:
        job["status"] = "failed"
        job["finished_at"] = _now_iso()
        job["error"] = str(e)

    _save_data_unify_job(job)


@router.post(
    "/data/unify/run",
    response_model=DataUnifyJob,
    summary="Lanza un job de unificación histórica (historico/*)",
)
def launch_data_unify(req: DataUnifyRequest, background: BackgroundTasks) -> DataUnifyJob:
    """Crea un job de unificación y lo ejecuta en background."""
    job_id = f"unify-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    job = {
        "id": job_id,
        "status": "created",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "mode": req.mode,
        "ultimos": req.ultimos,
        "desde": req.desde,
        "hasta": req.hasta,
        "out_uri": None,
        "meta": None,
        "error": None,
    }
    _save_data_unify_job(job)
    background.add_task(_run_data_unify_job, job_id)
    return DataUnifyJob(**job)


@router.get(
    "/data/unify/{job_id}",
    response_model=DataUnifyJob,
    summary="Devuelve el estado de un job de unificación histórica",
)
def get_data_unify_job(job_id: str) -> DataUnifyJob:
    job = _load_data_unify_job(job_id)
    return DataUnifyJob(**job)


@router.get(
    "/data/unify",
    response_model=list[DataUnifyJob],
    summary="Lista jobs de unificación recientes",
)
def list_data_unify_jobs(limit: int = 20) -> list[DataUnifyJob]:
    jobs = _list_data_unify_jobs(limit=limit)
    return [DataUnifyJob(**j) for j in jobs]


class FeaturesPrepareRequest(BaseModel):
    """Request para crear el feature-pack persistente para entrenamiento.

    - dataset_id: id lógico (periodo o etiqueta) que define la carpeta de salida:
        artifacts/features/<dataset_id>/

    - input_uri (opcional): fuente de datos etiquetados.
      Si no se especifica, se intenta:
        1) historico/unificado_labeled.parquet
        2) data/labeled/<dataset_id>_beto.parquet
    """
    dataset_id: str
    input_uri: Optional[str] = None
    output_dir: Optional[str] = None


class FeaturesPrepareJob(BaseModel):
    """Estado de un job de feature-pack."""
    id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    dataset_id: str
    input_uri: Optional[str] = None
    output_dir: Optional[str] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None


def _features_job_path(job_id: str) -> Path:
    """Ruta al archivo JSON de un job de feature-pack."""
    return FEATURES_PREP_JOBS_DIR / f"{job_id}.json"


def _load_features_job(job_id: str) -> dict:
    """Carga un job de feature-pack desde disco."""
    path = _features_job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_features_job(job: dict) -> None:
    """Guarda un job de feature-pack en disco."""
    FEATURES_PREP_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _features_job_path(job["id"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _list_features_jobs(limit: int = 50) -> list[dict]:
    """Lista jobs de feature-pack recientes."""
    if not FEATURES_PREP_JOBS_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(FEATURES_PREP_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out[:limit]


def _resolve_features_input(dataset_id: str, input_uri: Optional[str]) -> str:
    """Resuelve la fuente de datos etiquetados para el feature-pack."""
    if input_uri:
        return input_uri

    hist = BASE_DIR / "historico" / "unificado_labeled.parquet"
    if hist.exists():
        return "historico/unificado_labeled.parquet"

    labeled = DATA_LABELED_DIR / f"{dataset_id}_beto.parquet"
    if labeled.exists():
        return f"data/labeled/{dataset_id}_beto.parquet"

    raise FileNotFoundError(
        "No se encontró input etiquetado para feature-pack. "
        "Corre BETO (data/labeled/<periodo>_beto.parquet) o genera historico/unificado_labeled.parquet."
    )


def _run_features_prepare_job(job_id: str) -> None:
    """Ejecuta el job de feature-pack en background."""
    job = _load_features_job(job_id)
    job["status"] = "running"
    job["started_at"] = _now_iso()
    _save_features_job(job)

    try:
        dataset_id = str(job["dataset_id"])
        input_uri = _resolve_features_input(dataset_id, job.get("input_uri"))

        out_dir = job.get("output_dir")
        if not out_dir:
            out_dir = str(BASE_DIR / "artifacts" / "features" / dataset_id)

        # Import lazy (este módulo se crea en el Paso 4)
        try:
            from neurocampus.data.features_prepare import prepare_feature_pack
        except Exception as e:
            raise RuntimeError(
                "No se pudo importar neurocampus.data.features_prepare.prepare_feature_pack. "
                "Asegúrate de aplicar el Paso 4 antes de ejecutar este job."
            ) from e

        artifacts = prepare_feature_pack(
            base_dir=BASE_DIR,
            dataset_id=dataset_id,
            input_uri=input_uri,
            output_dir=out_dir,
        )

        job["status"] = "done"
        job["finished_at"] = _now_iso()
        job["input_uri"] = input_uri
        job["output_dir"] = out_dir
        job["artifacts"] = artifacts

    except Exception as e:
        job["status"] = "failed"
        job["finished_at"] = _now_iso()
        job["error"] = str(e)

    _save_features_job(job)


@router.post(
    "/data/features/prepare/run",
    response_model=FeaturesPrepareJob,
    summary="Lanza un job para crear artifacts/features/<dataset_id>/train_matrix.parquet",
)
def launch_features_prepare(req: FeaturesPrepareRequest, background: BackgroundTasks) -> FeaturesPrepareJob:
    """Crea un job de feature-pack y lo ejecuta en background."""
    job_id = f"feat-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    job = {
        "id": job_id,
        "status": "created",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "dataset_id": req.dataset_id,
        "input_uri": req.input_uri,
        "output_dir": req.output_dir,
        "artifacts": None,
        "error": None,
    }
    _save_features_job(job)
    background.add_task(_run_features_prepare_job, job_id)
    return FeaturesPrepareJob(**job)


@router.get(
    "/data/features/prepare/{job_id}",
    response_model=FeaturesPrepareJob,
    summary="Devuelve el estado de un job de feature-pack",
)
def get_features_prepare_job(job_id: str) -> FeaturesPrepareJob:
    job = _load_features_job(job_id)
    return FeaturesPrepareJob(**job)


@router.get(
    "/data/features/prepare",
    response_model=list[FeaturesPrepareJob],
    summary="Lista jobs de feature-pack recientes",
)
def list_features_prepare_jobs(limit: int = 20) -> list[FeaturesPrepareJob]:
    jobs = _list_features_jobs(limit=limit)
    return [FeaturesPrepareJob(**j) for j in jobs]

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
