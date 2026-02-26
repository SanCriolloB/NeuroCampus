"""neurocampus.app.routers.predicciones
=======================================

Router de predicciones para la pestaña **Predicciones** del frontend.

Flujo completo:
- Endpoints de listing: datasets, teachers, materias.
- Predicción individual por par docente–materia.
- Job de predicción por lote con polling de estado.
- Endpoints heredados de P2.2/P2.4: predict, model-info, outputs.

Todos los endpoints de este router usan exclusivamente la family
``score_docente`` (regresión 0–50). El champion se selecciona
automáticamente por ``dataset_id``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from neurocampus.app.schemas.predicciones import (
    BatchJobResponse,
    BatchRunRequest,
    DatasetInfoResponse,
    HealthResponse,
    IndividualPredictionRequest,
    IndividualPredictionResponse,
    MateriaInfoResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResolvedResponse,
    PredictionsPreviewResponse,
    PredictionRunInfoResponse,
    TeacherInfoResponse,
)
from neurocampus.predictions.loader import (
    ChampionNotFoundError,
    PredictorNotFoundError,
    PredictorNotReadyError,
    load_predictor_by_champion,
    load_predictor_by_run_id,
)
from neurocampus.services.predictions_service import (
    InferenceNotAvailableError,
    load_inference_model,
    predict_dataframe,
    predict_from_feature_pack,
    save_predictions_parquet,
    resolve_predictions_parquet_path,
    load_predictions_preview,
)
from neurocampus.utils.model_context import fill_context
from neurocampus.utils.paths import artifacts_dir, rel_artifact_path
from neurocampus.utils.score_postprocess import (
    build_comparison,
    build_radar,
    compute_confidence,
    compute_risk,
    INDICATOR_NAMES,
)
from neurocampus.utils.predictions_run_io import (
    create_pred_run_dir,
    list_pred_runs,
    write_pred_meta,
)
from neurocampus.predictions.bundle import bundle_paths
from neurocampus.data.features_prepare import load_feature_pack

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predicciones", tags=["Predicciones"])

# ---------------------------------------------------------------------------
# Estado in-memory para jobs de batch (patrón idéntico a modelos.py)
# ---------------------------------------------------------------------------

#: Diccionario de estado por job_id para el polling de /batch/{job_id}.
_PRED_ESTADOS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Constantes y helpers internos
# ---------------------------------------------------------------------------

_FAMILY = "score_docente"
"""Family fija para todos los endpoints de predicción de esta pestaña."""

def _period_key(ds: str) -> tuple:
    """Ordena dataset_ids tipo 'YYYY-N' cronológicamente."""
    m = re.match(r"^\s*(\d{4})-(\d{1,2})\s*$", str(ds))
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _list_pair_datasets() -> List[str]:
    """Lista dataset_ids que tienen pair_matrix.parquet disponible."""
    base = artifacts_dir() / "features"
    if not base.exists():
        return []
    return sorted(
        [p.name for p in base.iterdir() if (p / "pair_matrix.parquet").exists()],
        key=_period_key,
    )


def _champion_exists(dataset_id: str) -> bool:
    """Verifica si hay un champion score_docente para el dataset."""
    try:
        load_predictor_by_champion(dataset_id=dataset_id, family=_FAMILY, use_cache=False)
        return True
    except Exception:
        return False


def _get_calif_means(row: pd.Series) -> List[float]:
    """Extrae mean_calif_1..10 de una fila del pair_matrix."""
    result: List[float] = []
    for i in range(1, len(INDICATOR_NAMES) + 1):
        col = f"mean_calif_{i}"
        if col in row.index:
            try:
                result.append(float(row[col]))
            except (TypeError, ValueError):
                result.append(0.0)
    return result


def _get_cohorte_means(df: pd.DataFrame, materia_key: str) -> List[float]:
    """Promedio por dimensión para todos los pares de una materia."""
    subset = df[df["materia_key"] == materia_key]
    result: List[float] = []
    for i in range(1, len(INDICATOR_NAMES) + 1):
        col = f"mean_calif_{i}"
        if col in df.columns and len(subset) > 0:
            result.append(float(subset[col].mean()))
        else:
            result.append(0.0)
    return result
    # Campos top-level del manifest
    if ctx.get("dataset_id") is not None:
        out["dataset_id"] = str(ctx["dataset_id"])
    if ctx.get("model_name") is not None:
        out["model_name"] = str(ctx["model_name"])
    if ctx.get("task_type") is not None:
        out["task_type"] = str(ctx["task_type"])
    if ctx.get("input_level") is not None:
        out["input_level"] = str(ctx["input_level"])

    # target_col debe evitar null cuando sea razonable
    if ctx.get("target_col") is not None:
        out["target_col"] = str(ctx["target_col"])
    if out.get("target_col") is None:
        out["target_col"] = "target"

    # Campos extra (family y otros)
    extra = out.get("extra") if isinstance(out.get("extra"), dict) else {}
    # limpiar nulls heredados del predictor.json para no romper UI
    extra = {k: v for k, v in dict(extra).items() if v is not None}
    if ctx.get("family") is not None:
        extra["family"] = str(ctx["family"])
    if ctx.get("data_source") is not None:
        extra["data_source"] = str(ctx["data_source"])
    else:
        extra["data_source"] = str(extra.get("data_source") or "feature_pack")

    for k in ("data_plan", "split_mode", "val_ratio", "target_mode"):
        v = ctx.get(k)
        if v is not None:
            extra[k] = v

    if extra:
        out["extra"] = extra


def _apply_ctx_to_manifest(predictor: dict, ctx: dict) -> dict:
    """Aplica contexto resuelto al manifest del predictor para eliminar nulls.

    En algunos runs legacy, ``predictor.json`` puede traer valores como
    ``"unknown"`` o ``"null"`` en campos críticos (``task_type``, ``input_level``, ...).
    Esos valores son *truthy* en Python, por lo que un chequeo simple
    ``if not out.get(field)`` no los reemplaza.

    Esta función aplica el contexto resuelto por
    :func:`neurocampus.utils.model_context.fill_context` y considera explícitamente
    ciertos literales como *ausentes*.

    Notes
    -----
    - Mantener esta lógica aquí permite que tanto ``/predicciones/model-info``
      como ``/predicciones/predict`` devuelvan metadata limpia para UI.
    """
    missing = {"", "none", "null", "unknown", "n/a"}

    def _is_missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip().lower() in missing
        return False

    out = dict(predictor or {})
    for field in ("task_type", "input_level", "target_col", "family", "dataset_id", "model_name"):
        if _is_missing(out.get(field)) and ctx.get(field):
            out[field] = ctx[field]

    # Backfill en extra.* (útil para UI legacy que lee predictor.extra)
    extra = out.get("extra") if isinstance(out.get("extra"), dict) else {}
    for key in ("family", "dataset_id", "model_name", "data_source", "data_plan", "split_mode", "target_mode", "val_ratio"):
        if _is_missing(extra.get(key)) and ctx.get(key) is not None:
            extra[key] = ctx.get(key)
    if extra:
        out["extra"] = extra

    return out

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check del módulo de Predicciones."""
    base = artifacts_dir()
    return HealthResponse(status="ok", artifacts_dir=str(base))

# ===========================================================================
# ENDPOINTS DE LISTING — pestaña Predicciones
# ===========================================================================

@router.get(
    "/datasets",
    response_model=List[DatasetInfoResponse],
    summary="Lista datasets disponibles para predicción score_docente",
)
def list_datasets() -> List[DatasetInfoResponse]:
    result: List[DatasetInfoResponse] = []
    for ds in _list_pair_datasets():
        pair_meta: Dict[str, Any] = {}
        pm_path = artifacts_dir() / "features" / ds / "pair_meta.json"
        if pm_path.exists():
            try:
                pair_meta = json.loads(pm_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        result.append(
            DatasetInfoResponse(
                dataset_id=ds,
                n_pairs=int(pair_meta.get("n_pairs", 0)),
                n_docentes=int(pair_meta.get("n_docentes", 0)),
                n_materias=int(pair_meta.get("n_materias", 0)),
                has_champion=_champion_exists(ds),
                created_at=pair_meta.get("created_at"),
            )
        )
    return result


@router.get(
    "/runs",
    response_model=List[PredictionRunInfoResponse],
    summary="Lista runs batch persistidos de predicción para un dataset",
)
def list_runs(dataset_id: str) -> List[PredictionRunInfoResponse]:
    """Lista runs persistidos para el bloque **Historial**.

    Este endpoint lee ``meta.json`` de cada run bajo ``artifacts/predictions/``
    y retorna información mínima para que el frontend pueda:

    - abrir vista previa (``/predicciones/outputs/preview``)
    - descargar el parquet (``/predicciones/outputs/file``)

    Notes
    -----
    - Para runs antiguos cuyo ``meta.json`` no incluya ``predictions_uri``, se
      intenta inferir el path estándar ``predictions.parquet``.
    """
    ds = str(dataset_id or "").strip()
    if not ds:
        raise HTTPException(status_code=422, detail="dataset_id es requerido")

    metas = list_pred_runs(ds)
    out: List[PredictionRunInfoResponse] = []

    for meta in metas:
        pred_run_id = str(meta.get("pred_run_id") or "")
        predictions_uri = meta.get("predictions_uri")

        # Compat: inferir predictions_uri cuando no está en meta.json
        if (not predictions_uri) and pred_run_id:
            candidate = artifacts_dir() / "predictions" / ds / _FAMILY / pred_run_id / "predictions.parquet"
            if candidate.exists():
                predictions_uri = rel_artifact_path(candidate)
            else:
                legacy = artifacts_dir() / "predictions" / ds / _FAMILY / pred_run_id / "predicciones.parquet"
                if legacy.exists():
                    predictions_uri = rel_artifact_path(legacy)

        out.append(
            PredictionRunInfoResponse(
                pred_run_id=pred_run_id,
                dataset_id=str(meta.get("dataset_id") or ds),
                family=str(meta.get("family") or _FAMILY),
                created_at=meta.get("created_at"),
                n_pairs=int(meta.get("n_pairs") or 0),
                champion_run_id=meta.get("champion_run_id"),
                model_name=meta.get("model_name"),
                predictions_uri=predictions_uri,
            )
        )

    return out


@router.get(
    "/teachers",

    response_model=List[TeacherInfoResponse],
    summary="Lista docentes únicos de un dataset",
)
def list_teachers(dataset_id: str) -> List[TeacherInfoResponse]:
    feat_dir = artifacts_dir() / "features" / str(dataset_id)
    idx_path = feat_dir / "teacher_index.json"

    if not idx_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"teacher_index.json no encontrado para dataset_id={dataset_id}. "
                "Ejecuta feature-pack/prepare primero."
            ),
        )

    index: Dict[str, int] = json.loads(idx_path.read_text(encoding="utf-8"))

    counts: Dict[str, int] = {}
    pair_path = feat_dir / "pair_matrix.parquet"
    if pair_path.exists():
        try:
            df_pair = pd.read_parquet(pair_path, columns=["teacher_key", "n_docente"])
            counts = (
                df_pair.drop_duplicates("teacher_key")
                .set_index("teacher_key")["n_docente"]
                .astype(int)
                .to_dict()
            )
        except Exception:
            pass

    return [
        TeacherInfoResponse(
            teacher_key=key,
            teacher_id=tid,
            n_encuestas=int(counts.get(key, 0)),
        )
        for key, tid in sorted(index.items())
    ]


@router.get(
    "/materias",
    response_model=List[MateriaInfoResponse],
    summary="Lista materias únicas de un dataset",
)
def list_materias(dataset_id: str) -> List[MateriaInfoResponse]:
    feat_dir = artifacts_dir() / "features" / str(dataset_id)
    idx_path = feat_dir / "materia_index.json"

    if not idx_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"materia_index.json no encontrado para dataset_id={dataset_id}. "
                "Ejecuta feature-pack/prepare primero."
            ),
        )

    index: Dict[str, int] = json.loads(idx_path.read_text(encoding="utf-8"))

    counts: Dict[str, int] = {}
    pair_path = feat_dir / "pair_matrix.parquet"
    if pair_path.exists():
        try:
            df_pair = pd.read_parquet(pair_path, columns=["materia_key", "n_materia"])
            counts = (
                df_pair.drop_duplicates("materia_key")
                .set_index("materia_key")["n_materia"]
                .astype(int)
                .to_dict()
            )
        except Exception:
            pass

    return [
        MateriaInfoResponse(
            materia_key=key,
            materia_id=mid,
            n_encuestas=int(counts.get(key, 0)),
        )
        for key, mid in sorted(index.items())
    ]


# ===========================================================================
# PREDICCIÓN INDIVIDUAL
# ===========================================================================

@router.post(
    "/individual",
    response_model=IndividualPredictionResponse,
    summary="Predicción individual de score para un par docente–materia",
)
def predict_individual(req: IndividualPredictionRequest) -> IndividualPredictionResponse:
    ds = str(req.dataset_id).strip()
    teacher_key = str(req.teacher_key).strip()
    materia_key = str(req.materia_key).strip()

    pair_path = artifacts_dir() / "features" / ds / "pair_matrix.parquet"
    if not pair_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"pair_matrix.parquet no existe para dataset_id={ds}. "
                "Ejecuta feature-pack/prepare primero."
            ),
        )

    try:
        df_pair = pd.read_parquet(pair_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando pair_matrix: {e}") from e

    mask = (df_pair["teacher_key"] == teacher_key) & (df_pair["materia_key"] == materia_key)
    row_df = df_pair[mask]
    if len(row_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Par teacher_key='{teacher_key}' / materia_key='{materia_key}' "
                f"no encontrado en dataset_id={ds}."
            ),
        )

    row = row_df.iloc[0]

    try:
        bundle = load_predictor_by_champion(dataset_id=ds, family=_FAMILY)
    except ChampionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorNotReadyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando champion: {e}") from e

    champion_run_id = str(bundle.run_id)
    model_name = str(bundle.predictor.get("model_name") or "unknown")

    try:
        model = load_inference_model(bundle)
        preds, _ = predict_dataframe(model, row_df.reset_index(drop=True), return_proba=False)
    except InferenceNotAvailableError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error en inferencia individual para %s/%s", teacher_key, materia_key)
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}") from e

    score_total_pred = float(np.clip(preds[0]["score_total_pred"], 0.0, 50.0))

    n_par = int(row.get("n_par", 0))
    n_docente = int(row.get("n_docente", 0))
    n_materia = int(row.get("n_materia", 0))
    mean_score = float(row.get("mean_score_total_0_50", 0.0) or 0.0)
    std_score = float(row.get("std_score_total_0_50", 0.0) or 0.0)

    risk = compute_risk(score_total_pred)
    confidence = compute_confidence(n_par=n_par, std_score=std_score)

    calif_means_docente = _get_calif_means(row)
    calif_means_cohorte = _get_cohorte_means(df_pair, materia_key)

    radar = build_radar(
        calif_means=calif_means_docente,
        score_total_pred=score_total_pred,
        mean_score_total=mean_score,
    )
    comparison = build_comparison(
        calif_means_docente=calif_means_docente,
        calif_means_cohorte=calif_means_cohorte,
    )

    timeline = _build_timeline(
        teacher_key=teacher_key,
        materia_key=materia_key,
        current_ds=ds,
        score_total_pred=score_total_pred,
    )

    from neurocampus.app.schemas.predicciones import (
        EvidenceInfo,
        HistoricalStats,
        TimelinePoint,
        RadarPoint,
        ComparisonPoint,
    )

    return IndividualPredictionResponse(
        dataset_id=ds,
        teacher_key=teacher_key,
        materia_key=materia_key,
        score_total_pred=round(score_total_pred, 2),
        risk=risk,
        confidence=confidence,
        cold_pair=(n_par == 0),
        evidence=EvidenceInfo(n_par=n_par, n_docente=n_docente, n_materia=n_materia),
        historical=HistoricalStats(mean_score=round(mean_score, 2), std_score=round(std_score, 2)),
        radar=[RadarPoint(**p) for p in radar],
        comparison=[ComparisonPoint(**p) for p in comparison],
        timeline=[TimelinePoint(**p) for p in timeline],
        champion_run_id=champion_run_id,
        model_name=model_name,
    )


def _build_timeline(
    *,
    teacher_key: str,
    materia_key: str,
    current_ds: str,
    score_total_pred: float,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for ds in _list_pair_datasets():
        pair_path = artifacts_dir() / "features" / ds / "pair_matrix.parquet"
        if not pair_path.exists():
            continue
        try:
            df = pd.read_parquet(
                pair_path,
                columns=["teacher_key", "materia_key", "mean_score_total_0_50", "n_par"],
            )
        except Exception:
            continue

        mask = (df["teacher_key"] == teacher_key) & (df["materia_key"] == materia_key)
        sub = df[mask]
        if len(sub) == 0:
            continue

        point: Dict[str, Any] = {
            "semester": ds,
            "real": round(float(sub["mean_score_total_0_50"].iloc[0]), 2),
        }
        if ds == current_ds:
            point["predicted"] = round(float(score_total_pred), 2)

        result.append(point)

    return result


# ===========================================================================
# PREDICCIÓN POR LOTE (batch) — job asíncrono con polling
# ===========================================================================

@router.post(
    "/batch/run",
    response_model=BatchJobResponse,
    status_code=202,
    summary="Lanza un job de predicción por lote (todos los pares del dataset)",
)
def batch_run(req: BatchRunRequest, bg: BackgroundTasks) -> BatchJobResponse:
    ds = str(req.dataset_id).strip()

    pair_path = artifacts_dir() / "features" / ds / "pair_matrix.parquet"
    if not pair_path.exists():
        raise HTTPException(status_code=404, detail=f"pair_matrix.parquet no existe para dataset_id={ds}.")
    if not _champion_exists(ds):
        raise HTTPException(
            status_code=404,
            detail=f"No hay champion score_docente para dataset_id={ds}.",
        )

    job_id = str(uuid.uuid4())
    _PRED_ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "dataset_id": ds,
        "pred_run_id": None,
        "n_pairs": None,
        "predictions_uri": None,
        "champion_run_id": None,
        "error": None,
    }

    bg.add_task(_run_batch_job, job_id, ds)

    return BatchJobResponse(job_id=job_id, status="queued", progress=0.0, dataset_id=ds)


@router.get(
    "/batch/{job_id}",
    response_model=BatchJobResponse,
    summary="Estado de un job de predicción por lote",
)
def batch_status(job_id: str) -> BatchJobResponse:
    st = _PRED_ESTADOS.get(job_id)
    if not isinstance(st, dict):
        raise HTTPException(status_code=404, detail=f"job_id={job_id} no encontrado.")

    return BatchJobResponse(
        job_id=job_id,
        status=str(st.get("status", "unknown")),
        progress=float(st.get("progress", 0.0)),
        dataset_id=str(st.get("dataset_id", "")),
        pred_run_id=st.get("pred_run_id"),
        n_pairs=st.get("n_pairs"),
        predictions_uri=st.get("predictions_uri"),
        champion_run_id=st.get("champion_run_id"),
        error=st.get("error"),
    )


def _run_batch_job(job_id: str, dataset_id: str) -> None:
    st = _PRED_ESTADOS[job_id]
    st["status"] = "running"
    st["progress"] = 0.05

    try:
        df_pair = pd.read_parquet(artifacts_dir() / "features" / dataset_id / "pair_matrix.parquet")
        st["progress"] = 0.15

        bundle = load_predictor_by_champion(dataset_id=dataset_id, family=_FAMILY, use_cache=False)
        champion_run_id = str(bundle.run_id)
        model_name = str(getattr(bundle, "predictor", {}).get("model_name") or "unknown")
        model = load_inference_model(bundle)
        st["champion_run_id"] = champion_run_id
        st["progress"] = 0.30

        preds_raw, _ = predict_dataframe(model, df_pair.reset_index(drop=True), return_proba=False)
        st["progress"] = 0.70

        records: List[Dict[str, Any]] = []
        for i, pred in enumerate(preds_raw):
            row = df_pair.iloc[i]
            score = float(np.clip(pred["score_total_pred"], 0.0, 50.0))
            n_par = int(row.get("n_par", 0))
            std_score = float(row.get("std_score_total_0_50", 0.0) or 0.0)

            records.append(
                {
                    "teacher_key": str(row.get("teacher_key", "")),
                    "materia_key": str(row.get("materia_key", "")),
                    "teacher_id": int(row.get("teacher_id", -1)),
                    "materia_id": int(row.get("materia_id", -1)),
                    "score_total_pred": round(score, 2),
                    "risk": compute_risk(score),
                    "confidence": compute_confidence(n_par=n_par, std_score=std_score),
                    "n_par": n_par,
                    "n_docente": int(row.get("n_docente", 0)),
                    "n_materia": int(row.get("n_materia", 0)),
                    "cold_pair": n_par == 0,
                    "mean_score_total_0_50": float(row.get("mean_score_total_0_50", 0.0) or 0.0),
                    "std_score_total_0_50": round(std_score, 3),
                }
            )

        st["progress"] = 0.85

        pred_run_id, out_dir = create_pred_run_dir(dataset_id)
        out_parquet = out_dir / "predictions.parquet"
        pd.DataFrame(records).to_parquet(out_parquet, index=False)

        import datetime as dt

        predictions_uri = rel_artifact_path(out_parquet)
        meta = {
            "pred_run_id": pred_run_id,
            "dataset_id": dataset_id,
            "champion_run_id": champion_run_id,
            "model_name": model_name,
            "n_pairs": len(records),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "family": _FAMILY,
            "predictions_uri": predictions_uri,
        }
        write_pred_meta(out_dir, meta)

        st["status"] = "completed"
        st["progress"] = 1.0
        st["pred_run_id"] = pred_run_id
        st["n_pairs"] = len(records)
        st["predictions_uri"] = predictions_uri

    except Exception as e:
        logger.exception("Error en batch job %s", job_id)
        st["status"] = "failed"
        st["error"] = str(e)

@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    run_id: str | None = None,
    dataset_id: str | None = None,
    family: str | None = None,
    use_champion: bool = False,
) -> ModelInfoResponse:
    """Retorna metadata del modelo (P2.2: resolve/validate sin inferencia).

    Permite a clientes (p.ej. frontend) consultar qué predictor se usará y con qué contrato.

    Errores esperados:
    - 404: champion.json o predictor bundle no existe.
    - 422: request inválido o predictor no listo.
    """
    try:
        if use_champion:
            if not dataset_id:
                raise HTTPException(status_code=422, detail="dataset_id es requerido cuando use_champion=true")
            loaded = load_predictor_by_champion(dataset_id=dataset_id, family=family)
            resolved_from = "champion"
        else:
            if not run_id:
                raise HTTPException(status_code=422, detail="run_id es requerido cuando use_champion=false")
            loaded = load_predictor_by_run_id(run_id)
            resolved_from = "run_id"

        metrics = None
        metrics_path = loaded.run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("metrics.json inválido en %s; se omite en model-info", metrics_path)


        # Backfill de contexto (P2.1): evitar null/unknown en campos críticos
        ctx = fill_context(
            family=family or None,
            dataset_id=dataset_id or (loaded.predictor.get("dataset_id") if isinstance(loaded.predictor, dict) else None),
            model_name=(loaded.predictor.get("model_name") if isinstance(loaded.predictor, dict) else None),
            metrics=metrics,
            predictor_manifest=loaded.predictor if isinstance(loaded.predictor, dict) else None,
        )
        predictor_out = _apply_ctx_to_manifest(
            loaded.predictor if isinstance(loaded.predictor, dict) else {},
            ctx,
        )
        run_dir_logical = f"artifacts/runs/{loaded.run_id}"

        return ModelInfoResponse(
            resolved_run_id=loaded.run_id,
            resolved_from=resolved_from,
            run_dir=run_dir_logical,
            predictor=predictor_out,
            preprocess=loaded.preprocess,
            metrics=metrics,
            note="P2.2: model-info (resolución/validación del bundle; sin inferencia).",
        )

    except ChampionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorNotReadyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error resolviendo predictor bundle en model-info")

        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise

        raise HTTPException(status_code=500, detail="Error interno resolviendo predictor bundle") from e


@router.post("/predict", response_model=PredictResolvedResponse)
def predict(req: PredictRequest) -> PredictResolvedResponse:
    """Resuelve y valida el predictor bundle (ahora con inferencia real en P2.3).

    Errores esperados:
    - 404: champion.json o predictor bundle no existe.
    - 422: bundle existe pero no está listo (ej. model.bin placeholder).
    """
    try:
        if req.use_champion:
            if not req.dataset_id:
                raise HTTPException(status_code=422, detail="dataset_id es requerido cuando use_champion=true")
            # Cargar el predictor desde champion
            loaded = load_predictor_by_champion(dataset_id=req.dataset_id, family=req.family)
            resolved_from = "champion"
        else:
            if not req.run_id:
                raise HTTPException(status_code=422, detail="run_id es requerido cuando use_champion=false")
            # Cargar el predictor desde run_id
            loaded = load_predictor_by_run_id(req.run_id)
            resolved_from = "run_id"


        metrics = None
        metrics_path = loaded.run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("metrics.json inválido en %s; se omite en predict", metrics_path)

        # Backfill de contexto (P2.1): evitar null/unknown en campos críticos
        ctx = fill_context(
            family=req.family or None,
            dataset_id=req.dataset_id or (loaded.predictor.get("dataset_id") if isinstance(loaded.predictor, dict) else None),
            model_name=(loaded.predictor.get("model_name") if isinstance(loaded.predictor, dict) else None),
            metrics=metrics,
            predictor_manifest=loaded.predictor if isinstance(loaded.predictor, dict) else None,
        )
        predictor_out = _apply_ctx_to_manifest(
            loaded.predictor if isinstance(loaded.predictor, dict) else {},
            ctx,
        )
        # ------------------------------------------------------------
        # P2.4: inferencia opt-in
        #
        # Para no romper P2.2/tests existentes, el endpoint solo ejecuta
        # inferencia si el cliente la solicita explícitamente.
        # - do_inference=true (nuevo)
        # - o input_uri="feature_pack" (compat)
        # ------------------------------------------------------------
        do_inference = bool(getattr(req, "do_inference", False)) or (
            req.input_uri and str(req.input_uri).strip().lower() == "feature_pack"
        )

        predictions = None
        predictions_uri = None
        out_schema = None
        warnings = None
        model_info = None
        note = "P2.2: resolución/validación OK. Inferencia deshabilitada (do_inference=false)."

        if do_inference:
            dataset_id = str(ctx.get("dataset_id") or req.dataset_id or loaded.predictor.get("dataset_id") or "")
            if not dataset_id:
                raise HTTPException(
                    status_code=422,
                    detail="No se pudo resolver dataset_id para inferir desde feature_pack",
                )

            input_level = str(req.input_level or ctx.get("input_level") or loaded.predictor.get("input_level") or "row")

            try:
                predictions, out_schema, warnings = predict_from_feature_pack(
                    bundle=loaded,
                    dataset_id=dataset_id,
                    input_level=input_level,
                    limit=int(req.limit or 50),
                    offset=int(req.offset or 0),
                    ids=req.ids,
                    return_proba=bool(req.return_proba),
                )
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            except InferenceNotAvailableError as e:
                # Bundle resuelve, pero el modelo no está cargable.
                raise HTTPException(status_code=422, detail=str(e)) from e

            model_info = {
                "dataset_id": dataset_id,
                "family": ctx.get("family") or req.family,
                "model_name": ctx.get("model_name") or loaded.predictor.get("model_name"),
                "task_type": ctx.get("task_type") or loaded.predictor.get("task_type"),
                "input_level": input_level,
                "target_col": ctx.get("target_col") or loaded.predictor.get("target_col"),
                "data_source": ctx.get("data_source"),
            }
            note = "P2.4: inferencia ejecutada desde feature_pack."


        # ------------------------------------------------------------
        # P2.4-C: persistencia opt-in
        #
        # Si `persist=true`, guardamos predictions.parquet bajo artifacts/predictions/
        # y devolvemos `predictions_uri` para consumo posterior.
        # ------------------------------------------------------------
        if bool(getattr(req, "persist", False)):
            if not do_inference:
                raise HTTPException(status_code=422, detail="persist requiere do_inference=true")

            # Intentar resolver family de forma robusta
            extra = loaded.predictor.get("extra") if isinstance(loaded.predictor, dict) else None
            fam_val = req.family
            if not fam_val and isinstance(extra, dict):
                fam_val = extra.get("family")

            paths = save_predictions_parquet(
                run_id=loaded.run_id,
                dataset_id=dataset_id,
                family=str(fam_val) if fam_val else None,
                input_level=input_level,
                predictions=predictions or [],
                schema=out_schema,
            )
            predictions_uri = rel_artifact_path(paths["predictions"])
            note = note + f" Persistido en {predictions_uri}."

        # Nota: evitamos relativizar Paths porque en tests se sobreescribe NC_ARTIFACTS_DIR
        # luego de import-time en algunos módulos. Este string es el contrato estable.
        run_dir_logical = f"artifacts/runs/{loaded.run_id}"

        return PredictResolvedResponse(
            resolved_run_id=loaded.run_id,
            resolved_from=resolved_from,
            run_dir=run_dir_logical,
            predictor=predictor_out,
            preprocess=loaded.preprocess,
            predictions=predictions,
            predictions_uri=predictions_uri,
            model_info=model_info,
            output_schema=out_schema,
            warnings=warnings,
            note=note,
        )

    except ChampionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PredictorNotReadyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        # Log completo para diagnóstico
        logger.exception("Error resolviendo predictor bundle")

        # En pytest, queremos ver el traceback real (evita 500 genérico que oculta la causa).
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise

        # En runtime normal, mantener mensaje estable (sin filtrar stacktrace al cliente).
        raise HTTPException(status_code=500, detail="Error interno resolviendo predictor bundle") from e

@router.get("/outputs/preview", response_model=PredictionsPreviewResponse)
def outputs_preview(
    predictions_uri: str,
    limit: int = 50,
    offset: int = 0,
) -> PredictionsPreviewResponse:
    """Retorna una vista previa (JSON) de un `predictions.parquet` persistido."""
    try:
        rows, columns, schema = load_predictions_preview(
            predictions_uri=predictions_uri,
            limit=limit,
            offset=offset,
        )
        return PredictionsPreviewResponse(
            predictions_uri=str(predictions_uri),
            rows=rows,
            columns=columns,
            output_schema=schema,
            note="P2.4: preview de outputs persistidos.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/outputs/file")
def outputs_file(predictions_uri: str):
    """Descarga el `predictions.parquet` persistido como archivo."""
    try:
        path = resolve_predictions_parquet_path(predictions_uri)
        return FileResponse(
            path=path,
            media_type="application/octet-stream",
            filename=path.name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

