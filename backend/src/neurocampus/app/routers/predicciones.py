"""
Router de Predicciones (P2.2).

En P2.2 el endpoint `/predicciones/predict` NO hace inferencia real aún.
Su objetivo es:
- resolver run_id (directo o vía champion),
- cargar y validar el predictor bundle,
- retornar metadata (predictor.json + preprocess.json).

En P2.3/P2.4 se agregará inferencia y escritura de outputs.
"""

from __future__ import annotations

import os
import json
import logging
logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Any

from neurocampus.app.schemas.predicciones import (
    HealthResponse,
    PredictRequest,
    PredictResolvedResponse,
    ModelInfoResponse,
    PredictionsPreviewResponse,
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
    predict_from_feature_pack,
    save_predictions_parquet,
    resolve_predictions_parquet_path,
    load_predictions_preview,
)
from neurocampus.utils.paths import artifacts_dir, rel_artifact_path
from neurocampus.predictions.bundle import bundle_paths
from neurocampus.data.features_prepare import load_feature_pack
from neurocampus.utils.model_context import fill_context


router = APIRouter(prefix="/predicciones", tags=["Predicciones"])


def _apply_ctx_to_manifest(predictor: dict, ctx: dict) -> dict:
    """Aplica `ctx` al dict predictor.json para evitar null/unknown en campos críticos.

    Importante: esto NO reescribe predictor.json en disco; solo ajusta el response.
    """
    out = dict(predictor or {})

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
    extra = dict(extra)
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

    return out


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check del módulo de Predicciones."""
    base = artifacts_dir()
    return HealthResponse(status="ok", artifacts_dir=str(base))



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

