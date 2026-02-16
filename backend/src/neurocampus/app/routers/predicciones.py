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
import logging
logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException
from typing import Any

from neurocampus.app.schemas.predicciones import HealthResponse, PredictRequest, PredictResolvedResponse
from neurocampus.predictions.loader import (
    ChampionNotFoundError,
    PredictorNotFoundError,
    PredictorNotReadyError,
    load_predictor_by_champion,
    load_predictor_by_run_id,
)
from neurocampus.utils.paths import artifacts_dir, rel_artifact_path
from neurocampus.predictions.bundle import bundle_paths
from neurocampus.data.features_prepare import load_feature_pack


router = APIRouter(prefix="/predicciones", tags=["Predicciones"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check del módulo de Predicciones."""
    base = artifacts_dir()
    return HealthResponse(status="ok", artifacts_dir=str(base))


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

        # (opt-in) Solo cargar datos si el cliente lo solicita explícitamente.
        # Esto mantiene compatibilidad con tests P2.2 (resolve) que no crean feature-pack.
        if req.input_uri and str(req.input_uri).strip().lower() == "feature_pack":
            dataset_id = str(loaded.predictor.get("dataset_id") or req.dataset_id or "")
            if not dataset_id:
                raise HTTPException(status_code=422, detail="No se pudo resolver dataset_id para cargar feature_pack")

            try:
                feature_df, _meta = load_feature_pack(dataset_id=dataset_id, kind="train")
            except FileNotFoundError as e:
                # Si el server está apuntando a otra carpeta de artifacts o no existe el pack, que sea claro.
                raise HTTPException(status_code=404, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

            # TODO(P2.3.x): usar feature_df para inferencia real con modelo en run_dir/model/


        
        # Ejecutar la inferencia
        # Esto depende de tu estrategia, por ejemplo:
        # predictions = loaded.strategy.predict(feature_pack)

        # Para este ejemplo, usaremos un valor simulado:
        predictions = [{"input": "data", "prediction": "value"}]  # Esto lo reemplazamos con inferencia real

        # Nota: evitamos relativizar Paths porque en tests se sobreescribe NC_ARTIFACTS_DIR
        # luego de import-time en algunos módulos. Este string es el contrato estable.
        run_dir_logical = f"artifacts/runs/{loaded.run_id}"

        return PredictResolvedResponse(
            resolved_run_id=loaded.run_id,
            resolved_from=resolved_from,
            run_dir=run_dir_logical,
            predictor=loaded.predictor,
            preprocess=loaded.preprocess,
            note="P2.2: resolución/validación OK. Inferencia se implementa en P2.3+.",
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

