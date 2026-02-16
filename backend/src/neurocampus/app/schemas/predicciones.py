"""
Schemas de API para Predicciones (P2).

Objetivo
--------
Definir contratos HTTP estables para:

- GET /predicciones/health
- POST /predicciones/predict (P2.2: solo valida/resolve bundle; no inferencia real aún)

Notas
-----
- En P2.2 el endpoint /predict servirá para:
  - resolver run_id por champion si aplica,
  - validar existencia del predictor bundle,
  - devolver metadata (sin ejecutar inferencia).
- En P2.3/P2.4 se agregará inferencia real y outputs (parquet/JSON).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class PredictRequest(BaseModel):
    """Request unificado para predicción.

    Modos:
    - Por run_id directo:
        {"run_id": "..."}
    - Por champion:
        {"dataset_id": "...", "family": "...", "use_champion": true}

    Nota:
    - `model_name` se deja opcional para compatibilidad futura; el loader lo obtiene del manifest.
    """

    run_id: Optional[str] = Field(default=None, description="Run ID a usar para predicción.")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID (requerido si use_champion=true).")
    family: Optional[str] = Field(default=None, description="Familia del champion (recomendado).")

    use_champion: bool = Field(
        default=False,
        description="Si true, resuelve run_id desde champion.json usando dataset_id/family.",
    )

    # Para P2.3+ (cuando haya inferencia real)
    input_uri: Optional[str] = Field(default=None, description="Fuente de datos a predecir (parquet/csv).")
    data_source: str = Field(default="feature_pack", description="Origen de datos: feature_pack (default).")

    # Alias de compat (por si clientes mandan model/model_name)
    model_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("model_name", "model"),
        description="Opcional. Para integraciones futuras; actualmente se infiere del run/champion.",
    )

    model_config = ConfigDict(populate_by_name=True)


class PredictResolvedResponse(BaseModel):
    """Respuesta de P2.2: resolución/validación del predictor bundle (sin inferencia)."""

    resolved_run_id: str
    resolved_from: str = Field(description="run_id|champion")
    run_dir: str = Field(description="Ruta lógica/absoluta del run_dir (según configuración).")

    predictor: Dict[str, Any] = Field(description="Contenido de predictor.json")
    preprocess: Dict[str, Any] = Field(description="Contenido de preprocess.json (puede ser vacío).")

    note: str = Field(description="Nota informativa del estado del endpoint.")



class ModelInfoResponse(BaseModel):
    """Respuesta de P2.2: metadata del modelo/predictor bundle (sin inferencia)."""

    resolved_run_id: str
    resolved_from: str = Field(description="run_id|champion")
    run_dir: str = Field(description="Ruta lógica/absoluta del run_dir (según configuración).")

    predictor: Dict[str, Any] = Field(description="Contenido de predictor.json")
    preprocess: Dict[str, Any] = Field(description="Contenido de preprocess.json (puede ser vacío).")

    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contenido de metrics.json (si existe).",
    )

    note: str = Field(description="Nota informativa del estado del endpoint.")


class HealthResponse(BaseModel):
    """Health check de predicciones."""

    status: str = "ok"
    artifacts_dir: str
    schema_version: int = 1
