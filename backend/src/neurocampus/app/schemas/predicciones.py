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

    # P2.4: control explícito de inferencia (mantiene compatibilidad con P2.2)
    do_inference: bool = Field(
        default=False,
        description="Si true, ejecuta inferencia. Si false, solo resuelve/valida el bundle (P2.2).",
    )

    # Selección del feature_pack
    input_level: Optional[str] = Field(
        default=None,
        description="Nivel de entrada: row|pair. Si None, se usa predictor.json[input_level].",
    )
    limit: int = Field(default=50, ge=1, le=500, description="Máximo de filas a predecir (para respuestas pequeñas).")
    offset: int = Field(default=0, ge=0, description="Offset posicional dentro del feature_pack.")
    ids: Optional[list[int]] = Field(
        default=None,
        description="Índices posicionales a predecir (toma prioridad sobre offset/limit).",
    )
    return_proba: bool = Field(
        default=True,
        description="Si true, incluye probabilidades para clasificación (cuando el modelo lo soporte).",
    )

    persist: bool = Field(
        default=False,
        description="Si true, persiste predictions.parquet en artifacts/predictions/... y retorna predictions_uri.",
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

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class PredictResolvedResponse(BaseModel):
    """Respuesta de P2.2: resolución/validación del predictor bundle (sin inferencia)."""

    resolved_run_id: str
    resolved_from: str = Field(description="run_id|champion")
    run_dir: str = Field(description="Ruta lógica/absoluta del run_dir (según configuración).")

    predictor: Dict[str, Any] = Field(description="Contenido de predictor.json")
    preprocess: Dict[str, Any] = Field(description="Contenido de preprocess.json (puede ser vacío).")

    # P2.4: salida de inferencia (opcional). Si no hay inferencia, estos campos serán None.
    predictions: Optional[list[Dict[str, Any]]] = Field(
        default=None,
        description="Predicciones normalizadas (row/pair).",
    )

    predictions_uri: Optional[str] = Field(
        default=None,
        description="Ruta lógica donde se persistieron las predicciones (cuando persist=true).",
    )
    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadatos del modelo/run usados para inferir (subset estable para UI).",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("schema", "output_schema"),
        serialization_alias="schema",
        description="Esquema de salida (campos/probabilidades).",
    )
    warnings: Optional[list[str]] = Field(
        default=None,
        description="Advertencias (fallbacks, supuestos, etc.).",
    )

    note: str = Field(description="Nota informativa del estado del endpoint.")

    model_config = ConfigDict(protected_namespaces=())



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
