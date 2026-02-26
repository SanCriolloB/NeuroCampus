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

from typing import Any, Dict, Optional, List


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




class PredictionsPreviewResponse(BaseModel):
    """Vista previa de predicciones persistidas."""

    predictions_uri: str = Field(description="Ruta lógica donde están persistidas las predicciones.")
    rows: list[Dict[str, Any]] = Field(description="Filas (preview) desde predictions.parquet.")
    columns: list[str] = Field(description="Columnas presentes en `rows`.")

    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("schema", "output_schema"),
        serialization_alias="schema",
        description="schema.json asociado (si existe).",
    )

    note: str = Field(description="Nota informativa del endpoint.")



class HealthResponse(BaseModel):
    """Health check de predicciones."""

    status: str = "ok"
    artifacts_dir: str
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Schemas para endpoints de la pestaña Predicciones (score_docente)
# ---------------------------------------------------------------------------


class DatasetInfoResponse(BaseModel):
    """Información de un dataset disponible para predicción.

    Expone metadatos del dataset (derivados de pair_matrix/pair_meta) junto al estado del champion.
    """

    dataset_id: str = Field(description="Identificador del dataset (ej. '2024-2').")
    n_pairs: int = Field(default=0, description="Número de pares docente–materia en pair_matrix.")
    n_docentes: int = Field(default=0, description="Número de docentes únicos.")
    n_materias: int = Field(default=0, description="Número de materias únicas.")
    has_champion: bool = Field(
        default=False,
        description="True si existe champion.json para score_docente en este dataset.",
    )
    created_at: Optional[str] = Field(default=None, description="Timestamp ISO de creación del dataset/pair_matrix.")


class TeacherInfoResponse(BaseModel):
    """Información de un docente disponible en el dataset."""

    teacher_key: str = Field(description="Clave normalizada del docente (cedula o nombre).")
    teacher_id: int = Field(description="Índice numérico asignado en teacher_index.json.")
    n_encuestas: int = Field(default=0, description="Total de encuestas del docente en el dataset.")


class MateriaInfoResponse(BaseModel):
    """Información de una materia disponible en el dataset."""

    materia_key: str = Field(description="Clave normalizada de la materia (código).")
    materia_id: int = Field(description="Índice numérico asignado en materia_index.json.")
    n_encuestas: int = Field(default=0, description="Total de encuestas de la materia en el dataset.")


class RadarPoint(BaseModel):
    """Un punto en el radar de indicadores (una dimensión)."""

    indicator: str = Field(description="Nombre del indicador (ej. 'Planificación').")
    actual: float = Field(description="Promedio histórico real (escala 0–5).")
    prediccion: float = Field(description="Valor proyectado proporcional al score predicho (escala 0–5).")


class ComparisonPoint(BaseModel):
    """Un punto del bar chart comparativo (una dimensión)."""

    dimension: str = Field(description="Nombre de la dimensión.")
    docente: float = Field(description="Promedio del par específico (escala 0–5).")
    cohorte: float = Field(description="Promedio de todos los pares de esa materia (escala 0–5).")


class TimelinePoint(BaseModel):
    """Un punto de la serie temporal (un período/semestre)."""

    semester: str = Field(description="Identificador del período (dataset_id, ej. '2023-1').")
    real: Optional[float] = Field(
        default=None,
        description="Score histórico real del par en ese período (mean_score_total_0_50).",
    )
    predicted: Optional[float] = Field(
        default=None,
        description="Score predicho por el champion (solo para el período seleccionado).",
    )


class EvidenceInfo(BaseModel):
    """Estadísticas de evidencia para un par docente–materia."""

    n_par: int = Field(description="Encuestas históricas de este par específico.")
    n_docente: int = Field(description="Encuestas totales del docente en el dataset.")
    n_materia: int = Field(description="Encuestas totales de la materia en el dataset.")


class HistoricalStats(BaseModel):
    """Estadísticas históricas del par en el dataset actual."""

    mean_score: float = Field(description="Score promedio histórico del par (0–50).")
    std_score: float = Field(description="Desviación estándar del score histórico.")


class IndividualPredictionRequest(BaseModel):
    """Request para predicción individual de un par docente–materia."""

    dataset_id: str = Field(description="Dataset sobre el que se predice.")
    teacher_key: str = Field(description="Clave normalizada del docente.")
    materia_key: str = Field(description="Clave normalizada de la materia.")


class IndividualPredictionResponse(BaseModel):
    """Respuesta completa de predicción individual.

    Contiene todos los datos necesarios para poblar los charts de la UI sin llamadas adicionales.
    """

    dataset_id: str
    teacher_key: str
    materia_key: str

    score_total_pred: float = Field(description="Score predicho por el champion (0–50).")
    risk: str = Field(description="Categoría de riesgo: 'low' | 'medium' | 'high'.")
    confidence: float = Field(description="Tasa de confiabilidad (0–1) basada en evidencia histórica.")
    cold_pair: bool = Field(description="True si el par nunca apareció en el dataset (sin historial).")

    evidence: EvidenceInfo
    historical: HistoricalStats

    radar: List[RadarPoint] = Field(description="Datos para el RadarChart (10 dimensiones).")
    comparison: List[ComparisonPoint] = Field(description="Datos para el BarChart comparativo.")
    timeline: List[TimelinePoint] = Field(description="Serie temporal de scores reales + predicción del período actual.")

    champion_run_id: str = Field(description="Run ID del champion usado para la predicción.")
    model_name: str = Field(description="Nombre del modelo champion (ej. 'rbm_restringida').")

    model_config = ConfigDict(protected_namespaces=())


class BatchRunRequest(BaseModel):
    """Request para lanzar un job de predicción por lote."""

    dataset_id: str = Field(description="Dataset a predecir (todos los pares de pair_matrix).")


class BatchJobResponse(BaseModel):
    """Estado de un job de predicción por lote."""

    job_id: str = Field(description="Identificador del job (UUID).")
    status: str = Field(description="Estado: 'queued' | 'running' | 'completed' | 'failed'.")
    progress: float = Field(default=0.0, description="Progreso entre 0.0 y 1.0.")

    pred_run_id: Optional[str] = Field(default=None, description="ID del run cuando completa.")
    dataset_id: str
    n_pairs: Optional[int] = Field(default=None, description="Pares procesados (cuando completa).")
    predictions_uri: Optional[str] = Field(
        default=None,
        description="Ruta lógica del parquet generado (cuando completa).",
    )
    champion_run_id: Optional[str] = Field(default=None, description="Run ID del champion usado.")
    error: Optional[str] = Field(default=None, description="Mensaje de error (si status='failed').")


class PredictionRunInfoResponse(BaseModel):
    """Metadatos de un run persistido de predicción (batch).

    Este schema está pensado para poblar el bloque **Historial** de la pestaña
    Predicciones, permitiendo reabrir el preview o descargar el parquet
    asociado a un run anterior.
    """

    pred_run_id: str = Field(description="Identificador del run (ej. 'pred_20260226_153012').")
    dataset_id: str = Field(description="Dataset asociado.")
    family: str = Field(description="Familia del modelo (en esta pestaña: score_docente).")

    created_at: Optional[str] = Field(default=None, description="Timestamp ISO (UTC) de creación.")
    n_pairs: int = Field(default=0, description="Número de pares predichos en el run.")

    champion_run_id: Optional[str] = Field(default=None, description="Run ID del champion usado.")
    model_name: Optional[str] = Field(default=None, description="Nombre lógico del modelo (si se registró).")

    predictions_uri: Optional[str] = Field(
        default=None,
        description="Ruta lógica a predictions.parquet para preview/descarga.",
    )

    model_config = ConfigDict(protected_namespaces=())

