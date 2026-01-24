# backend/src/neurocampus/app/schemas/modelos.py
# Día 4 (B): ampliar hparams, tipar history. Mantiene compat. con lo que A dejó.
# Día 5 (B): agregar campos de metodología de selección de datos.

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class EntrenarRequest(BaseModel):
    modelo: str = Field(
        pattern="^(rbm_general|rbm_restringida)$",
        description="Tipo de modelo a entrenar."
    )
    data_ref: Optional[str] = Field(
        default=None,
        description="Referencia al dataset (p.ej. localfs://datasets/ultimo.parquet). "
                    "Si no se provee, el backend intentará usar 'historico/unificado.parquet' si existe."
    )
    epochs: int = Field(
        default=5, ge=1, le=500,
        description="Número de épocas de entrenamiento (1..500)."
    )
    # hparams enriquecidos (respetar nombres pactados Día 4 A)
    hparams: Dict[str, Optional[float | int]] = Field(
        default={
            "n_visible": None,  # si es None se infiere del dataset
            "n_hidden": 32,
            "lr": 0.01,
            "batch_size": 64,
            "cd_k": 1,
            "momentum": 0.5,
            "weight_decay": 0.0,
            "seed": 42
        },
        description="Hiperparámetros del entrenamiento."
    )

    # ------------------------------
    # Día 5 (B): Metodología de datos
    # ------------------------------
    metodologia: Optional[Literal["periodo_actual", "acumulado", "ventana"]] = Field(
        default="periodo_actual",
        description=(
            "Estrategia de selección de datos: "
            "'periodo_actual' (solo el periodo actual), "
            "'acumulado' (<= periodo_actual), "
            "'ventana' (últimos N periodos)."
        )
    )
    periodo_actual: Optional[str] = Field(
        default=None,
        description=(
            "Periodo de referencia con formato 'YYYY-SEM' (p.ej. '2024-2'). "
            "Si se omite, el backend intentará inferir el máximo presente en el dataset."
        )
    )
    ventana_n: Optional[int] = Field(
        default=4, ge=1,
        description="Tamaño de la ventana para la metodología 'ventana' (entero >= 1)."
    )

class EpochItem(BaseModel):
    epoch: int
    loss: float
    recon_error: Optional[float] = None
    grad_norm: Optional[float] = None
    time_epoch_ms: Optional[float] = None

class EntrenarResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Entrenamiento lanzado"

class EstadoResponse(BaseModel):
    job_id: str
    status: str  # running|completed|failed|unknown
    metrics: Dict[str, float] = {}
    history: List[EpochItem] = []

class RunSummary(BaseModel):
    """
    Resumen ligero de un run para listados.

    Campos:
      - run_id: nombre del directorio dentro de artifacts/runs
      - model_name: nombre lógico del modelo asociado (rbm, rbm_general, etc.)
      - dataset_id: dataset asociado al run si fue registrado o inferible
      - created_at: ISO8601 (UTC) derivado de mtime del directorio
      - metrics: subset de métricas principales (accuracy, f1, etc.)
    """
    run_id: str
    model_name: str
    dataset_id: Optional[str] = None
    created_at: str
    metrics: Dict[str, Any] = {}


class RunDetails(BaseModel):
    """
    Detalle completo de un run.

    Incluye:
      - metrics: contenido completo de metrics.json
      - config: contenido de config.snapshot.yaml o config.yaml (si existe)
      - artifact_path: ruta absoluta o relativa al directorio del run (para depuración)
    """
    run_id: str
    dataset_id: Optional[str] = None
    metrics: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None


class ChampionInfo(BaseModel):
    """
    Información del modelo campeón (champion) para consumo por Predicciones/Dashboard.

    - model_name: nombre del modelo campeón
    - dataset_id: dataset asociado (si aplica)
    - metrics: métricas registradas del champion
    - path: ruta del directorio campeón en artifacts/champions
    """
    model_name: str
    dataset_id: Optional[str] = None
    metrics: Dict[str, Any]
    path: str