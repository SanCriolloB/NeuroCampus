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
