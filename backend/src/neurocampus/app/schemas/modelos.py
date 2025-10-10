# backend/src/neurocampus/app/schemas/modelos.py
# Día 4 (B): ampliar hparams, tipar history. Mantiene compat. con lo que A dejó.

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class EntrenarRequest(BaseModel):
    modelo: str = Field(pattern="^(rbm_general|rbm_restringida)$")
    data_ref: Optional[str] = Field(
        default=None,
        description="Referencia al dataset (p.ej. localfs://datasets/ultimo.parquet)."
    )
    epochs: int = Field(default=5, ge=1, le=500)
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
        }
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
