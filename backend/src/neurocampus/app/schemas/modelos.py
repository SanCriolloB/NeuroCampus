from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EntrenarRequest(BaseModel):
    modelo: str = Field(..., pattern="^(rbm_general|rbm_restringida)$")
    data_ref: str = "localfs://datasets/ultimo.parquet"   # referencia lógica
    epochs: int = 5
    hparams: Dict[str, Any] = {}

class EntrenarResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

class EstadoResponse(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, float] = {}