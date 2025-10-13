# backend/src/neurocampus/app/schemas/prediccion.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class OnlineInput(BaseModel):
    calificaciones: Dict[str, float] = Field(default_factory=dict)
    comentario: str = ""

class PrediccionOnlineRequest(BaseModel):
    job_id: Optional[str] = None
    family: str = "sentiment_desempeno"
    input: OnlineInput

class PrediccionOnlineResponse(BaseModel):
    label_top: str
    scores: Dict[str, float]
    sentiment: Dict[str, float]
    confidence: float
    latency_ms: int
    correlation_id: str

class PrediccionBatchItem(BaseModel):
    id: Optional[str] = None
    calificaciones: Dict[str, float] = Field(default_factory=dict)
    comentario: str = ""

class PrediccionBatchResponse(BaseModel):
    batch_id: str
    summary: Dict[str, Any]
    sample: List[Dict[str, Any]]
    artifact: str
    correlation_id: str
