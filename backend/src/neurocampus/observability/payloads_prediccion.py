# backend/src/neurocampus/observability/payloads_prediccion.py
from __future__ import annotations
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class PredRequested(BaseModel):
    correlation_id: str = Field(..., description="UUID v4 de la solicitud")
    family: str = Field(..., description="Familia/ensamble de modelo")
    mode: Literal["online", "batch"]
    n_items: int = Field(..., ge=1)
    ts: Optional[datetime] = Field(default=None, description="UTC timestamp")

class PredCompleted(BaseModel):
    correlation_id: str
    latencia_ms: int = Field(..., ge=0)
    n_items: int = Field(..., ge=1)
    distribucion_labels: Optional[Dict[str, int]] = None
    distribucion_sentiment: Optional[Dict[str, float]] = None
    ts: Optional[datetime] = None

class PredFailed(BaseModel):
    correlation_id: str
    error_code: str
    error: str
    stage: Optional[Literal["vectorize", "predict", "postprocess", "io"]] = None
    ts: Optional[datetime] = None
