# backend/src/neurocampus/app/routers/prediccion.py
from fastapi import APIRouter, UploadFile, File, Request
from typing import List, Dict, Any
import pandas as pd  # usado si recibimos archivo; engine puede ser polars si se prefiere
from neurocampus.app.schemas.prediccion import (
    PrediccionOnlineRequest, PrediccionOnlineResponse,
    PrediccionBatchResponse, PrediccionBatchItem
)
from neurocampus.prediction.facades.prediccion_facade import predict_online, predict_batch

router = APIRouter(prefix="/prediccion", tags=["prediccion"])

@router.post("/online", response_model=PrediccionOnlineResponse)
async def prediccion_online(req: Request, body: PrediccionOnlineRequest):
    cid = getattr(req.state, "correlation_id", None)  # middleware ya lo inyecta
    return predict_online(body.model_dump(), correlation_id=cid)

@router.post("/batch", response_model=PrediccionBatchResponse, status_code=201)
async def prediccion_batch(req: Request, file: UploadFile | None = File(default=None)):
    """
    Variante mínima: si llega `file`, lo leemos a filas {id, calificaciones, comentario}.
    Si no, podríamos aceptar JSON con data_ref (extensible).
    """
    cid = getattr(req.state, "correlation_id", None)
    items: List[Dict[str,Any]] = []
    if file:
        # Ejemplo CSV esperado con columnas: id, comentario, pregunta_1..pregunta_10
        df = pd.read_csv(file.file)
        for _, row in df.iterrows():
            califs = {c: float(row[c]) for c in df.columns if c.startswith("pregunta_")}
            items.append({"id": str(row.get("id","")), "comentario": str(row.get("comentario","")), "calificaciones": califs})
    # TODO: soportar JSON con data_ref y adaptación a adapters existentes
    summary, _rows = predict_batch(items, correlation_id=cid)
    return summary
