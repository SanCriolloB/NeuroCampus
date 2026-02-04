# backend/src/neurocampus/app/routers/prediccion.py
from fastapi import APIRouter, UploadFile, File, Request
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np

from neurocampus.app.schemas.prediccion import (
    PrediccionOnlineRequest, PrediccionOnlineResponse,
    PrediccionBatchResponse, PrediccionBatchItem
)
from neurocampus.prediction.facades.prediccion_facade import predict_online, predict_batch

router = APIRouter(tags=["prediccion"])

# --------------------------
# Reglas de decisión (inferencia)
# --------------------------
# Si el modelo "duda" entre NEG y NEU, favorecemos NEG cuando:
#   - p_neg >= neg_min, o
#   - p_neg - p_neu >= neg_neu_margin
# Si POS es muy claro, priorizamos POS.
_POS_MIN = 0.55
_NEG_MIN = 0.35
_NEG_NEU_MARGIN = 0.05

_IDX2LBL = {0: "neg", 1: "neu", 2: "pos"}

def _decide_with_rules_from_proba(proba: Union[List[float], np.ndarray],
                                  pos_min: float = _POS_MIN,
                                  neg_min: float = _NEG_MIN,
                                  neg_neu_margin: float = _NEG_NEU_MARGIN) -> str:
    """
    Aplica reglas costo-sensibles sobre un vector de proba [p_neg, p_neu, p_pos].
    Devuelve 'neg' | 'neu' | 'pos'. Si el vector no es válido, cae en argmax.
    """
    p = np.asarray(proba, dtype=float).reshape(-1)
    if p.size != 3 or not np.isfinite(p).all():
        # fallback seguro
        return _IDX2LBL[int(np.nanargmax(p))] if p.size == 3 else "neu"

    p_neg, p_neu, p_pos = p[0], p[1], p[2]

    # 1) POS claro
    if p_pos >= pos_min:
        return "pos"

    # 2) NEG razonable si no fue POS
    if (p_neg >= neg_min) or ((p_neg - p_neu) >= neg_neu_margin):
        return "neg"

    # 3) En otro caso, NEU
    return "neu"


@router.post("/online", response_model=PrediccionOnlineResponse)
async def prediccion_online(req: Request, body: PrediccionOnlineRequest):
    """
    Endpoint de predicción online.

    - Devuelve errores como JSON (no 500 text/plain genérico).
    - Hace JSON-safe la salida (numpy/torch) para evitar 500 por serialización.
    """
    import os
    import logging
    import traceback

    import numpy as np
    from fastapi import HTTPException
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    try:
        import torch  # opcional
    except Exception:
        torch = None

    log = logging.getLogger(__name__)
    debug = os.environ.get("NEUROCAMPUS_DEBUG", "0") == "1"

    # ---- aquí ejecutas tu pipeline actual ----
    try:
        # IMPORTANTE: respeta tu implementación actual.
        # Normalmente aquí ya existe algo como:
        #   tpl = PlantillaPrediccion(...)
        #   out = tpl.run_online(body)  / tpl.run(body) / tpl.predict(...)
        # Ajusta SOLO estas 2 líneas si tu template usa otro método.
        payload = body.model_dump(exclude_none=True) if hasattr(body, "model_dump") else body.dict(exclude_none=True)
        out = predict_online(payload, correlation_id=getattr(req.state, "correlation_id", None))
    
    except Exception as e:
        log.exception("prediccion_online failed (cid=%s)", getattr(req.state, "correlation_id", None))
        detail = {"detail": "prediction_failed", "error": str(e)}
        if debug:
            detail["traceback"] = traceback.format_exc()
        raise HTTPException(status_code=500, detail=detail)

    # ---- JSON-safe encoding (evita 500 por numpy/torch en la respuesta) ----
    custom = {
        np.integer: int,
        np.floating: float,
        np.ndarray: lambda a: a.tolist(),
    }
    if torch is not None:
        custom[torch.Tensor] = lambda t: t.detach().cpu().tolist()

    content = jsonable_encoder(out, custom_encoder=custom)
    return JSONResponse(content=content)



@router.post("/batch", response_model=PrediccionBatchResponse, status_code=201)
async def prediccion_batch(req: Request, file: UploadFile | None = File(default=None)):
    """
    Variante mínima: si llega `file`, lo leemos a filas {id, calificaciones, comentario}.
    Si no, podríamos aceptar JSON con data_ref (extensible).
    Nota: aquí no forzamos aún la regla costo-sensible porque el contrato del facade/respuesta
    puede variar según implementación. Si tu batch devuelve 'proba' por ítem y el schema lo permite,
    puedes replicar el mismo post-proceso que en /online.
    """
    cid = getattr(req.state, "correlation_id", None)
    items: List[Dict[str, Any]] = []
    if file:
        # Ejemplo CSV esperado con columnas: id, comentario, pregunta_1..pregunta_10
        df = pd.read_csv(file.file)
        for _, row in df.iterrows():
            califs = {c: float(row[c]) for c in df.columns if c.startswith("pregunta_")}
            items.append({
                "id": str(row.get("id", "")),
                "comentario": str(row.get("comentario", "")),
                "calificaciones": califs
            })

    # TODO (opcional): soportar JSON con data_ref y adaptación a adapters existentes
    summary, _rows = predict_batch(items, correlation_id=cid)

    # Si tu summary incluye proba por item y el schema lo soporta, puedes aplicar aquí la misma regla:
    # for it in summary.get("items", []):
    #     proba = it.get("proba")
    #     if isinstance(proba, (list, tuple, np.ndarray)) and len(proba) == 3:
    #         it["label"] = _decide_with_rules_from_proba(proba)
    #         it["decision_rule"] = {"pos_min": _POS_MIN, "neg_min": _NEG_MIN, "neg_neu_margin": _NEG_NEU_MARGIN}

    return summary
