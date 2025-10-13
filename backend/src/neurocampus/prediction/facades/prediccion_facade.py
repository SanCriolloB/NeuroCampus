# backend/src/neurocampus/prediction/facades/prediccion_facade.py
from typing import Dict, Any, List, Tuple
import json
import os

from neurocampus.prediction.templates.plantilla_prediccion import PlantillaPrediccion
from neurocampus.prediction.chain.posprocesado import format_output
from neurocampus.prediction.strategies.etiquetado import vectorize_simple, infer_stub

CHAMPION_PATH = "artifacts/champions/sentiment_desempeno/current.json"

def _load_artifacts_from_job(job_id: str | None) -> dict:
    """
    Carga artefactos del campeón o del job_id indicado.
    Reemplazar por lectura real de: vectorizer.pkl, rbm*.ckpt, head.pkl, etc.
    """
    # Campeón por defecto
    if job_id is None and os.path.exists(CHAMPION_PATH):
        with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        job_id = meta.get("job_id")
    return {"job_id": job_id, "vectorizer": None, "model": None}

_TEMPLATE = PlantillaPrediccion(
    artifacts_loader=_load_artifacts_from_job,
    vectorizer=vectorize_simple,
    infer_fn=infer_stub,
    postprocess=format_output
)

def predict_online(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    return _TEMPLATE.predict_online(payload, correlation_id=correlation_id)

def predict_batch(items: List[Dict[str,Any]], correlation_id: str | None = None) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    return _TEMPLATE.predict_batch(items, correlation_id=correlation_id)
