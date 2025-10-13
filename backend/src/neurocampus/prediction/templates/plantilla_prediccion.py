# backend/src/neurocampus/prediction/templates/plantilla_prediccion.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time
import uuid

from neurocampus.observability.eventos_prediccion import (
    emit_requested, emit_completed, emit_failed
)  # Eventos prediction.* ya definidos en Día 6 (A)  # noqa: F401

# Nota: el middleware de correlación ya inyecta request.state.correlation_id
# y se refleja como cabecera X-Correlation-Id; aquí solo lo propagamos si llega. 
# (Ver Día 6 A)  # noqa: E501

class PlantillaPrediccion:
    """
    Orquesta el pipeline de predicción:
      - Carga artefactos publicados (campeón) o por job_id.
      - Vectoriza inputs.
      - Ejecuta inferencia (scores probabilísticos).
      - Aplica post-procesado (calibración/umbrales/formato).
      - Emite eventos prediction.* con telemetría.
    """

    def __init__(self, artifacts_loader, vectorizer, infer_fn, postprocess):
        """
        artifacts_loader: callable(job_id|None) -> dict con handles/paths a modelos, vectorizador, etc.
        vectorizer: callable(texto:str, califs:dict) -> features (X) para inferencia.
        infer_fn: callable(artifacts, X) -> dict con probabilidades por clase/etiquetas adicionales.
        postprocess: callable(scores:dict) -> (label_top:str, scores:dict, sentiment:dict, confidence:float)
        """
        self._artifacts_loader = artifacts_loader
        self._vectorizer = vectorizer
        self._infer_fn = infer_fn
        self._postprocess = postprocess

    def predict_online(self, payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
        """
        Cumple contrato de POST /prediccion/online (v0.6.0).
        Body esperado:
          {
            "job_id": "uuid-opcional",
            "family": "sentiment_desempeno",
            "input": { "calificaciones": {...}, "comentario": "..." }
          }
        """
        cid = correlation_id or f"cid-{uuid.uuid4()}"
        started = time.time()
        try:
            emit_requested(cid, family=payload.get("family","sentiment_desempeno"), mode="online", n_items=1)
            job_id = payload.get("job_id")
            inp = payload["input"]
            X = self._vectorizer(inp.get("comentario",""), inp.get("calificaciones", {}))
            artifacts = self._artifacts_loader(job_id)
            raw = self._infer_fn(artifacts, X)
            label_top, scores, sentiment, confidence = self._postprocess(raw)
            lat_ms = int((time.time()-started)*1000)
            emit_completed(cid, latencia_ms=lat_ms, n_items=1,
                           distribucion_labels={label_top:1},
                           distribucion_sentiment=sentiment)
            return {
                "label_top": label_top,
                "scores": scores,           # probabilidades por materia (0..1)
                "sentiment": sentiment,     # {"pos":..,"neu":..,"neg":..}
                "confidence": confidence,   # típicamente max(scores.values())
                "latency_ms": lat_ms,
                "correlation_id": cid
            }
        except Exception as e:
            emit_failed(cid, error=str(e), stage="predict_online")
            raise

    def predict_batch(self, batch_items: List[Dict[str, Any]], correlation_id: str | None = None) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
        """
        Cumple contrato de POST /prediccion/batch (v0.6.0):
          - Procesa N filas y devuelve summary + sample + artifact path.
        """
        cid = correlation_id or f"cid-{uuid.uuid4()}"
        started = time.time()
        try:
            emit_requested(cid, family="sentiment_desempeno", mode="batch", n_items=len(batch_items))
            artifacts = self._artifacts_loader(None)
            results = []
            for row in batch_items:
                X = self._vectorizer(row.get("comentario",""), row.get("calificaciones", {}))
                raw = self._infer_fn(artifacts, X)
                label_top, scores, sentiment, confidence = self._postprocess(raw)
                results.append({
                    "id": row.get("id"),
                    "label_top": label_top,
                    "confidence": confidence,
                    "scores": scores,
                    "sentiment": sentiment
                })
            # TODO: persistir parquet y devolver artifact ref
            artifact_ref = f"localfs://predictions/batch/{uuid.uuid4()}.parquet"
            lat_ms = int((time.time()-started)*1000)
            emit_completed(cid, latencia_ms=lat_ms, n_items=len(batch_items),
                           distribucion_labels={}, distribucion_sentiment={})
            summary = {"rows": len(batch_items), "ok": len(results), "errors": 0, "engine": "pandas"}
            sample = results[:2]
            return {
                "batch_id": str(uuid.uuid4()),
                "summary": summary,
                "sample": sample,
                "artifact": artifact_ref,
                "correlation_id": cid
            }, results
        except Exception as e:
            emit_failed(cid, error=str(e), stage="predict_batch")
            raise
