# backend/src/neurocampus/prediction/templates/plantilla_prediccion.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time
import uuid
import numpy as np


def _to_jsonable(x: Any) -> Any:
    """
    Convierte tipos no serializables (numpy/torch/pandas) a tipos JSON-safe.
    - np.int64/np.float32 -> int/float
    - np.ndarray -> list
    - torch.Tensor -> list/float/int
    - np.str_ -> str
    """
    # Evita imports duros si no están instalados
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    # None / primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # dict
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # numpy scalars / arrays
    if np is not None:
        if isinstance(x, (getattr(np, "integer", ()), getattr(np, "floating", ()))):
            return x.item()
        if isinstance(x, getattr(np, "ndarray", ())):
            return x.tolist()
        if isinstance(x, getattr(np, "str_", ())):
            return str(x)

    # torch tensors
    if torch is not None:
        if isinstance(x, getattr(torch, "Tensor", ())):
            x_det = x.detach().cpu()
            if x_det.numel() == 1:
                return x_det.item()
            return x_det.tolist()

    # objetos con .item() (pandas scalar, numpy scalar ya cubierto, etc.)
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass

    # fallback: intenta string
    return str(x)


from neurocampus.observability.eventos_prediccion import (
    emit_requested, emit_completed, emit_failed
)  # Eventos prediction.* ya definidos en Día 6 (A)  # noqa: F401

# Nota: el middleware de correlación ya inyecta request.state.correlation_id
# y se refleja como cabecera X-Correlation-Id; aquí solo lo propagamos si llega. 
# (Ver Día 6 A)  # noqa: E501

class PlantillaPrediccion:
    def __init__(self, artifacts_loader, vectorizer=None, infer_fn=None, postprocess=None):
        """
        artifacts_loader: callable(job_id|None) -> dict/obj con handles/paths a modelos, etc.
        vectorizer: callable(texto:str, califs:dict) -> X para inferencia (opcional; puede venir None)
        infer_fn: callable(artifacts, X) -> raw scores (opcional; puede venir None)
        postprocess: callable(raw) -> (label_top, scores, sentiment, confidence) (opcional; puede venir None)
        """
        self._artifacts_loader = artifacts_loader
        self._vectorizer = vectorizer
        self._infer_fn = infer_fn
        self._postprocess = postprocess

    # -------------------------
    # Helpers robustos (fallback)
    # -------------------------
    def _extract_model(self, artifacts):
        """Intenta sacar un 'modelo' desde artifacts (dict u objeto)."""
        if artifacts is None:
            return None

        # Si ya es un modelo (tiene predict_*), devuélvelo
        if hasattr(artifacts, "predict_proba_df") or hasattr(artifacts, "predict_df") or hasattr(artifacts, "predict_proba") or hasattr(artifacts, "predict"):
            return artifacts

        if isinstance(artifacts, dict):
            for k in ("model", "strategy", "estrategia", "predictor"):
                v = artifacts.get(k)
                if v is not None and (
                    hasattr(v, "predict_proba_df") or hasattr(v, "predict_df") or hasattr(v, "predict_proba") or hasattr(v, "predict")
                ):
                    return v

        # también podría venir como atributo
        for k in ("model", "strategy", "estrategia", "predictor"):
            if hasattr(artifacts, k):
                v = getattr(artifacts, k)
                if v is not None and (
                    hasattr(v, "predict_proba_df") or hasattr(v, "predict_df") or hasattr(v, "predict_proba") or hasattr(v, "predict")
                ):
                    return v

        return None

    def _fallback_vectorize_online(self, comentario: str, calificaciones: dict):
        """
        Fallback simple: arma un DataFrame 1xN con calificaciones.

        Normalización importante:
        - Soporta llaves del request: pregunta_<n> y calif_<n>.
        - Siempre crea columnas canónicas calif_<n>, porque los modelos RBM
        suelen entrenarse con esas columnas.
        """
        import pandas as pd
        import re

        calificaciones = calificaciones or {}
        row: dict[str, Any] = {}

        for k, v in calificaciones.items():
            try:
                fv = float(v)
            except Exception:
                continue

            key = str(k)
            row[key] = fv

            m = re.match(r"^(pregunta|calif)_(\d+)$", key)
            if m:
                row[f"calif_{m.group(2)}"] = fv

            m2 = re.match(r"^calif_pregunta_(\d+)$", key)
            if m2:
                row[f"calif_{m2.group(1)}"] = fv

        if comentario is not None:
            row["comentario"] = str(comentario)

        return pd.DataFrame([row])


    def _fallback_infer(self, artifacts: Any, X, family: str = "sentiment_desempeno") -> Dict[str, Any]:
        """
        Fallback infer:
        - Extrae `model` desde artifacts (dict o wrapper)
        - Ejecuta predict_proba sobre DataFrame
        - Devuelve un dict compatible con `posprocesado.format_output`
        """
        if isinstance(artifacts, dict) and artifacts.get("error"):
            raise RuntimeError(f"Error cargando artifacts (family={family}): {artifacts['error']}")

        model = None
        if isinstance(artifacts, dict):
            model = artifacts.get("model")
        if model is None:
            model = self._extract_model(artifacts)

        if model is None:
            raise RuntimeError("No hay 'model' en artifacts (champion no cargó).")

        labels = ["neg", "neu", "pos"]
        if isinstance(artifacts, dict) and artifacts.get("labels"):
            try:
                labels = list(artifacts.get("labels"))
            except Exception:
                labels = ["neg", "neu", "pos"]

        p = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
        elif hasattr(model, "predict_proba_df"):
            p = model.predict_proba_df(X)
        else:
            raise RuntimeError("El model no expone predict_proba/predict_proba_df")

        # normalizar a vector 1D
        try:
            if hasattr(p, "ndim") and getattr(p, "ndim") == 2:
                p = p[0]
            p = list(p)
        except Exception:
            p = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        scores_any = {labels[i]: float(p[i]) for i in range(min(len(labels), len(p)))}

        scores = {
            "neg": float(scores_any.get("neg", 0.0)),
            "neu": float(scores_any.get("neu", 0.0)),
            "pos": float(scores_any.get("pos", 0.0)),
        }
        s = sum(scores.values()) or 1.0
        scores = {k: v / s for k, v in scores.items()}

        label_top = max(scores, key=scores.get)
        confidence = float(scores.get(label_top, 0.0))

        return {
            "labels": ["neg", "neu", "pos"],
            "proba": [scores["neg"], scores["neu"], scores["pos"]],
            "scores": scores,
            # Claves esperadas por format_output
            "sentiment_scores": scores,
            "materia_scores": {},
            "label_top": label_top,
            "confidence": confidence,
        }


    def _fallback_postprocess(self, raw):
        """
        Convierte raw -> (label_top, scores, sentiment, confidence)
        """
        labels = raw.get("labels") or ["neg", "neu", "pos"]
        proba = raw.get("proba")

        if proba is None:
            raise RuntimeError("raw no contiene 'proba' (fallback_postprocess).")

        sentiment = {labels[i]: float(proba[i]) for i in range(min(len(labels), len(proba)))}
        label_top = max(sentiment, key=sentiment.get)
        confidence = float(sentiment[label_top])
        scores = sentiment  # por ahora scores = distribución (sirve para smoke test)
        return label_top, scores, sentiment, confidence

    # -------------------------
    # API methods
    # -------------------------
    def predict_online(self, payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
        cid = correlation_id or f"cid-{uuid.uuid4()}"
        started = time.time()
        stage = "io"

        try:
            job_id = payload.get("job_id")

            # Define family
            family = payload.get("family") or "sentiment_desempeno"

            # Input del request (aquí estaba el bug: inp no existía)
            inp = payload.get("input") or {}
            if not isinstance(inp, dict):
                inp = {}

            emit_requested(cid, family=family, mode="online", n_items=1)

            # 1) load artifacts
            stage = "io"
            artifacts = self._artifacts_loader(job_id, family)

            # 2) vectorize
            stage = "vectorize"
            comentario = inp.get("comentario", "") or ""
            calificaciones = inp.get("calificaciones", {}) or {}
            if not isinstance(calificaciones, dict):
                calificaciones = {}

            if callable(self._vectorizer):
                X = self._vectorizer(comentario, calificaciones)
            else:
                X = self._fallback_vectorize_online(comentario, calificaciones)

            # 3) infer
            stage = "predict"
            if callable(self._infer_fn):
                # Algunos infer_fn aceptan (artifacts, X) y otros (artifacts, X, family)
                try:
                    import inspect
                    n_params = len(inspect.signature(self._infer_fn).parameters)
                    if n_params >= 3:
                        raw = self._infer_fn(artifacts, X, family)
                    else:
                        raw = self._infer_fn(artifacts, X)
                except Exception:
                    # fallback conservador
                    raw = self._infer_fn(artifacts, X)
            else:
                raw = self._fallback_infer(artifacts, X, family)

            # 4) postprocess
            stage = "postprocess"
            if callable(self._postprocess):
                label_top, scores, sentiment, confidence = self._postprocess(raw)
            else:
                label_top, scores, sentiment, confidence = self._fallback_postprocess(raw)

            lat_ms = int((time.time() - started) * 1000)
            emit_completed(
                cid,
                latencia_ms=lat_ms,
                n_items=1,
                distribucion_labels={label_top: 1},
                distribucion_sentiment=sentiment,
            )

            return {
                "label_top": label_top,
                "scores": scores,
                "sentiment": sentiment,
                "confidence": confidence,
                "latency_ms": lat_ms,
                "correlation_id": cid,
            }

        except Exception as e:
            # stage debe ser SOLO: vectorize|predict|postprocess|io
            if stage not in {"io", "vectorize", "predict", "postprocess"}:
                stage = "io"
            emit_failed(cid, error=str(e), stage=stage)
            raise


    def predict_batch(self, batch_items: List[Dict[str, Any]], correlation_id: str | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        cid = correlation_id or f"cid-{uuid.uuid4()}"
        started = time.time()
        stage = "io"

        try:
            emit_requested(cid, family="sentiment_desempeno", mode="batch", n_items=len(batch_items))

            # artifacts una sola vez
            artifacts = self._artifacts_loader(None)

            results = []
            for row in batch_items:
                # vectorize
                stage = "vectorize"
                if callable(self._vectorizer):
                    X = self._vectorizer(row.get("comentario", ""), row.get("calificaciones", {}))
                else:
                    X = self._fallback_vectorize_online(row.get("comentario", ""), row.get("calificaciones", {}))

                # infer
                stage = "predict"
                if callable(self._infer_fn):
                    raw = self._infer_fn(artifacts, X)
                else:
                    raw = self._fallback_infer(artifacts, X)

                # postprocess
                stage = "postprocess"
                if callable(self._postprocess):
                    label_top, scores, sentiment, confidence = self._postprocess(raw)
                else:
                    label_top, scores, sentiment, confidence = self._fallback_postprocess(raw)

                results.append(
                    {
                        "id": row.get("id"),
                        "label_top": label_top,
                        "confidence": confidence,
                        "scores": scores,
                        "sentiment": sentiment,
                    }
                )

            artifact_ref = f"localfs://predictions/batch/{uuid.uuid4()}.parquet"
            lat_ms = int((time.time() - started) * 1000)

            emit_completed(cid, latencia_ms=lat_ms, n_items=len(batch_items), distribucion_labels={}, distribucion_sentiment={})

            summary = {"rows": len(batch_items), "ok": len(results), "errors": 0, "engine": "pandas"}
            sample = results[:2]

            return (
                {
                    "batch_id": str(uuid.uuid4()),
                    "summary": summary,
                    "sample": sample,
                    "artifact": artifact_ref,
                    "correlation_id": cid,
                },
                results,
            )

        except Exception as e:
            emit_failed(cid, error=str(e), stage=stage)
            raise

    run_online = predict_online
    run_batch  = predict_batch
