# backend/src/neurocampus/prediction/facades/prediccion_facade.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from pathlib import Path

from neurocampus.prediction.templates.plantilla_prediccion import PlantillaPrediccion
from neurocampus.prediction.chain.posprocesado import format_output
from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral
from neurocampus.models.strategies.modelo_rbm_restringida import RBMRestringida


# -----------------------------------------------------------------------------
# Configuración de “campeón” (champion)
# -----------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
CHAMPIONS_ROOT = ARTIFACTS_DIR / "champions"

# Compat/fallback (soporte de tu código previo)
LEGACY_CHAMPION_JSON = Path("artifacts/champions/sentiment_desempeno/current.json")


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------
def _find_latest_champion_json(dataset_id: str | None = None) -> Path | None:
    # Si me dan dataset_id explícito: uso ese champion.json
    if dataset_id:
        p = CHAMPIONS_ROOT / dataset_id / "champion.json"
        return p if p.exists() else None

    # Si NO: tomo el champion.json más reciente por mtime
    candidates = list(CHAMPIONS_ROOT.glob("*/champion.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_model_from_champion_json(champion_json: Path):
    champ = json.loads(champion_json.read_text(encoding="utf-8"))

    dataset_id = champ.get("dataset_id") or champ.get("metrics", {}).get("dataset_id") or champion_json.parent.name
    model_name = champ.get("model_name") or champ.get("metrics", {}).get("model_name")
    model_path = champ.get("path") or champ.get("metrics", {}).get("path")

    if not model_name:
        raise ValueError(f"champion.json sin model_name: {champion_json}")

    # 1) path absoluto/guardado por el promote
    model_dir = Path(model_path) if model_path else (CHAMPIONS_ROOT / dataset_id / model_name)

    # 2) fallback por si el path guardado es raro
    if not model_dir.exists():
        model_dir = CHAMPIONS_ROOT / dataset_id / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"No existe model_dir={model_dir} (desde {champion_json})")

    if model_name == "rbm_general":
        model = RBMGeneral.load(str(model_dir))
    elif model_name == "rbm_restringida":
        model = RBMRestringida.load(str(model_dir))
    else:
        raise ValueError(f"model_name no soportado en predicción: {model_name}")

    return dataset_id, model_name, model_dir, model



def _load_artifacts_from_job(
    job_id: str | None,
    family: str = "sentiment_desempeno",
    dataset_id: str | None = None) -> dict:

    cj = _find_latest_champion_json(dataset_id=dataset_id)

    if cj is None:
        return {
            "job_id": job_id,
            "family": family,
            "error": f"No hay champion.json en {CHAMPIONS_ROOT}",
        }

    try:
        ds, mn, model_dir, model = _load_model_from_champion_json(cj)
        return {
            "job_id": job_id,
            "family": family,
            "dataset_id": ds,
            "model_name": mn,
            "champion_json": str(cj),
            "model_dir": str(model_dir),
            "model": model,
            "labels": ["neg", "neu", "pos"],
            "error": None,
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "family": family,
            "error": str(e),
        }


def _row_to_df(row: Dict[str, Any], feat_cols: List[str]) -> "pd.DataFrame":
    """
    Convierte un payload de entrada en un DataFrame con EXACTAMENTE feat_cols.
    - Si el payload viene con {"calificaciones": {...}, "comentario": "..."} se expanden las calificaciones.
    - Cualquier columna faltante se rellena con 0.0.
    """
    import pandas as pd

    flat: Dict[str, Any] = {}

    # 1) Expandir bloque calificaciones si existe
    calif = row.get("calificaciones")
    if isinstance(calif, dict):
        for k, v in calif.items():
            # normalizamos: admite 'pregunta_1' o 'calif_1' como keys
            if k.startswith("pregunta_"):
                flat[k.replace("pregunta_", "calif_")] = v
            else:
                flat[k] = v

    # 2) Copiar las llaves tope si vienen sueltas (p_neg, p_neu, p_pos, etc.)
    for k, v in row.items():
        if k == "calificaciones":
            continue
        # si ya están en feat_cols, se copian
        if k in feat_cols:
            flat[k] = v

    # 3) Construir DF con el orden exacto de feat_cols (rellenar faltantes)
    data = {}
    for c in feat_cols:
        data[c] = flat.get(c, 0.0)  # tolerante: faltante -> 0.0
    df = pd.DataFrame([data])
    return df


def _infer_real(model: RBMGeneral, payload: Dict[str, Any], feat_cols: List[str]) -> Dict[str, Any]:
    """
    Inferencia para un único item. Devuelve dict con scores y label_top.
    """
    import numpy as np

    df = _row_to_df(payload, feat_cols)
    proba = model.predict_proba_df(df)  # np.ndarray [N, 3] en orden [neg, neu, pos]
    if proba.ndim == 2 and proba.shape[0] == 1:
        p = proba[0]
    else:
        p = proba[0] if len(proba) else np.array([1/3, 1/3, 1/3], dtype=float)

    labels = ["neg", "neu", "pos"]
    top_idx = int(p.argmax())
    return {
        "scores": {"neg": float(p[0]), "neu": float(p[1]), "pos": float(p[2])},
        "label_top": labels[top_idx],
        "confidence": float(p[top_idx]),
    }


# -----------------------------------------------------------------------------
# Plantilla de predicción
#   - artifacts_loader -> carga RBMGeneral y feat_cols del campeón
#   - vectorizer       -> None (el modelo ya vectoriza internamente)
#   - infer_fn         -> usamos inferencia real con feat_cols del modelo
#   - postprocess      -> mantiene tu cadena de posprocesado
# -----------------------------------------------------------------------------

def _artifacts_loader_wrapper(job_id: str | None, family: str = "sentiment_desempeno"):
    return _load_artifacts_from_job(job_id, family=family)


def _infer_wrapper(artifacts: dict, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adaptador de inferencia que usa el modelo real.
    PlantillaPrediccion le pasa `artifacts` (de loader) y un `row`.
    """
    model: Optional[RBMGeneral] = artifacts.get("model")
    feat_cols: Optional[List[str]] = artifacts.get("feat_cols")
    if artifacts.get("error"):
        return {"error": artifacts["error"], "label_top": "neu", "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0}}
    if model is None or not feat_cols:
        return {"error": "Modelo o feat_cols no disponibles.", "label_top": "neu", "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0}}
    return _infer_real(model, row, feat_cols)


_TEMPLATE = PlantillaPrediccion(
    artifacts_loader=_artifacts_loader_wrapper,
    vectorizer=None,
    infer_fn=None,              # <-- CLAVE
    postprocess=format_output,
)



# -----------------------------------------------------------------------------
# API pública
# -----------------------------------------------------------------------------
def predict_online(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    # Compat: algunas plantillas exponen predict_online, otras run_online
    fn = getattr(_TEMPLATE, "predict_online", None) or getattr(_TEMPLATE, "run_online", None)
    if fn is None:
        raise AttributeError(
            "PlantillaPrediccion no expone predict_online ni run_online. "
            "Revisa backend/src/neurocampus/prediction/templates/plantilla_prediccion.py"
        )
    return fn(payload, correlation_id=correlation_id)


def predict_batch(items: List[Dict[str, Any]], correlation_id: str | None = None):
    fn = getattr(_TEMPLATE, "predict_batch", None) or getattr(_TEMPLATE, "run_batch", None)
    if fn is None:
        raise AttributeError(
            "PlantillaPrediccion no expone predict_batch ni run_batch. "
            "Revisa backend/src/neurocampus/prediction/templates/plantilla_prediccion.py"
        )
    return fn(items, correlation_id=correlation_id)
