# backend/src/neurocampus/prediction/facades/prediccion_facade.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from pathlib import Path

from neurocampus.prediction.templates.plantilla_prediccion import PlantillaPrediccion
from neurocampus.prediction.chain.posprocesado import format_output
from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral

# -----------------------------------------------------------------------------
# Configuración de “campeón” (champion)
# -----------------------------------------------------------------------------
# Familia por defecto (debe coincidir con la usada por cmd_autoretrain.py)
DEFAULT_FAMILY = "with_text"

# Directorio donde cmd_autoretrain promueve el campeón:
CHAMPIONS_DIR = Path("artifacts") / "champions" / DEFAULT_FAMILY

# Compat/fallback (soporte de tu código previo)
LEGACY_CHAMPION_JSON = Path("artifacts/champions/sentiment_desempeno/current.json")


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _find_champion_dir() -> Optional[Path]:
    """
    Determina el directorio del campeón actual.
    1) Usa artifacts/champions/<family>/latest.txt si existe.
    2) Fallback: artifacts/champions/<family>/best_meta.json -> 'job_dir'
    3) Fallback legacy: artifacts/champions/sentiment_desempeno/current.json -> 'job_id'
       (resuelve a artifacts/jobs/<job_id>)
    """
    # (1) latest.txt
    latest_txt = CHAMPIONS_DIR / "latest.txt"
    if latest_txt.exists():
        p = Path(latest_txt.read_text(encoding="utf-8").strip())
        if p.exists():
            return p

    # (2) best_meta.json
    best_meta = _read_json(CHAMPIONS_DIR / "best_meta.json")
    if best_meta and "job_dir" in best_meta:
        p = Path(best_meta["job_dir"])
        if p.exists():
            return p

    # (3) legacy current.json
    legacy = _read_json(LEGACY_CHAMPION_JSON)
    if legacy and "job_id" in legacy:
        p = Path("artifacts") / "jobs" / str(legacy["job_id"])
        if p.exists():
            return p

    return None


def _load_artifacts_from_job(job_id: str | None) -> dict:
    """
    Carga artefactos de un job específico o, si no se da job_id, del campeón actual.
    Devuelve:
      {
        "job_id": str | None,
        "model": RBMGeneral | None,
        "feat_cols": List[str] | None,
        "error": str | None
      }
    """
    try:
        # Determinar el directorio de artefactos
        if job_id:
            job_dir = Path("artifacts") / "jobs" / job_id
        else:
            job_dir = _find_champion_dir()

        if not job_dir or not job_dir.exists():
            return {"job_id": job_id, "model": None, "feat_cols": None, "error": "Champion/job no encontrado."}

        # Cargar modelo
        model = RBMGeneral.load(str(job_dir))

        # Obtener feat_cols desde meta.json o job_meta.json (ambas son soportadas)
        feat_cols: Optional[List[str]] = None
        meta = _read_json(job_dir / "meta.json") or _read_json(job_dir / "job_meta.json")
        if meta:
            feat_cols = meta.get("feat_cols") or meta.get("feature_cols") or meta.get("feature_columns")
        if not feat_cols and getattr(model, "feat_cols_", None):
            feat_cols = list(model.feat_cols_)

        # Seguridad mínima
        if not feat_cols or len(feat_cols) == 0:
            # Si no hay feat_cols persistidas, hacemos fallback a las del modelo
            feat_cols = list(getattr(model, "feat_cols_", []) or [])

        return {"job_id": job_dir.name, "model": model, "feat_cols": feat_cols, "error": None}
    except Exception as ex:
        return {"job_id": job_id, "model": None, "feat_cols": None, "error": f"Error cargando artefactos: {ex}"}


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
def _artifacts_loader_wrapper(job_id: str | None) -> dict:
    """
    Adaptador a la interfaz esperada por PlantillaPrediccion:
    devuelve {"job_id", "model", "feat_cols"} y, si hubo problema, "error".
    """
    return _load_artifacts_from_job(job_id)


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
    vectorizer=None,          # El modelo maneja su propio vectorizador/normalizador
    infer_fn=_infer_wrapper,
    postprocess=format_output
)


# -----------------------------------------------------------------------------
# API pública
# -----------------------------------------------------------------------------
def predict_online(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    """
    Espera un payload con, por ejemplo:
    {
      "family": "with_text",        # opcional (por ahora se ignora si difiere de DEFAULT_FAMILY)
      "job_id": null,               # opcional: si lo das, carga ese job; si no, carga campeón
      "input": {
         "calificaciones": {"calif_1": 4.2, "calif_2": 4.1, ...},
         "p_neg": 0.10, "p_neu": 0.20, "p_pos": 0.70,   # opcional
         "comentario": "texto..."                        # opcional; no se vectoriza aquí
      }
    }
    """
    return _TEMPLATE.predict_online(payload, correlation_id=correlation_id)


def predict_batch(items: List[Dict[str, Any]], correlation_id: str | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    items: lista de payloads (cada uno con "input": {...}) como en predict_online.
    """
    return _TEMPLATE.predict_batch(items, correlation_id=correlation_id)
