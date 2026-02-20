"""neurocampus.utils.model_context

Helper reutilizable para completar metadata de modelos sin `null` críticos.

Contexto
--------
En P2.1 necesitamos que distintos contratos (API de Modelos, API de
Predicciones y `predictor.json`) expongan un *contexto* consistente.

En runs reales, el contexto suele estar en ``metrics.json`` bajo
``params.req`` (snapshot del request normalizado). Sin embargo, algunos
endpoints/bundles solo leen:

- campos top-level en metrics (``metrics.task_type``), o
- valores parciales (``predictor.json``)

lo que resulta en `null`/`"unknown"` para campos críticos.

Regla única de precedencia
-------------------------
Para resolver cada campo se aplica, *en este orden*:

1) ``metrics.params.req.<campo>``
2) ``metrics.<campo>`` (top-level)
3) ``predictor.json.<campo>`` (top-level) o ``predictor.json.extra.<campo>``
4) fallback seguro por ``family``

Este módulo NO escribe archivos; solo resuelve valores para que otros
módulos puedan persistir o responder sin `null`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_MISSING_STRINGS = {"", "none", "null", "unknown", "n/a"}


def _norm_str(v: Any) -> Optional[str]:
    """Normaliza un valor a `str` (si es razonable) o `None`."""

    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in _MISSING_STRINGS:
            return None
        return s
    # Evitar convertir dict/list a string accidentalmente
    if isinstance(v, (dict, list, tuple, set)):
        return None
    try:
        s2 = str(v).strip()
    except Exception:
        return None
    return None if s2.lower() in _MISSING_STRINGS else (s2 or None)


def extract_req(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Extrae ``metrics.params.req`` como dict.

    Args:
        metrics: payload de metrics.json (dict-like).

    Returns:
        Un dict (posiblemente vacío) con el snapshot del request.
    """

    params = metrics.get("params") if isinstance(metrics, Mapping) else None
    if not isinstance(params, Mapping):
        return {}
    req = params.get("req")
    return dict(req) if isinstance(req, Mapping) else {}


_FAMILY_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "sentiment_desempeno": {
        "task_type": "classification",
        "input_level": "row",
        "target_col": "y_sentimiento",
    },
    "score_docente": {
        "task_type": "regression",
        "input_level": "pair",
        "target_col": "target_score",
    },
}


def _norm_task_type(v: Any) -> Optional[str]:
    s = _norm_str(v)
    if not s:
        return None
    s2 = s.lower()
    if s2 in {"classification", "regression"}:
        return s2
    # aliases comunes
    if s2 in {"clf", "class", "clasificacion", "clasificación"}:
        return "classification"
    if s2 in {"reg", "regresion", "regresión"}:
        return "regression"
    return None


def _norm_input_level(v: Any) -> Optional[str]:
    s = _norm_str(v)
    if not s:
        return None
    s2 = s.lower()
    if s2 in {"row", "pair"}:
        return s2
    # aliases
    if s2 in {"rows", "fila", "filas"}:
        return "row"
    if s2 in {"pairs", "par", "pares"}:
        return "pair"
    return None


def _pick_first(*candidates: Any) -> Optional[str]:
    for c in candidates:
        s = _norm_str(c)
        if s:
            return s
    return None


def _pick_first_from_dicts(key: str, *dicts: Mapping[str, Any]) -> Optional[str]:
    for d in dicts:
        if isinstance(d, Mapping):
            v = d.get(key)
            s = _norm_str(v)
            if s:
                return s
    return None

def _norm_float(v: Any) -> Optional[float]:
    """Normaliza un valor a float o None (sin explotar)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = _norm_str(v)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def fill_context(
    *,
    family: Optional[str],
    dataset_id: Optional[str],
    model_name: Optional[str],
    metrics: Optional[Mapping[str, Any]],
    predictor_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resuelve campos críticos de contexto aplicando la precedencia.

    Args:
        family: hint (por ejemplo, query param del endpoint).
        dataset_id: hint (por ejemplo, query param o parte del run).
        model_name: hint (por ejemplo, query param o parte del run).
        metrics: dict de metrics.json.
        predictor_manifest: dict de predictor.json (si existe).

    Returns:
        Dict con claves estables para UI/API:
        ``family``, ``dataset_id``, ``model_name``, ``task_type``, ``input_level``,
        ``target_col``, ``data_source`` y opcionales (``data_plan``, ``split_mode``,
        ``val_ratio``, ``target_mode``).
    """

    m: Mapping[str, Any] = metrics or {}
    req = extract_req(m)

    pred: Mapping[str, Any] = predictor_manifest or {}
    pred_extra = pred.get("extra") if isinstance(pred.get("extra"), Mapping) else {}

    # Campos principales de identificación (preferir métricas/predictor, luego hints)
    family_out = _pick_first(
        _pick_first_from_dicts("family", req, m),
        _pick_first_from_dicts("family", pred, pred_extra),
        family,
    )

    dataset_id_out = _pick_first(
        _pick_first_from_dicts("dataset_id", req, m),
        _pick_first_from_dicts("dataset_id", pred, pred_extra),
        dataset_id,
    )

    model_name_out = _pick_first(
        _pick_first_from_dicts("model_name", req, m),
        _pick_first_from_dicts("model_name", pred, pred_extra),
        model_name,
    )

    # Campos críticos (regla exacta: req -> top-level metrics -> predictor)
    task_type_raw = _pick_first_from_dicts("task_type", req, m, pred, pred_extra)
    input_level_raw = _pick_first_from_dicts("input_level", req, m, pred, pred_extra)
    target_col_raw = _pick_first_from_dicts("target_col", req, m, pred, pred_extra)
    data_source_raw = _pick_first_from_dicts("data_source", req, m, pred, pred_extra)

    task_type_out = _norm_task_type(task_type_raw)
    input_level_out = _norm_input_level(input_level_raw)
    target_col_out = _norm_str(target_col_raw)
    data_source_out = _norm_str(data_source_raw)

    # Fallback seguro por family
    fam_key = (family_out or "").strip().lower()
    fam_fb = _FAMILY_FALLBACKS.get(fam_key)
    if fam_fb:
        if not task_type_out:
            task_type_out = fam_fb.get("task_type")
        if not input_level_out:
            input_level_out = fam_fb.get("input_level")
        if not target_col_out:
            target_col_out = _norm_str(fam_fb.get("target_col"))

    # Defaults finales (solo si todo falla)
    task_type_out = task_type_out or "classification"
    input_level_out = input_level_out or "row"

    # data_source: default estable cuando no hay información (mejor que null)
    data_source_out = data_source_out or "feature_pack"

    # Opcionales: se resuelven con misma precedencia, pero pueden quedar None
    data_plan_out = _pick_first_from_dicts("data_plan", req, m, pred, pred_extra)
    split_mode_out = _pick_first_from_dicts("split_mode", req, m, pred, pred_extra)
    val_ratio_out = _norm_float(_pick_first_from_dicts("val_ratio", req, m, pred, pred_extra))
    target_mode_out = _pick_first_from_dicts("target_mode", req, m, pred, pred_extra)

    return {
        "family": family_out,
        "dataset_id": dataset_id_out,
        "model_name": model_name_out,
        "task_type": task_type_out,
        "input_level": input_level_out,
        "target_col": target_col_out,
        "data_source": data_source_out,
        # extras opcionales
        "data_plan": _norm_str(data_plan_out),
        "split_mode": _norm_str(split_mode_out),
        "val_ratio": val_ratio_out,
        "target_mode": _norm_str(target_mode_out),
    }
