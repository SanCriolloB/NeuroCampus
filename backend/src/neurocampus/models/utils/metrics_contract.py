"""
neurocampus.models.utils.metrics_contract
==========================================

Contrato central de métricas por family.

Define:
- Qué métrica principal usar por family (primary_metric + mode min/max).
- Cómo normalizar y estandarizar el dict de métricas de un run.
- Qué campos meta deben existir siempre en metrics.json.

Uso típico
----------
::

    from neurocampus.models.utils.metrics_contract import standardize_run_metrics

    # En el router, antes de guardar metrics.json:
    final_metrics = standardize_run_metrics(
        raw_metrics,
        family="sentiment_desempeno",
        task_type="classification",
    )

"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Versión del contrato (bumpar si cambia el formato)
# ---------------------------------------------------------------------------
METRICS_VERSION: int = 1

# ---------------------------------------------------------------------------
# Tabla de contratos por family
# ---------------------------------------------------------------------------
_FAMILY_CONTRACT: Dict[str, Dict[str, Any]] = {
    # Clasificación de sentimiento / desempeño
    "sentiment_desempeno": {
        "primary_metric": "val_f1_macro",
        "primary_metric_mode": "max",
        "required_val": ["val_f1_macro", "val_accuracy"],
        "required_train": ["f1_macro", "accuracy"],
        "task_type": "classification",
    },
    # Regresión de score docente
    "score_docente": {
        "primary_metric": "val_rmse",
        "primary_metric_mode": "min",
        "required_val": ["val_rmse", "val_mae", "val_r2"],
        "required_train": ["train_rmse", "train_mae", "train_r2"],
        "task_type": "regression",
    },
}

# Fallbacks por task_type cuando la family no está en la tabla
_TASK_TYPE_FALLBACK: Dict[str, Dict[str, Any]] = {
    "classification": {
        "primary_metric": "val_f1_macro",
        "primary_metric_mode": "max",
        "required_val": ["val_f1_macro", "val_accuracy"],
        "required_train": ["f1_macro", "accuracy"],
    },
    "regression": {
        "primary_metric": "val_rmse",
        "primary_metric_mode": "min",
        "required_val": ["val_rmse", "val_mae", "val_r2"],
        "required_train": ["train_rmse", "train_mae", "train_r2"],
    },
}

# Mapeo de keys legacy → keys estándar
_LEGACY_KEY_MAP: Dict[str, str] = {
    "f1": "f1_macro",
    "acc": "accuracy",
    "val_f1": "val_f1_macro",
    "val_acc": "val_accuracy",
}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def primary_metric_for_family(family: str, task_type: str = "") -> Tuple[str, str]:
    """
    Retorna ``(primary_metric, mode)`` para una family.

    ``mode`` es ``"max"`` o ``"min"``.

    Si la family no está en el contrato, usa el fallback por task_type.
    Si tampoco hay task_type conocido, retorna ``("loss", "min")``.

    Ejemplos
    --------
    >>> primary_metric_for_family("sentiment_desempeno")
    ('val_f1_macro', 'max')
    >>> primary_metric_for_family("score_docente")
    ('val_rmse', 'min')
    """
    family_key = str(family or "").lower().strip()
    contract = _FAMILY_CONTRACT.get(family_key)
    if contract:
        return contract["primary_metric"], contract["primary_metric_mode"]

    tt = str(task_type or "").lower().strip()
    fallback = _TASK_TYPE_FALLBACK.get(tt)
    if fallback:
        return fallback["primary_metric"], fallback["primary_metric_mode"]

    return "loss", "min"


def standardize_run_metrics(
    raw: Dict[str, Any],
    *,
    family: str,
    task_type: str,
) -> Dict[str, Any]:
    """
    Normaliza y estandariza el dict de métricas de un run.

    Acciones:
    1. Mapea keys legacy (``f1`` → ``f1_macro``, etc.).
    2. Asegura que los campos requeridos existan (``None`` si faltan).
    3. Calcula ``primary_metric_value``:
       - Usa ``val_*`` si existe y no es ``None``.
       - Fallback al equivalente de train.
       - Si nada, ``None``.
    4. Añade ``primary_metric``, ``primary_metric_mode``, ``metrics_version``.
    5. Preserva todos los campos originales (no elimina nada).

    Parámetros
    ----------
    raw:
        Dict de métricas sin procesar.
    family:
        Family del run (``"sentiment_desempeno"``, ``"score_docente"``, etc.).
    task_type:
        Tipo de tarea (``"classification"``, ``"regression"``).

    Retorna
    -------
    Nuevo dict con los campos estandarizados añadidos.
    """
    out = dict(raw)  # copia; no muta el original

    # 1) Mapear keys legacy
    for old_key, new_key in _LEGACY_KEY_MAP.items():
        if old_key in out and new_key not in out:
            out[new_key] = out[old_key]

    # 2) Resolver contrato
    family_key = str(family or "").lower().strip()
    tt = str(task_type or "").lower().strip()
    contract = _FAMILY_CONTRACT.get(family_key) or _TASK_TYPE_FALLBACK.get(tt) or {}

    primary_metric: str = contract.get("primary_metric", "loss")
    primary_metric_mode: str = contract.get("primary_metric_mode", "min")
    required_val: list = contract.get("required_val", [])
    required_train: list = contract.get("required_train", [])

    # 3) Asegurar que los campos requeridos existan (None si faltan)
    for k in required_val + required_train:
        if k not in out:
            out[k] = None

    # 4) Calcular primary_metric_value
    #    - Preferir val
    #    - Fallback: train equivalente (quitar "val_" prefijo → "train_" o sin prefijo)
    primary_value: Optional[float] = None
    val_candidate = out.get(primary_metric)
    if isinstance(val_candidate, (int, float)):
        primary_value = float(val_candidate)
    else:
        # Buscar fallback en train
        train_key = _train_key_for(primary_metric)
        train_candidate = out.get(train_key)
        if isinstance(train_candidate, (int, float)):
            primary_value = float(train_candidate)

    # 5) Añadir campos de contrato
    out["primary_metric"] = primary_metric
    out["primary_metric_mode"] = primary_metric_mode
    out["primary_metric_value"] = primary_value
    out["metrics_version"] = METRICS_VERSION

    return out


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _train_key_for(val_key: str) -> str:
    """
    Devuelve la key de train equivalente a una val_key.

    Ejemplos:
      ``val_f1_macro``  → ``f1_macro``
      ``val_rmse``      → ``train_rmse``
      ``val_accuracy``  → ``accuracy``
    """
    if val_key.startswith("val_"):
        base = val_key[4:]  # quitar "val_"
        # Para regresión: val_rmse → train_rmse
        for regression_suffix in ("rmse", "mae", "r2"):
            if base == regression_suffix:
                return f"train_{base}"
        # Para clasificación: val_f1_macro → f1_macro, val_accuracy → accuracy
        return base
    return val_key
