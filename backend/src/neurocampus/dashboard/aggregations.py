"""neurocampus.dashboard.aggregations

Agregaciones del Dashboard basadas **exclusivamente** en histórico.

Este módulo se apoya en :mod:`neurocampus.dashboard.queries` para:

- cargar ``historico/unificado.parquet`` (processed)
- cargar ``historico/unificado_labeled.parquet`` (labeled)
- aplicar filtros estándar (periodo / rango / docente / asignatura / programa)

y expone agregaciones reutilizables para endpoints como:

- ``GET /dashboard/series`` (evolución por periodo)
- ``GET /dashboard/sentimiento`` (distribución/series de sentimiento)
- ``GET /dashboard/rankings`` (top/bottom por docente o asignatura)

Decisiones de diseño
--------------------
- Primera versión (según plan): agregar con pandas y mantener código defensivo.
- Evitamos asumir un esquema rígido: detectamos columnas comunes (p.ej. score)
  y si faltan retornamos errores explícitos para que el router responda 400/424.

Referencias
-----------
Plan de trabajo Dashboard (Fase C, C3.3): se recomienda exponer series,
sentimiento y rankings como endpoints independientes.  # noqa: D400
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neurocampus.dashboard.queries import (
    DashboardFilters,
    apply_filters,
    load_labeled,
    load_processed,
    sort_periodos,
)


# ---------------------------------------------------------------------------
# Constantes y detección de columnas (defensivo)
# ---------------------------------------------------------------------------

# Preferimos score_total (labeled) si existe; en processed puede no existir.
_SCORE_CANDIDATES: Sequence[str] = (
    "score_total",
    "score",
    "score_promedio",
    "promedio",
    "calificacion",
    "calif",
)

# Reutilizamos el mismo alfabeto y heurísticas de dataset-level dashboard
# (ver neurocampus.data.datos_dashboard).
_SENTIMENT_ALPHABET: Tuple[str, str, str] = ("neg", "neu", "pos")
_SENTIMENT_LABEL_CANDIDATES: Sequence[str] = (
    "sentiment_label_teacher",  # salida típica BETO/teacher labeling
    "y_sentimiento",            # etiquetas humanas/curadas
    "sentiment_label",
    "sentimiento",
    "label",
    "label_sentimiento",
    "target",
)
_SENTIMENT_PROBA_COLS: Tuple[str, str, str] = ("p_neg", "p_neu", "p_pos")


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Retorna la primera columna que exista en el DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_score_col(df: pd.DataFrame) -> Optional[str]:
    """Detecta una columna numérica de score para agregaciones.

    La intención es mantener retro-compatibilidad entre datasets donde el score
    pueda llamarse distinto (p.ej. ``score_total`` en labeled).
    """
    for c in _SCORE_CANDIDATES:
        if c in df.columns:
            return c
    # Fallback: primera columna numérica (excluyendo ids/textos comunes).
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    # Orden determinista por nombre.
    numeric_cols.sort(key=lambda x: str(x))
    return str(numeric_cols[0])


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Valida que `required` exista en `df` o lanza ValueError."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en histórico: {missing}")


def _safe_mean(series: pd.Series) -> Optional[float]:
    """Mean numérico tolerante a NaN; retorna None si no hay datos."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if x.dropna().empty:
        return None
    return float(x.mean())


# ---------------------------------------------------------------------------
# Series por periodo
# ---------------------------------------------------------------------------

SUPPORTED_SERIES_METRICS: Tuple[str, ...] = (
    "evaluaciones",
    "score_promedio",
    "docentes",
    "asignaturas",
)


def series_por_periodo(metric: str, filters: DashboardFilters) -> List[Dict[str, Any]]:
    """Calcula una serie agregada por ``periodo`` desde processed histórico.

    Parameters
    ----------
    metric:
        Métrica de la serie. Soportadas:
        - ``evaluaciones``: conteo de filas.
        - ``score_promedio``: promedio de una columna de score detectada.
        - ``docentes``: número de docentes únicos (si existe columna ``docente``).
        - ``asignaturas``: número de asignaturas únicas (si existe columna ``asignatura``).
    filters:
        Filtros estándar del Dashboard (periodo o rango y dimensiones).

    Returns
    -------
    list[dict]
        Lista de puntos con la forma:
        ``[{"periodo": "2024-1", "value": 123.0}, ...]``.

    Raises
    ------
    ValueError
        Si `metric` no está soportada o faltan columnas necesarias.
    FileNotFoundError
        Si el histórico processed no existe.
    """
    metric = (metric or "").strip()
    if metric not in SUPPORTED_SERIES_METRICS:
        raise ValueError(f"metric no soportada: {metric}. Soportadas={SUPPORTED_SERIES_METRICS}")

    # Carga defensiva: por ahora leemos todo (primera versión), pero dejamos
    # abierta la optimización por columnas en el futuro.
    df = load_processed()
    df = apply_filters(df, filters)

    _require_columns(df, ["periodo"])

    grouped = df.groupby("periodo", dropna=False)
    if metric == "evaluaciones":
        ser = grouped.size()
    elif metric == "docentes":
        _require_columns(df, ["docente"])
        ser = grouped["docente"].nunique(dropna=True)
    elif metric == "asignaturas":
        _require_columns(df, ["asignatura"])
        ser = grouped["asignatura"].nunique(dropna=True)
    else:  # score_promedio
        score_col = _detect_score_col(df)
        if score_col is None:
            raise ValueError("No se pudo detectar columna de score para score_promedio")
        ser = grouped[score_col].apply(_safe_mean)

    # Asegurar orden UX consistente (periodos ordenados).
    items: List[Dict[str, Any]] = []
    for periodo in sort_periodos([str(x) for x in ser.index.tolist()]):
        v = ser.get(periodo)
        # pandas puede devolver np types / None; normalizamos.
        if v is None or (isinstance(v, float) and np.isnan(v)):
            value = None
        else:
            value = float(v)
        items.append({"periodo": str(periodo), "value": value})
    return items


# ---------------------------------------------------------------------------
# Rankings (top/bottom) por docente o asignatura
# ---------------------------------------------------------------------------

SUPPORTED_RANKING_BY: Tuple[str, ...] = ("docente", "asignatura")
SUPPORTED_RANKING_METRICS: Tuple[str, ...] = ("score_promedio", "evaluaciones")


def rankings(
    *,
    by: str,
    metric: str,
    order: str,
    limit: int,
    filters: DashboardFilters,
) -> List[Dict[str, Any]]:
    """Calcula rankings por docente o asignatura.

    Parameters
    ----------
    by:
        Dimensión del ranking: ``docente`` o ``asignatura``.
    metric:
        Métrica para ordenar:
        - ``score_promedio``: promedio de score detectado
        - ``evaluaciones``: conteo de filas
    order:
        ``asc`` o ``desc``.
    limit:
        Máximo de items a retornar (top N).
    filters:
        Filtros estándar.

    Returns
    -------
    list[dict]
        ``[{"key": "<docente>", "value": 12.3}, ...]``
    """
    by = (by or "").strip()
    metric = (metric or "").strip()
    order = (order or "desc").strip().lower()

    if by not in SUPPORTED_RANKING_BY:
        raise ValueError(f"by no soportado: {by}. Soportados={SUPPORTED_RANKING_BY}")
    if metric not in SUPPORTED_RANKING_METRICS:
        raise ValueError(f"metric no soportada: {metric}. Soportadas={SUPPORTED_RANKING_METRICS}")
    if order not in ("asc", "desc"):
        raise ValueError("order debe ser 'asc' o 'desc'")

    try:
        limit_i = int(limit)
    except Exception:
        limit_i = 10
    limit_i = max(1, min(limit_i, 200))

    df = load_processed()
    df = apply_filters(df, filters)

    _require_columns(df, ["periodo", by])
    grouped = df.groupby(by, dropna=True)

    if metric == "evaluaciones":
        agg = grouped.size().astype(float)
    else:
        score_col = _detect_score_col(df)
        if score_col is None:
            raise ValueError("No se pudo detectar columna de score para rankings score_promedio")
        agg = grouped[score_col].apply(_safe_mean).astype(float)

    asc = order == "asc"
    agg = agg.sort_values(ascending=asc)

    out: List[Dict[str, Any]] = []
    for key, value in agg.head(limit_i).items():
        k = str(key)
        v = None if (value is None or (isinstance(value, float) and np.isnan(value))) else float(value)
        out.append({"key": k, "value": v})
    return out


# ---------------------------------------------------------------------------
# Sentimiento (requiere histórico labeled)
# ---------------------------------------------------------------------------

def _normalize_sentiment_label(v: Any) -> Optional[str]:
    """Normaliza un label de sentimiento a `neg|neu|pos`.

    Acepta variantes comunes:
    - neg/neu/pos
    - negativo/neutral/positivo
    - negative/neutral/positive
    - -1/0/1 (como string o numérico)
    """
    if v is None:
        return None

    # Caso numérico (incluye numpy scalars)
    try:
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            if float(v) < 0:
                return "neg"
            if float(v) > 0:
                return "pos"
            return "neu"
    except Exception:
        pass

    s = str(v).strip().lower()
    if not s:
        return None

    mapping = {
        "neg": "neg",
        "neu": "neu",
        "pos": "pos",
        "negativo": "neg",
        "neutral": "neu",
        "positivo": "pos",
        "negative": "neg",
        "positive": "pos",
        # algunas fuentes usan 0/1/2 o -1/0/1 como string
        "-1": "neg",
        "0": "neu",
        "1": "pos",
        "2": "pos",
    }
    return mapping.get(s)


def sentimiento_distribucion(filters: DashboardFilters) -> Dict[str, Any]:
    """Distribución de sentimiento desde histórico labeled.

    Retorna conteos y porcentajes por (neg, neu, pos) y metadata de la fuente usada.

    Parameters
    ----------
    filters:
        Filtros estándar del Dashboard (periodo o rango y dimensiones).

    Returns
    -------
    dict
        Diccionario con:
        - ``buckets``: lista ``[{"label": "neg", "value": 0.25}, ...]`` (proporción 0..1)
        - ``counts``: conteos absolutos por label
        - ``total``: total de filas válidas consideradas
        - ``source``: ``"label"`` o ``"proba"``
        - ``column`` / ``columns``: columna(s) utilizadas

    Raises
    ------
    FileNotFoundError
        Si no existe ``historico/unificado_labeled.parquet``.
    ValueError
        Si no hay columnas compatibles con sentimiento en el histórico labeled.
    """
    df = load_labeled()
    df = apply_filters(df, filters)

    # Respuesta estable aun cuando no haya datos en el rango/periodo.
    if df.empty:
        counts = {k: 0 for k in _SENTIMENT_ALPHABET}
        return {
            "buckets": [{"label": k, "value": 0.0} for k in _SENTIMENT_ALPHABET],
            "counts": counts,
            "total": 0,
            "source": None,
            "column": None,
        }

    # 1) Preferimos labels explícitos si existen (mejor para auditoría).
    label_col = _first_existing_col(df, _SENTIMENT_LABEL_CANDIDATES)
    if label_col is not None:
        labels = df[label_col].map(_normalize_sentiment_label)
        vc = labels.dropna().value_counts().to_dict()
        counts = {k: int(vc.get(k, 0)) for k in _SENTIMENT_ALPHABET}
        total = int(sum(counts.values()))
        buckets = (
            [{"label": k, "value": (counts[k] / total) if total else 0.0} for k in _SENTIMENT_ALPHABET]
        )
        return {
            "buckets": buckets,
            "counts": counts,
            "total": total,
            "source": "label",
            "column": label_col,
        }

    # 2) Fallback: probabilidades p_neg/p_neu/p_pos (si están completas).
    if all(c in df.columns for c in _SENTIMENT_PROBA_COLS):
        probs = df[list(_SENTIMENT_PROBA_COLS)].apply(pd.to_numeric, errors="coerce")
        arr = probs.to_numpy(dtype=float)
        valid = np.isfinite(arr).any(axis=1)
        if not bool(valid.any()):
            counts = {k: 0 for k in _SENTIMENT_ALPHABET}
            return {
                "buckets": [{"label": k, "value": 0.0} for k in _SENTIMENT_ALPHABET],
                "counts": counts,
                "total": 0,
                "source": "proba",
                "columns": list(_SENTIMENT_PROBA_COLS),
            }

        idx = np.nanargmax(arr[valid], axis=1)
        labels = [ _SENTIMENT_ALPHABET[int(i)] for i in idx.tolist() ]
        vc = pd.Series(labels).value_counts().to_dict()
        counts = {k: int(vc.get(k, 0)) for k in _SENTIMENT_ALPHABET}
        total = int(sum(counts.values()))
        buckets = (
            [{"label": k, "value": (counts[k] / total) if total else 0.0} for k in _SENTIMENT_ALPHABET]
        )
        return {
            "buckets": buckets,
            "counts": counts,
            "total": total,
            "source": "proba",
            "columns": list(_SENTIMENT_PROBA_COLS),
        }

    raise ValueError(
        "No se detectaron columnas de sentimiento en histórico labeled. "
        f"Se buscó labels={list(_SENTIMENT_LABEL_CANDIDATES)} o probas={list(_SENTIMENT_PROBA_COLS)}."
    )


def sentimiento_serie_por_periodo(filters: DashboardFilters) -> List[Dict[str, Any]]:
    """Serie de sentimiento agregada por ``periodo`` desde histórico labeled.

    Devuelve puntos por periodo con proporciones (0..1) para neg/neu/pos.

    Returns
    -------
    list[dict]
        ``[{"periodo": "2024-1", "neg": 0.1, "neu": 0.2, "pos": 0.7, "total": 123}, ...]``

    Raises
    ------
    FileNotFoundError
        Si no existe ``historico/unificado_labeled.parquet``.
    ValueError
        Si no hay columnas compatibles con sentimiento.
    """
    df = load_labeled()
    df = apply_filters(df, filters)
    _require_columns(df, ["periodo"])

    if df.empty:
        return []

    label_col = _first_existing_col(df, _SENTIMENT_LABEL_CANDIDATES)
    use_proba = label_col is None and all(c in df.columns for c in _SENTIMENT_PROBA_COLS)
    if label_col is None and not use_proba:
        raise ValueError(
            "No se detectaron columnas de sentimiento para serie por periodo. "
            f"labels={list(_SENTIMENT_LABEL_CANDIDATES)} probas={list(_SENTIMENT_PROBA_COLS)}"
        )

    points: List[Dict[str, Any]] = []
    for periodo in sort_periodos(df["periodo"].astype(str).unique().tolist()):
        g = df.loc[df["periodo"].astype(str) == str(periodo)]

        if label_col is not None:
            labels = g[label_col].map(_normalize_sentiment_label).dropna()
            vc = labels.value_counts().to_dict()
            counts = {k: int(vc.get(k, 0)) for k in _SENTIMENT_ALPHABET}
            total = int(sum(counts.values()))
        else:
            probs = g[list(_SENTIMENT_PROBA_COLS)].apply(pd.to_numeric, errors="coerce")
            arr = probs.to_numpy(dtype=float)
            valid = np.isfinite(arr).any(axis=1)
            if not bool(valid.any()):
                total = 0
                counts = {k: 0 for k in _SENTIMENT_ALPHABET}
            else:
                idx = np.nanargmax(arr[valid], axis=1)
                labels = [ _SENTIMENT_ALPHABET[int(i)] for i in idx.tolist() ]
                vc = pd.Series(labels).value_counts().to_dict()
                counts = {k: int(vc.get(k, 0)) for k in _SENTIMENT_ALPHABET}
                total = int(sum(counts.values()))

        if total <= 0:
            points.append(
                {"periodo": str(periodo), "neg": 0.0, "neu": 0.0, "pos": 0.0, "total": 0}
            )
        else:
            points.append(
                {
                    "periodo": str(periodo),
                    "neg": counts["neg"] / total,
                    "neu": counts["neu"] / total,
                    "pos": counts["pos"] / total,
                    "total": total,
                }
            )

    return points
