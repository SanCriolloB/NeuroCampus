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

import ast
import re
from collections import Counter

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

# Preferimos score canónico en escala 0–50 si existe; en processed se deriva si falta.
_SCORE_CANDIDATES: Sequence[str] = (
    "score_total_0_50",
    "score_base_0_50",
    "score_total",
    "score",
    "rating",
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
    """Detecta una columna de score para agregaciones (defensivo).

    Priorizamos nombres canónicos en escala 0–50 (p.ej. ``score_total_0_50``)
    y mantenemos compatibilidad con históricos que expongan ``score_total``/``score``.

    Notes
    -----
    - A diferencia de una heurística "primera columna numérica", aquí **no**
      hacemos fallback a columnas numéricas arbitrarias (IDs, cédulas, etc.).
      Si no se detecta una columna conocida, retornamos ``None`` para que el
      router responda con un error explícito.
    """
    for c in _SCORE_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Valida que `required` exista en `df` o lanza ValueError."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en histórico: {missing}")



# ---------------------------------------------------------------------------
# Normalización de dimensiones (compatibilidad de esquemas)
# ---------------------------------------------------------------------------

# En el histórico "processed" los nombres de columnas varían por origen.
# El Dashboard trabaja con nombres canónicos (docente/asignatura) porque la UI
# y el contrato del endpoint los exponen así.
#
# Ejemplo real (datasets de evaluaciones): en processed vienen como:
# - profesor (nombre docente) / cedula_profesor (id)
# - materia (nombre asignatura) / codigo_materia (id)
#
# Mantenemos un mapeo defensivo y creamos columnas canónicas cuando sea posible.
_DIMENSION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "docente": ("docente", "profesor", "nombre_docente", "teacher", "cedula_profesor"),
    "asignatura": ("asignatura", "materia", "nombre_asignatura", "course", "codigo_materia"),
}


def _ensure_dimension_column(df: pd.DataFrame, dim: str) -> Optional[str]:
    """Asegura que exista la columna canónica `dim` en `df`.

    Si `df` ya contiene `dim`, no hace nada. Si no, intenta derivarla desde alias
    conocidos (ver `_DIMENSION_ALIASES`). Retorna el nombre de la columna
    canónica que se debe usar o `None` si no se pudo derivar.

    Notes
    -----
    - Se modifica el DataFrame **in-place** (agregando una columna) para que
      `apply_filters()` y agregaciones posteriores funcionen con el mismo
      contrato (docente/asignatura) sin duplicar lógica en routers.
    """
    dim = (dim or "").strip()
    if not dim:
        return None
    if dim in df.columns:
        return dim

    aliases = _DIMENSION_ALIASES.get(dim, (dim,))
    for candidate in aliases:
        if candidate in df.columns:
            # Creamos columna canónica apuntando a la serie existente
            df[dim] = df[candidate]
            return dim
    return None


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
    # Normalizamos dimensiones para que filtros/series sean consistentes entre
    # distintos esquemas de histórico (p.ej. profesor/materia).
    _ensure_dimension_column(df, "docente")
    _ensure_dimension_column(df, "asignatura")
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
            raise ValueError(
                "No se pudo detectar columna de score para score_promedio. "
                f"candidates={list(_SCORE_CANDIDATES)}"
            )
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
# Radar (promedios por pregunta)
# ---------------------------------------------------------------------------

# Métricas del radar: preguntas 1..10 del histórico processed.
# Se exponen como `pregunta_1`..`pregunta_10` para que la UI pueda mapear
# directamente sin asumir nombres distintos.
RADAR_PREGUNTAS: Tuple[str, ...] = tuple(f"pregunta_{i}" for i in range(1, 11))


def radar_preguntas(filters: DashboardFilters) -> List[Dict[str, Any]]:
    """Promedios por pregunta (radar) desde histórico processed.

    La pestaña Dashboard muestra un radar con el desempeño promedio en las
    preguntas del instrumento (``pregunta_1``..``pregunta_10``).

    Dado que en algunos históricos las preguntas pueden venir en escala 1–5
    o 0–50, aplicamos una normalización defensiva a escala 0–50:

    - Si un valor es <= 5.5 se considera escala 1–5 y se multiplica por 10.
    - Si un valor es > 5.5 se considera ya en 0–50 y se deja igual.

    Parameters
    ----------
    filters:
        Filtros estándar del Dashboard.

    Returns
    -------
    list[dict]
        ``[{"key": "pregunta_1", "value": 41.2}, ...]``
        Cuando no hay datos para un filtro, retorna ``value=None``.
    """
    df = load_processed()
    _ensure_dimension_column(df, 'docente')
    _ensure_dimension_column(df, 'asignatura')
    df = apply_filters(df, filters)

    available = [c for c in RADAR_PREGUNTAS if c in df.columns]
    if not available:
        # No hay columnas de preguntas en el histórico; devolvemos estructura estable.
        return [{"key": c, "value": None} for c in RADAR_PREGUNTAS]

    if df.empty:
        return [{"key": c, "value": None} for c in RADAR_PREGUNTAS]

    q = df[available].apply(pd.to_numeric, errors='coerce')
    q_0_50 = q.where(q > 5.5, q * 10.0)

    means = q_0_50.mean(axis=0, skipna=True)

    out: List[Dict[str, Any]] = []
    for c in RADAR_PREGUNTAS:
        if c not in means.index:
            out.append({"key": c, "value": None})
            continue
        v = means.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append({"key": c, "value": None})
        else:
            # Clip defensivo a 0..50 para evitar outliers por datos corruptos.
            out.append({"key": c, "value": float(np.clip(v, 0.0, 50.0))})
    return out


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
    # Normalizamos dimensiones para soportar históricos con columnas como
    # profesor/materia en vez de docente/asignatura.
    _ensure_dimension_column(df, "docente")
    _ensure_dimension_column(df, "asignatura")
    df = apply_filters(df, filters)

    _require_columns(df, ["periodo", by])
    grouped = df.groupby(by, dropna=True)

    if metric == "evaluaciones":
        agg = grouped.size().astype(float)
    else:
        score_col = _detect_score_col(df)
        if score_col is None:
            raise ValueError(
                "No se pudo detectar columna de score para rankings score_promedio. "
                f"candidates={list(_SCORE_CANDIDATES)}"
            )
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


# ---------------------------------------------------------------------------
# Wordcloud (histórico labeled)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+$")

# Stopwords mínimas (no exhaustivas). El objetivo es evitar ruido en wordcloud.
_STOPWORDS = {
    "de","la","el","y","a","en","que","los","las","un","una","por","para","con","del","al","se",
    "es","no","si","muy","mas","más","pero","como","lo","le","les","su","sus","mi","mis","tu","tus",
}


def _iter_wordcloud_tokens(value: Any) -> Iterable[str]:
    """Extrae tokens desde una celda de texto/lemmas.

    Soporta valores:
    - list/tuple/set de tokens
    - string con tokens separados por espacios/','/';'
    - string con representación de lista (ej: "['a','b']")
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    if isinstance(value, (list, tuple, set)):
        for t in value:
            if t is None:
                continue
            yield str(t)
        return
    if not isinstance(value, str):
        return

    s = value.strip()
    if not s:
        return

    # Intentar parsear listas serializadas como string.
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            for t in parsed:
                if t is None:
                    continue
                yield str(t)
            return

    # Fallback: split por separadores comunes.
    for t in re.split(r"[\s,;]+", s):
        if t:
            yield t


def _pick_wordcloud_source_column(df: pd.DataFrame) -> Optional[str]:
    """Elige la mejor columna fuente disponible para wordcloud.

    Nota
    ----
    En el histórico labeled, ``sugerencias_lemmatizadas`` puede existir pero estar
    completamente vacía (todo ``None``) para ciertos periodos. Si la elegimos solo
    por existencia, el wordcloud queda vacío aunque haya texto en ``texto_lemmas``
    u otras columnas.

    Esta función elige la primera columna que realmente tenga contenido
    tokenizable para el DataFrame ya filtrado.
    """
    candidates = (
        "sugerencias_lemmatizadas",
        "texto_lemmas",
        "texto_clean",
        "texto_raw_concat",
        "comentario",
        "observaciones",
    )

    for c in candidates:
        if c not in df.columns:
            continue

        nonnull = df[c].dropna()
        if nonnull.empty:
            continue

        # Caso especial: sugerencias_lemmatizadas suele ser lista (o string de lista).
        # Validamos que existan tokens reales antes de escogerla.
        if c == "sugerencias_lemmatizadas":
            for raw in nonnull.head(200).tolist():
                for tok in _iter_wordcloud_tokens(raw):
                    if str(tok).strip():
                        return c
            continue

        # Columnas string: basta con que exista al menos un string no vacío.
        try:
            if nonnull.astype(str).str.strip().ne("").any():
                return c
        except Exception:
            for raw in nonnull.head(200).tolist():
                if isinstance(raw, str) and raw.strip():
                    return c

    return None


def wordcloud_terms(filters: DashboardFilters, limit: int = 80) -> List[Dict[str, Any]]:
    """Construye wordcloud (top términos) desde histórico labeled.

    Parameters
    ----------
    filters:
        Filtros estándar del dashboard.
    limit:
        Cantidad máxima de tokens a retornar (ordenados por frecuencia desc).

    Returns
    -------
    list[dict]
        Lista de items: ``{"text": <token>, "value": <freq>}``.
    """
    df = load_labeled()
    df = apply_filters(df, filters)

    if df is None or df.empty:
        return []

    src = _pick_wordcloud_source_column(df)
    if not src:
        return []

    counter: Counter[str] = Counter()
    for raw in df[src].dropna().tolist():
        for tok in _iter_wordcloud_tokens(raw):
            t = tok.strip().lower()
            if len(t) < 3:
                continue
            if t in _STOPWORDS:
                continue
            if not _TOKEN_RE.match(t):
                continue
            counter[t] += 1

    if not counter:
        return []

    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, int(limit))]
    return [{"text": k, "value": int(v)} for k, v in items]
