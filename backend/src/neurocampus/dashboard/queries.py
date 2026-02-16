"""neurocampus.dashboard.queries

Queries del Dashboard basadas **exclusivamente** en histórico.

Este módulo concentra la lectura y filtrado de:

- ``historico/unificado.parquet`` (processed histórico)
- ``historico/unificado_labeled.parquet`` (labeled histórico)

y expone funciones reutilizables para implementar endpoints ``/dashboard/*``
sin duplicar lógica en routers.

Reglas de negocio (fuente de verdad)
-----------------------------------
1) El Dashboard **solo consulta histórico** (no consulta datasets individuales).
2) El Dashboard filtra por **periodo** o por **rango de periodos**.
3) Los filtros de UI (docente/asignatura/programa) deben aplicarse sobre el
   histórico resultante (processed o labeled según el endpoint).

Notas de implementación
----------------------
- Se prefiere lectura de columnas mínimas cuando el caller lo solicita.
- El formato de ``periodo`` se asume como ``YYYY-T`` (ej. ``2024-1``), pero la
  comparación se implementa de forma defensiva para no romper si aparece un
  valor inesperado.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rutas del histórico (resolución consistente con otros módulos)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Devuelve la raíz del repo NeuroCampus.

    Buscamos el directorio que contenga `data/` y `datasets/` subiendo desde
    este archivo. Si no se encuentra, usamos un fallback conservador.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data").exists() and (p / "datasets").exists():
            return p
    # Fallback: layout estándar (backend/src/neurocampus/dashboard/queries.py)
    return here.parents[4]


_HIST_DIR = _repo_root() / "historico"
_PROCESSED_PATH = _HIST_DIR / "unificado.parquet"
_LABELED_PATH = _HIST_DIR / "unificado_labeled.parquet"

# ---------------------------------------------------------------------------
# Score (0–50) derivado para histórico processed/labeled
# ---------------------------------------------------------------------------

_PREGUNTA_COLS: Tuple[str, ...] = tuple(f"pregunta_{i}" for i in range(1, 11))


def _ensure_score_0_50_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura columnas de score en escala 0–50 para el Dashboard.

    El histórico "processed" (``historico/unificado.parquet``) **no** incluye
    necesariamente una columna de score/rating lista para usar. Sin embargo, sí
    contiene las respuestas ``pregunta_1``..``pregunta_10``.

    Para que KPIs/series/rankings del Dashboard sean consistentes, derivamos un
    score 0–50 con la siguiente regla:

    - Convertimos ``pregunta_1..10`` a numérico (NaN si no se puede).
    - Calculamos el promedio por fila y lo re-escalamos multiplicando por 10
      (equivalente a la sumatoria cuando están las 10 preguntas presentes).

    La función es conservadora:
    - Si ya existe alguna columna de score, solo crea aliases faltantes.
    - No escribe a disco; solo agrega columnas en memoria.

    Parameters
    ----------
    df:
        DataFrame de histórico (processed o labeled).

    Returns
    -------
    pandas.DataFrame
        El mismo DataFrame con columnas de score disponibles cuando sea posible.
    """
    if df is None or df.empty:
        return df

    # Caso 1: ya tenemos score (p.ej. labeled trae score_total_0_50)
    base_col: Optional[str] = None
    if "score_total_0_50" in df.columns:
        base_col = "score_total_0_50"
    elif "score_total" in df.columns:
        base_col = "score_total"
    elif "score" in df.columns:
        base_col = "score"

    # Caso 2: derivar desde preguntas si no hay score
    if base_col is None:
        available = [c for c in _PREGUNTA_COLS if c in df.columns]
        if not available:
            return df

        q = df[available].apply(pd.to_numeric, errors="coerce")

        # Heurística de escala:
        # - Si un valor está en 1–5 (o 0–5), lo re-escalamos a 0–50 multiplicando por 10.
        # - Si un valor ya está en 0–50 (por ejemplo 47, 38, 42...), lo dejamos tal cual.
        #
        # Umbral 5.5: evita que valores decimales como 4.7 (=> 47) se traten como 0–50.
        q_0_50 = q.where(q > 5.5, q * 10.0)

        # Score final del dashboard en 0–50: promedio de preguntas ya normalizadas.
        score_0_50 = q_0_50.mean(axis=1, skipna=True)

        df["score_total_0_50"] = score_0_50.astype(float)
        base_col = "score_total_0_50"

        logger.debug(
            "Dashboard: score 0–50 derivado desde %s (columnas agregadas: score_total_0_50/score_total/score).",
            available,
        )

    # Normalizamos a float y creamos aliases para compatibilidad con agregaciones.
    base = pd.to_numeric(df[base_col], errors="coerce").astype(float)
    df[base_col] = base

    # Alias usados por aggregations.py (candidatos: score_total, score, etc.)
    if "score_total" not in df.columns:
        df["score_total"] = base
    if "score" not in df.columns:
        df["score"] = base

    # Para compatibilidad: si ya existía score_total/score pero no score_total_0_50,
    # exponemos también el alias (solo si no existe).
    if "score_total_0_50" not in df.columns:
        df["score_total_0_50"] = base

    return df


# ---------------------------------------------------------------------------
# Parsing y comparación de periodos
# ---------------------------------------------------------------------------

def _parse_periodo(value: str) -> Optional[Tuple[int, int]]:
    """Parsea un periodo con formato ``YYYY-T`` a una tupla comparable.

    Parameters
    ----------
    value:
        Periodo en formato string (idealmente ``YYYY-T``).

    Returns
    -------
    tuple[int, int] | None
        ``(year, term)`` si se puede parsear, en caso contrario ``None``.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or "-" not in s:
        return None
    year_s, term_s = s.split("-", 1)
    try:
        return (int(year_s), int(term_s))
    except ValueError:
        return None


def _periodo_key(value: str) -> Tuple[int, int, str]:
    """Clave de ordenamiento defensiva para `periodo`.

    Si no se puede parsear, se usa un fallback con year/term en 0 y el string
    original para que el orden sea determinista.
    """
    parsed = _parse_periodo(value)
    if parsed is None:
        return (0, 0, str(value))
    return (parsed[0], parsed[1], str(value))


def _periodo_ord(value: str) -> int:
    """Ordinal para comparaciones por rango.

    Devuelve `year * 10 + term` si se puede parsear; de lo contrario 0.
    """
    parsed = _parse_periodo(value)
    if parsed is None:
        return 0
    year, term = parsed
    return year * 10 + term


def sort_periodos(items: Iterable[str]) -> List[str]:
    """Ordena periodos de forma estable y defensiva."""
    return sorted({str(x) for x in items if str(x).strip()}, key=_periodo_key)


# ---------------------------------------------------------------------------
# Filtros de Dashboard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardFilters:
    """Filtros estándar aplicables a histórico del Dashboard.

    Un caller debe definir **o** `periodo` **o** un rango (`periodo_from` /
    `periodo_to`). Si ambos se proveen, se prioriza `periodo`.
    """

    periodo: Optional[str] = None
    periodo_from: Optional[str] = None
    periodo_to: Optional[str] = None

    docente: Optional[str] = None
    asignatura: Optional[str] = None
    programa: Optional[str] = None


def _ensure_periodo_column(df: pd.DataFrame) -> None:
    """Valida que el DataFrame tenga columna `periodo`.

    El histórico debe contener `periodo` por contrato (Fase A1). Si falta,
    levantamos error explícito para diagnóstico.
    """
    if "periodo" not in df.columns:
        raise KeyError("El histórico no contiene columna 'periodo' (contrato Dashboard)")


def apply_filters(df: pd.DataFrame, f: DashboardFilters) -> pd.DataFrame:
    """Aplica filtros del Dashboard sobre un DataFrame histórico.

    Parameters
    ----------
    df:
        DataFrame leído desde histórico.
    f:
        Filtros de Dashboard.

    Returns
    -------
    pandas.DataFrame
        Subconjunto filtrado (copia ligera vía máscara booleana).

    Notas
    -----
    - Filtro por periodo/rango se aplica sobre la columna `periodo`.
    - Los filtros por docente/asignatura/programa solo se aplican si existen
      las columnas correspondientes; si no existen, se ignoran (compatibilidad
      con diferentes layouts de histórico).
    """
    if df.empty:
        return df

    # Normalizamos alias de columnas de dimensiones para que el resto del
    # pipeline pueda operar con nombres canónicos (docente/asignatura/programa)
    # incluso si el histórico viene con nombres alternativos (p.ej. profesor,
    # materia). Esto evita que endpoints como /dashboard/catalogos queden vacíos
    # en datasets reales.
    df = _normalize_dim_aliases(df)

    _ensure_periodo_column(df)

    mask = pd.Series(True, index=df.index)

    # Periodo exacto tiene prioridad sobre rango
    if f.periodo and str(f.periodo).strip():
        mask &= df["periodo"].astype(str).str.strip().eq(str(f.periodo).strip())
    else:
        # Rango inclusivo si se proveen límites
        periodo_series_ord = df["periodo"].astype(str).map(_periodo_ord)
        if f.periodo_from and str(f.periodo_from).strip():
            mask &= periodo_series_ord.ge(_periodo_ord(str(f.periodo_from).strip()))
        if f.periodo_to and str(f.periodo_to).strip():
            mask &= periodo_series_ord.le(_periodo_ord(str(f.periodo_to).strip()))

    # Filtros categóricos: solo si la columna existe en el histórico.
    if f.docente and "docente" in df.columns:
        mask &= df["docente"].astype(str).str.strip().eq(str(f.docente).strip())
    if f.asignatura and "asignatura" in df.columns:
        mask &= df["asignatura"].astype(str).str.strip().eq(str(f.asignatura).strip())
    if f.programa and "programa" in df.columns:
        mask &= df["programa"].astype(str).str.strip().eq(str(f.programa).strip())

    return df.loc[mask]


# ---------------------------------------------------------------------------
# Lectura de histórico (processed / labeled)
# ---------------------------------------------------------------------------

def load_processed(columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Carga el histórico processed desde ``historico/unificado.parquet``.

    Parameters
    ----------
    columns:
        Lista opcional de columnas a leer (optimización). Si None, lee todo.

    Raises
    ------
    FileNotFoundError
        Si el parquet no existe.

    Returns
    -------
    pandas.DataFrame
        DataFrame del histórico processed.
    """
    if not _PROCESSED_PATH.exists():
        raise FileNotFoundError(f"No existe {_PROCESSED_PATH.as_posix()} (ejecuta unificación)")
    df = pd.read_parquet(_PROCESSED_PATH, columns=_augment_columns_for_aliases(columns))
    df = _normalize_dim_aliases(df)
    df = _ensure_score_0_50_columns(df)
    return df


def load_labeled(columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Carga el histórico labeled desde ``historico/unificado_labeled.parquet``."""
    if not _LABELED_PATH.exists():
        raise FileNotFoundError(
            f"No existe {_LABELED_PATH.as_posix()} (ejecuta BETO + unificación labeled)"
        )
    df = pd.read_parquet(_LABELED_PATH, columns=_augment_columns_for_aliases(columns))
    df = _normalize_dim_aliases(df)
    df = _ensure_score_0_50_columns(df)
    return df


# ---------------------------------------------------------------------------
# Helpers para endpoints (catálogos / kpis) - usados en pasos posteriores
# ---------------------------------------------------------------------------

def compute_catalogos(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Computa catálogos (docentes, asignaturas, programas) desde histórico filtrado."""
    df = _normalize_dim_aliases(df)
    if "docente" in df.columns:
        docentes = sorted({str(x).strip() for x in df["docente"].dropna().tolist() if str(x).strip()})
    else:
        docentes = []

    if "asignatura" in df.columns:
        asignaturas = sorted(
            {str(x).strip() for x in df["asignatura"].dropna().tolist() if str(x).strip()}
        )
    else:
        asignaturas = []

    if "programa" in df.columns:
        programas = sorted({str(x).strip() for x in df["programa"].dropna().tolist() if str(x).strip()})
    else:
        programas = []

    return docentes, asignaturas, programas


def _augment_columns_for_aliases(columns: Optional[List[str]]) -> Optional[List[str]]:
    """Aumenta `columns` para soportar creación de alias de dimensiones.

    Cuando un caller pide columnas canónicas (p.ej. ``docente``), pero el parquet
    contiene el nombre alternativo (p.ej. ``profesor``), necesitamos leer también
    la columna fuente para poder generar el alias.

    La función es conservadora: si `columns` es None, no hace nada.
    """
    if columns is None:
        return None

    wanted = {str(c) for c in columns}

    # Dimensiones canónicas -> posibles alias históricos.
    if "docente" in wanted:
        wanted.add("profesor")
        wanted.add("docente_nombre")
        wanted.add("nombre_docente")
    if "asignatura" in wanted:
        wanted.add("materia")
        wanted.add("asignatura_nombre")
        wanted.add("nombre_asignatura")
    if "programa" in wanted:
        wanted.add("programa_nombre")
        wanted.add("nombre_programa")

    # Orden determinista.
    return sorted(wanted)


def _normalize_dim_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas de dimensiones para el Dashboard.

    El histórico real puede venir con nombres como ``profesor``/``materia`` en vez
    de ``docente``/``asignatura``. Para evitar duplicar heurísticas en cada
    agregación/endpoint, creamos columnas canónicas cuando sea posible.

    Notes
    -----
    - Se crean *aliases* solo si la columna canónica no existe y la fuente sí.
    - No se elimina ninguna columna original (retro-compatibilidad).
    - Se retorna el mismo DataFrame (mutado in-place) para evitar copias grandes.
    """
    if df is None or df.empty:
        return df

    # Docente
    if "docente" not in df.columns:
        for src in ("profesor", "docente_nombre", "nombre_docente"):
            if src in df.columns:
                df["docente"] = df[src]
                break

    # Asignatura
    if "asignatura" not in df.columns:
        for src in ("materia", "asignatura_nombre", "nombre_asignatura"):
            if src in df.columns:
                df["asignatura"] = df[src]
                break

    # Programa
    if "programa" not in df.columns:
        for src in ("programa_nombre", "nombre_programa"):
            if src in df.columns:
                df["programa"] = df[src]
                break

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Computa KPIs básicos sobre un histórico filtrado.

    Este helper se usará en el endpoint ``/dashboard/kpis``.
    """
    kpis = {
        "evaluaciones": int(len(df)),
        "docentes": int(df["docente"].nunique()) if "docente" in df.columns else 0,
        "asignaturas": int(df["asignatura"].nunique()) if "asignatura" in df.columns else 0,
        "score_promedio": None,
    }

    # Score/rating promedio: intentamos columnas frecuentes sin asumir una sola.
    for col in ("score", "rating", "score_total", "score_promedio"):
        if col in df.columns:
            try:
                kpis["score_promedio"] = float(pd.to_numeric(df[col], errors="coerce").mean())
            except Exception:
                kpis["score_promedio"] = None
            break

    return kpis
