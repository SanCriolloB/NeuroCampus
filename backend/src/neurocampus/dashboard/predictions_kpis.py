"""neurocampus.dashboard.predictions_kpis

Helpers para KPIs del Dashboard basados en artifacts.

Este módulo cuenta predicciones persistidas en el layout:
    <repo_root>/artifacts/predictions/<dataset_id>/**/predictions.parquet

Diseño
------
- El Dashboard, por regla de negocio, SOLO consulta histórico para datos base.
- Sin embargo, el KPI "Predicciones Totales" proviene de artifacts/predictions.
- Este helper es robusto ante layouts variables (family/run/input_level).

Notas
-----
El conteo sin filtros usa metadata de Parquet cuando es posible (rápido).
Si se aplican filtros (docente/asignatura), se intenta lectura columnar mínima.
"""


from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import re

_TEACHER_COL_CANDIDATES = [
    "teacher",
    "docente",
    "teacher_key",
    "docente_key",
    "teacher_id",
    "docente_id",
    "teacher_name",
    "docente_nombre",
]

_SUBJECT_COL_CANDIDATES = [
    "materia",
    "asignatura",
    "subject",
    "materia_key",
    "asignatura_key",
    "subject_key",
]


_PERIOD_RE = re.compile(r"^\d{4}-\d+$")

def _parse_periodo(periodo: str) -> tuple[int, int]:
    """
    Convierte un periodo 'YYYY-1' / 'YYYY-2' a tupla (YYYY, semestre).

    Nota:
    - Hay dataset_ids históricos que NO son semestre (p.ej. 'evaluaciones_2025').
      Para rangos, esos deben ignorarse (no deben romper el servicio).
    """
    s = str(periodo).strip()
    if not _PERIOD_RE.match(s):
        raise ValueError(f"Periodo no parseable: {periodo!r}")
    y, t = s.split("-")
    return int(y), int(t)


def _periodo_in_range(periodo: str, periodo_from: str, periodo_to: str) -> bool:
    """
    True si `periodo` está dentro del rango [periodo_from, periodo_to], inclusivo.

    Si `periodo` no es parseable (no es tipo 'YYYY-n'), retorna False para no romper.
    """
    try:
        p = _parse_periodo(periodo)
        pf = _parse_periodo(periodo_from)
        pt = _parse_periodo(periodo_to)
    except ValueError:
        return False
    return pf <= p <= pt


def resolve_dataset_ids_from_period_filters(
    *,
    available_periodos: list[str],
    periodo: Optional[str] = None,
    periodo_from: Optional[str] = None,
    periodo_to: Optional[str] = None,
) -> list[str]:
    """
    Resuelve dataset_ids (periodos) a usar según filtros.

    Reglas:
    - Si viene `periodo`, se retorna exactamente ese valor (aunque no sea 'YYYY-n').
    - Si viene rango (from/to), se filtran solo los que son comparables 'YYYY-n'.
    - Si no viene nada, se retorna todo lo disponible.
    """
    if periodo:
        return [periodo]

    if periodo_from and periodo_to:
        return [p for p in available_periodos if _periodo_in_range(p, periodo_from, periodo_to)]

    return list(available_periodos)


def _iter_prediction_files(predictions_root: Path, dataset_id: str) -> list[Path]:
    """Retorna todos los parquets de predicción asociados al dataset_id."""
    base = predictions_root / dataset_id
    if not base.exists():
        return []
    files = list(base.rglob("predictions.parquet"))
    if files:
        return files
    return [p for p in base.rglob("*.parquet") if "predict" in p.name.lower()]


def _pick_latest(files: list[Path]) -> Optional[Path]:
    """Selecciona el parquet más reciente según mtime."""
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _normalize_str(x: str) -> str:
    """Normalización ligera para comparar filtros (strip + upper)."""
    return str(x).strip().upper()


def _count_rows_with_optional_filters(
    parquet_path: Path,
    *,
    docente: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> int:
    """Cuenta filas en un parquet; filtra por docente/asignatura si aplica."""
    if not docente and not asignatura:
        try:
            import pyarrow.parquet as pq  # type: ignore

            return int(pq.ParquetFile(parquet_path).metadata.num_rows)
        except Exception:
            return int(len(pd.read_parquet(parquet_path)))

    try:
        import pyarrow.parquet as pq  # type: ignore

        cols = pq.ParquetFile(parquet_path).schema.names
    except Exception:
        cols = []

    teacher_col = next((c for c in _TEACHER_COL_CANDIDATES if c in cols), None) if cols else None
    subject_col = next((c for c in _SUBJECT_COL_CANDIDATES if c in cols), None) if cols else None

    read_cols: list[str] = []
    if docente and teacher_col:
        read_cols.append(teacher_col)
    if asignatura and subject_col:
        read_cols.append(subject_col)

    if (docente and not teacher_col) or (asignatura and not subject_col) or not read_cols:
        try:
            import pyarrow.parquet as pq  # type: ignore

            return int(pq.ParquetFile(parquet_path).metadata.num_rows)
        except Exception:
            return int(len(pd.read_parquet(parquet_path)))

    df = pd.read_parquet(parquet_path, columns=read_cols)

    if docente and teacher_col:
        dval = _normalize_str(docente)
        df = df[df[teacher_col].astype(str).map(_normalize_str) == dval]
    if asignatura and subject_col:
        aval = _normalize_str(asignatura)
        df = df[df[subject_col].astype(str).map(_normalize_str) == aval]

    return int(len(df))


def count_predicciones_total(
    *,
    artifacts_dir: Path,
    dataset_ids: list[str],
    docente: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> int:
    """Cuenta predicciones persistidas para varios dataset_ids (periodos)."""
    predictions_root = artifacts_dir / "predictions"
    if not predictions_root.exists():
        return 0

    total = 0
    for ds in dataset_ids:
        latest = _pick_latest(_iter_prediction_files(predictions_root, ds))
        if latest is None:
            continue
        total += _count_rows_with_optional_filters(latest, docente=docente, asignatura=asignatura)
    return int(total)
