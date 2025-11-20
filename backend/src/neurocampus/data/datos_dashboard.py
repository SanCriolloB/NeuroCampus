# backend/src/neurocampus/data/datos_dashboard.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Iterable, Tuple

import pandas as pd

from neurocampus.app.schemas.datos import (
    DatasetResumenResponse,
    ColumnaResumen,
    DatasetSentimientosResponse,
    SentimentBreakdown,
    SentimentByGroup,
)


# ---------------------------------------------------------------------------
# Resolución de paths (reusa patrón de _repo_root_from_here de otros routers)
# ---------------------------------------------------------------------------

def _repo_root_from_here() -> Path:
    """
    Devuelve la raíz del repo asumiendo estructura backend/src/neurocampus/...
    """
    return Path(__file__).resolve().parents[4]


def _data_root() -> Path:
    root = _repo_root_from_here()
    return root / "data"


def _datasets_root() -> Path:
    root = _repo_root_from_here()
    return root / "datasets"


def resolve_processed_path(dataset_id: str) -> Path:
    """
    Heurística para encontrar el dataset 'procesado' que alimenta a BETO y modelos:
    1) data/processed/{dataset_id}.parquet
    2) data/processed/{dataset_id}.csv
    3) datasets/{dataset_id}.parquet
    4) datasets/{dataset_id}.csv
    """
    data_root = _data_root()
    candidates = [
        data_root / "processed" / f"{dataset_id}.parquet",
        data_root / "processed" / f"{dataset_id}.csv",
        _datasets_root() / f"{dataset_id}.parquet",
        _datasets_root() / f"{dataset_id}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró dataset procesado para '{dataset_id}'")


def resolve_labeled_path(dataset_id: str) -> Path:
    """
    Heurística para encontrar el dataset etiquetado por BETO/teacher:

    - data/labeled/{dataset_id}_beto.parquet
    - data/labeled/{dataset_id}_teacher.parquet
    """
    data_root = _data_root()
    candidates = [
        data_root / "labeled" / f"{dataset_id}_beto.parquet",
        data_root / "labeled" / f"{dataset_id}_teacher.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró dataset etiquetado para '{dataset_id}'")


# ---------------------------------------------------------------------------
# Helpers de resumen de dataset
# ---------------------------------------------------------------------------

def _detect_fecha_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["fecha", "fecha_evaluacion", "fecha_eval", "FECHA", "Fecha"]:
        if col in df.columns:
            return col
    return None


def _detect_docente_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["docente", "nombre_docente", "nombre_profesor", "nombre_prof"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _detect_asignatura_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["asignatura", "materia", "codigo_materia", "nombre_materia"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_processed_dataset(dataset_id: str) -> pd.DataFrame:
    path = resolve_processed_path(dataset_id)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def load_labeled_dataset(dataset_id: str) -> pd.DataFrame:
    path = resolve_labeled_path(dataset_id)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def build_dataset_resumen(df: pd.DataFrame, dataset_id: str) -> DatasetResumenResponse:
    df = df.copy()
    n_rows, n_cols = df.shape

    # periodos
    periodos: List[str] = []
    if "periodo" in df.columns:
        periodos = sorted({str(v) for v in df["periodo"].dropna().unique().tolist()})

    # fechas
    fecha_min = fecha_max = None
    fecha_col = _detect_fecha_col(df)
    if fecha_col:
        try:
            fechas = pd.to_datetime(df[fecha_col], errors="coerce")
            if not fechas.dropna().empty:
                fecha_min = fechas.min().date()
                fecha_max = fechas.max().date()
        except Exception:
            pass

    # docentes/asignaturas
    n_docentes = n_asignaturas = None
    docente_col = _detect_docente_col(df)
    if docente_col:
        n_docentes = int(df[docente_col].dropna().nunique())

    asignatura_col = _detect_asignatura_col(df)
    if asignatura_col:
        n_asignaturas = int(df[asignatura_col].dropna().nunique())

    # resumen de columnas
    columnas: List[ColumnaResumen] = []
    for col in df.columns:
        serie = df[col]
        non_nulls = int(serie.notna().sum())
        # Muestra de hasta 5 valores distintos
        uniq = serie.dropna().astype(str).unique().tolist()
        sample_values = uniq[:5]
        columnas.append(
            ColumnaResumen(
                name=str(col),
                dtype=str(serie.dtype),
                non_nulls=non_nulls,
                sample_values=sample_values,
            )
        )

    return DatasetResumenResponse(
        dataset_id=dataset_id,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        periodos=periodos,
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        n_docentes=n_docentes,
        n_asignaturas=n_asignaturas,
        columns=columnas,
    )


# ---------------------------------------------------------------------------
# Helpers de resumen de sentimientos (BETO/teacher)
# ---------------------------------------------------------------------------

def _detect_sentiment_col(df: pd.DataFrame) -> str:
    """
    Intenta encontrar la columna de etiqueta de sentimiento.
    """
    candidates = [
        "sentiment_label_teacher",
        "sentiment_label",
        "sentiment",
        "sent_docente",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"No se encontró columna de sentimiento en el dataset; "
        f"se buscaron: {', '.join(candidates)}"
    )


def _mk_breakdown(counts: pd.Series) -> List[SentimentBreakdown]:
    total = int(counts.sum()) or 1
    out: List[SentimentBreakdown] = []
    for label in ["neg", "neu", "pos"]:
        c = int(counts.get(label, 0))
        out.append(
            SentimentBreakdown(
                label=label,
                count=c,
                proportion=float(c / total),
            )
        )
    return out


def build_sentimientos_resumen(df: pd.DataFrame, dataset_id: str) -> DatasetSentimientosResponse:
    df = df.copy()

    # columna de texto para filtrar comentarios vacíos (si existe)
    text_col = None
    for cand in ["comentario", "comentarios", "Sugerencias", "sugerencias"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col:
        df = df[df[text_col].astype(str).str.strip() != ""]

    sentiment_col = _detect_sentiment_col(df)

    total = int(len(df))
    if total == 0:
        return DatasetSentimientosResponse(
            dataset_id=dataset_id,
            total_comentarios=0,
            global_counts=_mk_breakdown(pd.Series(dtype=int)),
            por_docente=[],
            por_asignatura=[],
        )

    # global
    global_counts = _mk_breakdown(df[sentiment_col].value_counts())

    # por docente
    por_docente: List[SentimentByGroup] = []
    docente_col = _detect_docente_col(df)
    if docente_col:
        for docente, sub in df.groupby(docente_col):
            counts = sub[sentiment_col].value_counts()
            por_docente.append(
                SentimentByGroup(
                    group=str(docente),
                    counts=_mk_breakdown(counts),
                )
            )

    # por asignatura
    por_asignatura: List[SentimentByGroup] = []
    asignatura_col = _detect_asignatura_col(df)
    if asignatura_col:
        for asig, sub in df.groupby(asignatura_col):
            counts = sub[sentiment_col].value_counts()
            por_asignatura.append(
                SentimentByGroup(
                    group=str(asig),
                    counts=_mk_breakdown(counts),
                )
            )

    return DatasetSentimientosResponse(
        dataset_id=dataset_id,
        total_comentarios=total,
        global_counts=global_counts,
        por_docente=por_docente,
        por_asignatura=por_asignatura,
    )
