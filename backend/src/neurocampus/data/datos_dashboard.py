# backend/src/neurocampus/data/datos_dashboard.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import json

from neurocampus.app.schemas.datos import (
    DatasetResumenResponse,
    DatasetPreviewResponse,
    DatasetSentimientosResponse,
    ColumnaResumen,
    SentimentLabel,
    SentimentBreakdown,
    SentimentByGroup,
)

_SENTIMENT_ALPHABET = ("neg", "neu", "pos")

# Columnas "hard label" conocidas (preferencia de uso)
_SENTIMENT_LABEL_CANDIDATES: Sequence[str] = (
    "sentiment_label_teacher",   # salida actual de BETO/teacher_labeling
    "y_sentimiento",             # etiquetas humanas/curadas
    "sentiment_label",
    "sentimiento",
    "label",
    "label_sentimiento",
    "target",
)

# Columnas de probabilidades (fallback)
_PROBA_COLS = ("p_neg", "p_neu", "p_pos")

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
    candidates = [
        # inglés
        "teacher",
        # español
        "docente", "profesor", "profesora",
        # variantes
        "nombre_docente", "nombre_profesor", "nombre_prof",
    ]

    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    return None


def _detect_asignatura_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "subject",
        "asignatura", "materia",
        "codigo_materia", "cod_materia",
        "nombre_materia", "nombre_asignatura",
    ]

    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_norm:
            return cols_norm[key]
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

def _norm_label_series(s: pd.Series) -> pd.Series:
    """Normaliza labels a {neg, neu, pos} en minúsculas; deja NA si vacío."""
    return (
        s.astype("string")
        .fillna(pd.NA)
        .str.strip()
        .str.lower()
        .replace({"negative": "neg", "neutral": "neu", "positive": "pos"})
    )


def resolve_sentiment_labels(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve una Serie (index=df.index) con labels normalizados {neg, neu, pos}.

    Prioridad:
    1) hard-label existente (sentiment_label_teacher, y_sentimiento, etc.)
    2) derivar desde probabilidades p_neg/p_neu/p_pos

    Lanza ValueError con mensaje accionable si no se puede resolver.
    """
    # 1) hard label
    for col in _SENTIMENT_LABEL_CANDIDATES:
        if col in df.columns:
            y = _norm_label_series(df[col])
            # si hay al menos algún valor válido, usamos esta columna
            if y.isin(_SENTIMENT_ALPHABET).any():
                return y

    # 2) probabilidades -> argmax
    if all(c in df.columns for c in _PROBA_COLS):
        P = df[list(_PROBA_COLS)].copy()
        # robustez ante NaN
        P = P.fillna(-1.0)
        idx = P.to_numpy(dtype=float).argmax(axis=1)
        labels = np.array(_SENTIMENT_ALPHABET, dtype=object)[idx]
        return pd.Series(labels, index=df.index, dtype="string")

    raise ValueError(
        "No se pudo resolver la etiqueta de sentimiento del dataset etiquetado. "
        f"Busqué hard-label en: {list(_SENTIMENT_LABEL_CANDIDATES)} "
        f"y/o probabilidades: {list(_PROBA_COLS)}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def try_resolve_sentiment_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    """Versión tolerante: devuelve None si no se puede resolver."""
    try:
        return resolve_sentiment_labels(df)
    except ValueError:
        return None


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

def _detect_sentiment_col(df: pd.DataFrame) -> str:
    """
    Devuelve el nombre de la columna que contiene la etiqueta de sentimiento.
    Contrato: retorna str o lanza KeyError con detalle (nunca retorna None).
    """
    candidates = list(_SENTIMENT_LABEL_CANDIDATES) + ["sentiment", "polarity"]

    # match directo
    for col in candidates:
        if col in df.columns:
            return col

    # match por normalización (minúsculas/espacios)
    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_norm:
            return cols_norm[key]

    raise KeyError(
        "No se encontró columna de sentimiento. "
        f"Se esperaban una de: {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def build_sentimientos_resumen(df: pd.DataFrame, dataset_id: str) -> DatasetSentimientosResponse:
    df = df.copy()

    # 1) Filtrado robusto de filas válidas (alineado con pipeline BETO)
    # - has_text: existe en tu parquet etiquetado
    # - accepted_by_teacher: existe en tu parquet etiquetado (gating del teacher)
    mask = pd.Series(True, index=df.index)

    if "has_text" in df.columns:
        mask &= df["has_text"].fillna(False).astype(bool)

    if "accepted_by_teacher" in df.columns:
        acc = df["accepted_by_teacher"].fillna(False).astype(bool)
        if bool(acc.any()):
            mask &= acc

    df = df.loc[mask].copy()

    # 2) Resolver columna de sentimiento (hard label). Preferimos teacher.
    sentiment_col = _detect_sentiment_col(df)

    # 3) Normalización defensiva de labels (neg/neu/pos)
    y = (
        df[sentiment_col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    df = df.loc[y.isin(["neg", "neu", "pos"])].copy()

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
    global_counts = _mk_breakdown(
        df[sentiment_col].astype("string").str.strip().str.lower().value_counts()
    )

    # por docente
    por_docente: List[SentimentByGroup] = []
    docente_col = _detect_docente_col(df)
    if docente_col:
        for docente, sub in df.groupby(docente_col):
            counts = sub[sentiment_col].astype("string").str.strip().str.lower().value_counts()
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
            counts = sub[sentiment_col].astype("string").str.strip().str.lower().value_counts()
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
# ---------------------------------------------------------------------------
# Preview tabular para UI (/datos/preview)
# ---------------------------------------------------------------------------

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _detect_id_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["id", "ID", "Id", "row_id", "codigo", "codigo_registro"])


def _detect_teacher_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "teacher", "docente", "nombre_docente", "nombre_profesor", "nombre_prof",
            "profesor", "profesora",
        ],
    )


def _detect_subject_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "subject", "asignatura", "materia", "nombre_materia",
            "codigo_materia", "cod_materia",
        ],
    )


def _detect_comment_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "comment", "comentario", "observaciones", "observacion", "obs",
            "sugerencias", "texto", "text",
        ],
    )


def _detect_rating_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        ["rating", "calificacion", "score", "promedio", "calif_promedio", "calif_media"],
    )


def _detect_rating_question_cols(df: pd.DataFrame) -> list[str]:
    # Detecta columnas tipo pregunta_1..N o calif_1..N
    cols = []
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("pregunta_") or cl.startswith("calif_"):
            # valida sufijo numérico
            suf = cl.split("_", 1)[1]
            if suf.isdigit():
                cols.append(c)
    return cols


def _compute_rating(df: pd.DataFrame) -> Optional[pd.Series]:
    # 1) Si ya existe una columna rating/calificacion, úsala
    rating_col = _detect_rating_col(df)
    if rating_col:
        s = pd.to_numeric(df[rating_col], errors="coerce")
        return s.round(2)

    # 2) Si existen pregunta_*/calif_* calcula promedio fila a fila
    qcols = _detect_rating_question_cols(df)
    if not qcols:
        return None

    sub = df[qcols].apply(pd.to_numeric, errors="coerce")
    mean = sub.mean(axis=1, skipna=True)

    # Heurística de escala:
    # - si max <= 5.5 => ya está 0..5
    # - si max <= 50  => asumir 0..50 (divide 10 -> 0..5)
    # - si max <= 100 => asumir 0..100 (divide 20 -> 0..5)
    vmax = float(pd.to_numeric(sub.max().max(), errors="coerce") or 0.0)
    if vmax <= 5.5:
        scale = 1.0
    elif vmax <= 50.0:
        scale = 10.0
    elif vmax <= 100.0:
        scale = 20.0
    else:
        scale = vmax / 5.0 if vmax > 0 else 1.0

    return (mean / scale).round(2)


def build_dataset_preview(
    df: pd.DataFrame,
    dataset_id: str,
    *,
    variant: Literal["processed", "labeled"] = "processed",
    mode: Literal["ui", "raw"] = "ui",
    limit: int = 25,
    offset: int = 0,
    source_path: Optional[str] = None,
) -> DatasetPreviewResponse:
    n_rows_total = int(len(df))
    n_cols = int(df.shape[1])

    # slice seguro
    start = max(0, int(offset))
    end = max(start, start + int(limit))
    df_slice = df.iloc[start:end].copy().reset_index(drop=True)

    if mode == "raw":
        columns = [str(c) for c in df_slice.columns]
        rows = json.loads(df_slice.to_json(orient="records", date_format="iso"))
        return DatasetPreviewResponse(
            dataset_id=dataset_id,
            variant=variant,
            mode=mode,
            source_path=source_path,
            n_rows_total=n_rows_total,
            n_cols=n_cols,
            columns=columns,
            rows=rows,
        )

    # mode == "ui": normaliza a columnas esperadas por la tabla del frontend
    id_col = _detect_id_col(df_slice)
    teacher_col = _detect_teacher_col(df_slice)
    subject_col = _detect_subject_col(df_slice)
    comment_col = _detect_comment_col(df_slice)
    sentiment_col = None
    if variant == "labeled":
        try:
            sentiment_col = _detect_sentiment_col(df_slice)
        except KeyError:
            sentiment_col = None
    rating_series = _compute_rating(df_slice)

    out = pd.DataFrame()
    out["ID"] = df_slice[id_col] if id_col else pd.Series(range(start + 1, start + 1 + len(df_slice)))
    out["Teacher"] = df_slice[teacher_col] if teacher_col else None
    out["Subject"] = df_slice[subject_col] if subject_col else None
    out["Rating"] = rating_series if rating_series is not None else None
    out["Comment"] = df_slice[comment_col] if comment_col else None

    if sentiment_col is not None:
        out["Sentiment"] = (
            df_slice[sentiment_col]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
    columns = [str(c) for c in out.columns]
    rows = json.loads(out.to_json(orient="records", date_format="iso"))

    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        variant=variant,
        mode=mode,
        source_path=source_path,
        n_rows_total=n_rows_total,
        n_cols=n_cols,
        columns=columns,
        rows=rows,
    )
