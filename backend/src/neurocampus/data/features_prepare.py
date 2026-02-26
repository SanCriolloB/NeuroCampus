# backend/src/neurocampus/data/features_prepare.py
"""
Feature-pack builder para NeuroCampus (pestaña Datos).

Genera artefactos persistentes:
- artifacts/features/<dataset_id>/train_matrix.parquet
- teacher_index.json, materia_index.json, bins.json, meta.json

Diseño:
- El "labeled" (BETO + embeddings) se mantiene sin one-hot/bins.
- El feature-pack agrega representación (bins/índices/one-hot) fuera del labeled.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import numpy as np
from neurocampus.data.score_total import ensure_score_columns, load_sidecar_score_meta


@dataclass(frozen=True)
class FeaturePackConfig:
    """Configuración estable para bins/representación."""
    score_bins: Tuple[int, ...] = (0, 10, 20, 30, 40, 50)
    score_q_labels: Tuple[int, ...] = (0, 1, 2, 3, 4)


def _ensure_dir(p: Path) -> None:
    """Crea el directorio si no existe."""
    p.mkdir(parents=True, exist_ok=True)


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Devuelve la primera columna existente dentro de candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_teacher_col(df: pd.DataFrame) -> Optional[str]:
    """Detecta columna de docente/profesor."""
    return _pick_first(df, ["cedula_profesor", "docente", "profesor", "teacher", "id_docente"])


def _detect_materia_col(df: pd.DataFrame) -> Optional[str]:
    """Detecta columna de materia/asignatura."""
    return _pick_first(df, ["codigo_materia", "materia", "asignatura", "subject", "id_materia"])


def _detect_score_col(df: pd.DataFrame) -> Optional[str]:
    """Detecta columna score/rating 0..50."""
    # NUEVO (Ruta 2): preferir score_total_0_50 cuando exista.
    # Mantener compatibilidad con datasets antiguos.
    return _pick_first(
        df,
        [
            # Ruta 2 (score total por BETO)
            "score_total_0_50",
            # Alias explícito del score base
            "score_base_0_50",
            # Legacy
            "rating",
            "score_0_50",
            "calificacion",
            "score",
            "score_total",
            "score_base",
        ],
    )


def _build_index(values: pd.Series) -> Dict[str, int]:
    """Mapping estable string->int (ordenado)."""
    uniq = sorted({str(v).strip() for v in values.fillna("").astype(str).tolist() if str(v).strip()})
    return {k: i for i, k in enumerate(uniq)}


def _apply_score_bins(df: pd.DataFrame, score_col: str, cfg: FeaturePackConfig) -> pd.DataFrame:
    """Crea score_0_50, score_q y one-hot score_q_*."""
    out = df.copy()
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).clip(0.0, 50.0)
    out["score_0_50"] = score

    bins = list(cfg.score_bins)
    labels = list(cfg.score_q_labels)

    out["score_q"] = pd.cut(out["score_0_50"], bins=bins, include_lowest=True, right=True, labels=labels)
    out["score_q"] = out["score_q"].astype("Int64").fillna(0)

    for q in labels:
        out[f"score_q_{q}"] = (out["score_q"] == q).astype(int)

    return out

# ---------------------------------------------------------------------------
# Text features (TF-IDF + LSA)
# ---------------------------------------------------------------------------

def _detect_text_col(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
    """Detecta una columna de texto libre (comentarios/opiniones) para features TF-IDF.

    El builder de feature-pack NO asume un nombre fijo de columna.
    Esta función aplica una heurística conservadora basada en nombres comunes.

    Parameters
    ----------
    df:
        DataFrame de entrada (labeled/processed).
    override:
        Si se especifica y existe en df, se usa esta columna.

    Returns
    -------
    Optional[str]
        Nombre de la columna de texto, si se encuentra.
    """
    if override and override in df.columns:
        return str(override)

    # Nombres típicos en datasets académicos / encuestas.
    candidates = [
        'comentario', 'comentarios', 'opinion', 'opiniones', 'texto', 'texto_libre',
        'respuesta', 'respuestas', 'review', 'reviews', 'feedback', 'observacion',
        'observaciones', 'descripcion', 'detalle', 'nota', 'notas',
    ]
    return _pick_first(df, candidates)


def _build_tfidf_lsa_features(
    *,
    text: pd.Series,
    n_components: int = 64,
    min_df: int = 2,
    max_features: int = 20000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Construye features TF-IDF + LSA (TruncatedSVD) de forma determinística.

    El resultado son columnas numéricas estables:

    - feat_t_1 .. feat_t_K

    Donde K puede ser menor que n_components si el dataset es pequeño.

    Notes
    -----
    - Esta transformación se aplica SOLO cuando el caller activa explícitamente
      ``text_feats_mode='tfidf_lsa'``.
    - El código está diseñado para no romper el build: si no hay texto suficiente
      o no se puede construir el vocabulario, retorna un DataFrame vacío y un
      meta explicativo en lugar de lanzar excepción.

    Parameters
    ----------
    text:
        Serie con texto libre. Se normaliza con fillna('').
    n_components:
        Número de componentes LSA deseados (default: 64).
    min_df:
        Frecuencia mínima de documento para incluir un token (default: 2).
    max_features:
        Límite superior del vocabulario TF-IDF (default: 20000).
    random_state:
        Semilla para reproducibilidad del SVD (default: 42).

    Returns
    -------
    (pd.DataFrame, Dict[str, Any])
        DataFrame con columnas feat_t_* y meta con estadísticas del build.
    """
    meta: Dict[str, Any] = {
        'mode': 'tfidf_lsa',
        'enabled': True,
        'n_components_requested': int(n_components),
        'min_df': int(min_df),
        'max_features': int(max_features),
    }

    # Import tardío: el módulo se usa en jobs/routers, pero no queremos
    # hacer hard-fail al importar el paquete si sklearn no está disponible.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
    except Exception as e:  # pragma: no cover
        meta['enabled'] = False
        meta['error'] = f'sklearn no disponible: {e}'
        return pd.DataFrame(), meta

    raw = text.fillna('').astype(str)
    has_text = raw.str.strip().ne('')
    meta['n_rows'] = int(len(raw))
    meta['text_coverage'] = float(has_text.mean()) if len(raw) else 0.0

    # Si hay muy poco texto, evitamos lanzar excepción: no aporta señal útil.
    if int(has_text.sum()) < 2:
        meta['enabled'] = False
        meta['reason'] = 'texto insuficiente para TF-IDF (se requieren >=2 filas con texto)'
        return pd.DataFrame(), meta

    try:
        vec = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True,
        )
        X = vec.fit_transform(raw.tolist())
        vocab_size = int(len(getattr(vec, 'vocabulary_', {}) or {}))
        meta['vocab_size'] = vocab_size
        meta['n_tfidf_features'] = int(X.shape[1])
        if X.shape[1] < 2 or X.shape[0] < 2:
            meta['enabled'] = False
            meta['reason'] = 'matriz TF-IDF demasiado pequeña para SVD'
            return pd.DataFrame(), meta

        # Ajuste seguro: SVD requiere k < min(n_samples, n_features).
        k_max = int(min(X.shape[0] - 1, X.shape[1] - 1))
        k = int(min(max(1, n_components), max(1, k_max)))
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        Z = svd.fit_transform(X)
        meta['n_components'] = int(k)
        try:
            meta['explained_variance_ratio_sum'] = float(np.sum(svd.explained_variance_ratio_))
        except Exception:
            pass

        cols = [f'feat_t_{i}' for i in range(1, k + 1)]
        out = pd.DataFrame(Z.astype('float32'), columns=cols)
        return out, meta
    except Exception as e:
        meta['enabled'] = False
        meta['error'] = str(e)
        return pd.DataFrame(), meta

# ---------------------------------------------------------------------------
# Pair-level features (Ruta 2: score_docente)
# ---------------------------------------------------------------------------

def _series_stats(s: pd.Series) -> Dict[str, float]:
    """Stats defensivos para una serie numérica."""
    if s is None or len(s) == 0:
        return {"min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan"), "mean": float("nan")}
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {"min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(x.min()),
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def _pick_tfidf_cols(df: pd.DataFrame) -> List[str]:
    """Detecta columnas TF-IDF+LSA estilo feat_t_1..N (si existen)."""
    cols = [c for c in df.columns if str(c).startswith("feat_t_")]

    def _key(c: str) -> int:
        try:
            return int(str(c).split("feat_t_", 1)[-1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


def _build_pair_matrix(
    *,
    df: pd.DataFrame,
    dataset_id: str,
    input_uri: str,
    teacher_col: str,
    materia_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Construye pair_matrix (1 fila = 1 par teacher_id-materia_id) + meta."""

    out = df.copy()
    out["teacher_key"] = out[teacher_col].fillna("").astype(str).str.strip()
    out["materia_key"] = out[materia_col].fillna("").astype(str).str.strip()

    if "teacher_id" not in out.columns or "materia_id" not in out.columns:
        raise ValueError("pair_matrix requiere teacher_id y materia_id (feature-pack ids)")

    has_text = "has_text" in out.columns
    has_accept = any(c in out.columns for c in ("accepted_by_teacher", "teacher_accepted", "accepted"))
    accept_col = next((c for c in ("accepted_by_teacher", "teacher_accepted", "accepted") if c in out.columns), None)

    # Fuente del target (Ruta 2)
    if "score_total_0_50" in out.columns:
        target_source_col = "score_total_0_50"
    elif "score_base_0_50" in out.columns:
        target_source_col = "score_base_0_50"
    else:
        target_source_col = score_col if score_col in out.columns else "score_0_50"

    out[target_source_col] = pd.to_numeric(out[target_source_col], errors="coerce")
    if "score_base_0_50" in out.columns:
        out["score_base_0_50"] = pd.to_numeric(out["score_base_0_50"], errors="coerce")
    if "score_total_0_50" in out.columns:
        out["score_total_0_50"] = pd.to_numeric(out["score_total_0_50"], errors="coerce")

    calif_cols = [c for c in out.columns if str(c).startswith("calif_") and str(c).split("_", 1)[-1].isdigit()]
    prob_cols = [c for c in ("p_neg", "p_neu", "p_pos") if c in out.columns]
    if ("sentiment_delta" not in out.columns) and ("p_pos" in out.columns) and ("p_neg" in out.columns):
        out["sentiment_delta"] = pd.to_numeric(out["p_pos"], errors="coerce") - pd.to_numeric(out["p_neg"], errors="coerce")

    sentiment_cols = [c for c in ("p_neg", "p_neu", "p_pos", "sentiment_conf", "sentiment_delta", "sentiment_signal") if c in out.columns]
    tfidf_cols = _pick_tfidf_cols(out)

    group_cols = ["teacher_id", "materia_id", "teacher_key", "materia_key"]

    agg: Dict[str, tuple[str, str]] = {
        "n_par": ("teacher_id", "size"),
        "target_score": (target_source_col, "mean"),
    }

    if "score_base_0_50" in out.columns:
        agg["mean_score_base_0_50"] = ("score_base_0_50", "mean")
        agg["std_score_base_0_50"] = ("score_base_0_50", "std")
    if "score_total_0_50" in out.columns:
        agg["mean_score_total_0_50"] = ("score_total_0_50", "mean")
        agg["std_score_total_0_50"] = ("score_total_0_50", "std")

    for c in calif_cols:
        agg[f"mean_{c}"] = (c, "mean")
        agg[f"std_{c}"] = (c, "std")

    for c in sentiment_cols:
        agg[f"mean_{c}"] = (c, "mean")

    for c in tfidf_cols:
        suf = str(c).split("feat_t_", 1)[-1]
        agg[f"mean_feat_t_{suf}"] = (c, "mean")

    if has_text:
        agg["text_coverage_pair"] = ("has_text", "mean")
    if has_accept and accept_col:
        agg["accept_rate_pair"] = (accept_col, "mean")

    pair = out.groupby(group_cols, dropna=False).agg(**agg).reset_index()

    for c in pair.columns:
        if c.startswith("std_"):
            pair[c] = pd.to_numeric(pair[c], errors="coerce").fillna(0.0)

    docente_counts = out.groupby("teacher_id", dropna=False).size().rename("n_docente").reset_index()
    materia_counts = out.groupby("materia_id", dropna=False).size().rename("n_materia").reset_index()

    pair = pair.merge(docente_counts, on="teacher_id", how="left")
    pair = pair.merge(materia_counts, on="materia_id", how="left")

    def _agg_entity(key: str) -> pd.DataFrame:
        cols_num: List[str] = []
        for c in ("score_total_0_50", "score_base_0_50"):
            if c in out.columns:
                cols_num.append(c)
        cols_num += sentiment_cols
        if has_text:
            cols_num.append("has_text")

        if not cols_num:
            return pd.DataFrame({key: out[key].unique()})

        tmp = out[[key] + cols_num].copy()
        for c in cols_num:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        agg_map = {c: "mean" for c in cols_num}
        ent = tmp.groupby(key, dropna=False).agg(agg_map)
        ent.columns = [f"{key}_mean_{c}" for c in cols_num]
        return ent.reset_index()

    docente_agg = _agg_entity("teacher_id")
    materia_agg = _agg_entity("materia_id")

    pair = pair.merge(docente_agg, on="teacher_id", how="left")
    pair = pair.merge(materia_agg, on="materia_id", how="left")

    pair["teacher_id"] = pd.to_numeric(pair["teacher_id"], errors="coerce").fillna(-1).astype(int)
    pair["materia_id"] = pd.to_numeric(pair["materia_id"], errors="coerce").fillna(-1).astype(int)
    pair["n_par"] = pd.to_numeric(pair["n_par"], errors="coerce").fillna(0).astype(int)
    pair["n_docente"] = pd.to_numeric(pair.get("n_docente"), errors="coerce").fillna(0).astype(int)
    pair["n_materia"] = pd.to_numeric(pair.get("n_materia"), errors="coerce").fillna(0).astype(int)

    # Asegurar columna de trazabilidad temporal para incremental window / split temporal
    if "periodo" not in pair.columns:
        pair = pair.copy()
        pair["periodo"] = str(dataset_id)

    meta: Dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "input_uri": str(input_uri),
        "target_col": str(target_source_col),
        "target_col_feature_pack": str(score_col),
        "tfidf_dims": int(len(tfidf_cols)),
        "has_text": bool(has_text),
        "has_accept": bool(has_accept),
        "has_periodo": True,
        "periodo_col": "periodo",
        "n_pairs": int(len(pair)),
        "n_docentes": int(pair["teacher_id"].nunique(dropna=True)) if "teacher_id" in pair.columns else 0,
        "n_materias": int(pair["materia_id"].nunique(dropna=True)) if "materia_id" in pair.columns else 0,
        "n_par_stats": _series_stats(pair["n_par"]) if "n_par" in pair.columns else {},
        "text_coverage_stats": _series_stats(pair["text_coverage_pair"]) if "text_coverage_pair" in pair.columns else {},
        "columns": pair.columns.tolist(),
        "blocks": {
            "evidence": True,
            "sentiment": bool(len(prob_cols) == 3 or any(c.startswith("mean_sentiment") for c in pair.columns)),
            "tfidf_lsa": bool(len(tfidf_cols) > 0),
            "calif": bool(len(calif_cols) > 0),
            "entity_agg": True,
        },
    }

    return pair, meta



def prepare_feature_pack(
    *,
    base_dir: Path,
    dataset_id: str,
    input_uri: str,
    output_dir: str,
    cfg: FeaturePackConfig = FeaturePackConfig(),
    text_feats_mode: str = 'none',
    text_col: Optional[str] = None,
    text_n_components: int = 64,
    text_min_df: int = 2,
    text_max_features: int = 20000,
    text_random_state: int = 42,
) -> Dict[str, str]:
    """
    Genera feature-pack desde un parquet/csv etiquetado.

    Args:
        base_dir: raíz del proyecto.
        dataset_id: periodo o id lógico.
        input_uri: ruta relativa (ej. 'data/labeled/2024-2_beto.parquet' o 'historico/unificado_labeled.parquet')
        output_dir: ruta relativa/absoluta (ej. 'artifacts/features/2024-2')
        cfg: bins estables.
        text_feats_mode: "none" (default) o "tfidf_lsa" para generar feat_t_* desde texto libre.
        text_col: Nombre de la columna de texto (si None, se intenta detectar por heurística).
        text_n_components: Dimensionalidad máxima de LSA (solo aplica a tfidf_lsa).
        text_min_df: min_df de TF-IDF (solo aplica a tfidf_lsa).
        text_max_features: límite de vocabulario TF-IDF (solo aplica a tfidf_lsa).

    Returns:
        dict con rutas relativas a artefactos generados.
    """
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    _ensure_dir(out_dir)

    inp = (base_dir / input_uri).resolve()
    if not inp.exists():
        raise FileNotFoundError(f"Input no existe: {inp}")

    if inp.suffix.lower() == ".parquet":
        df = pd.read_parquet(inp)
    elif inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
    else:
        raise ValueError(f"Formato no soportado: {inp.suffix}")

    # --- P0: backward compat score_* ---
    labeled_meta = load_sidecar_score_meta(inp, base_dir=base_dir)
    # ensure_score_columns retorna (df, score_col, score_debug)
    df, score_col, score_debug = ensure_score_columns(
        df,
        labeled_meta=labeled_meta,
        prefer_total=True,
        allow_derive=True,
    )


    teacher_col = _detect_teacher_col(df)
    materia_col = _detect_materia_col(df)



    if teacher_col is None:
        raise ValueError("No se detectó columna de docente (ej: cedula_profesor/docente/profesor).")
    if materia_col is None:
        raise ValueError("No se detectó columna de materia (ej: codigo_materia/materia/asignatura).")
    if score_col is None:
        raise ValueError("No se detectó columna score/rating (ej: rating/score_0_50).")

    teacher_index = _build_index(df[teacher_col])
    materia_index = _build_index(df[materia_col])

    df["teacher_id"] = df[teacher_col].fillna("").astype(str).map(lambda x: teacher_index.get(str(x).strip(), -1))
    df["materia_id"] = df[materia_col].fillna("").astype(str).map(lambda x: materia_index.get(str(x).strip(), -1))

    df = _apply_score_bins(df, score_col=score_col, cfg=cfg)

    # -------------------------------------------------------------------
    # Text feats (opcional): TF-IDF + LSA
    # -------------------------------------------------------------------
    # Por diseño, este feature-pack NO genera embeddings automáticamente a
    # menos que se active de forma explícita. Esto evita costos inesperados
    # en pipelines existentes y mantiene compatibilidad con flujos antiguos.

    mode = str(text_feats_mode or 'none').strip().lower()
    if mode not in ('none', 'tfidf_lsa'):
        raise ValueError(f"text_feats_mode inválido: {text_feats_mode!r}. Use 'none' o 'tfidf_lsa'.")

    text_meta: Dict[str, Any] = {
        'mode': mode,
        'enabled': False,
        'source_col': None,
        'n_components_requested': int(text_n_components),
        'min_df': int(text_min_df),
        'max_features': int(text_max_features),
    }

    # Si ya existen feat_t_* (por ejemplo, porque vienen del parquet labeled),
    # no regeneramos para evitar inconsistencias.
    has_existing_text_feats = any(str(c).startswith('feat_t_') for c in df.columns)

    if mode == 'tfidf_lsa' and (not has_existing_text_feats):
        detected_text_col = _detect_text_col(df, override=text_col)
        text_meta['source_col'] = detected_text_col

        if detected_text_col is not None:
            feats_df, meta = _build_tfidf_lsa_features(
                text=df[detected_text_col],
                n_components=int(text_n_components),
                min_df=int(text_min_df),
                max_features=int(text_max_features),
                random_state=int(text_random_state),
            )
            # Unir meta (manteniendo claves estables).
            text_meta.update({k: v for k, v in (meta or {}).items() if k not in ('mode',)})

            if len(feats_df.columns) > 0:
                # Añadir has_text para usarlo a nivel pair-matrix (coverage).
                if 'has_text' not in df.columns:
                    raw = df[detected_text_col].fillna('').astype(str)
                    df['has_text'] = raw.str.strip().ne('').astype(int)

                # Alineación de índices: ambos vienen del mismo df, por lo que
                # el join por posición es seguro.
                df = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
                text_meta['enabled'] = True
                text_meta['n_text_features'] = int(len(feats_df.columns))

    text_feat_cols = [c for c in df.columns if c.startswith("feat_t_")]

    # Probabilidades (si vienen del labeled BETO)
    # Soportamos distintos nombres y normalizamos SIEMPRE a: p_neg / p_neu / p_pos
    prob_triplets = [
        ("p_neg", "p_neu", "p_pos"),
        ("prob_neg", "prob_neu", "prob_pos"),
        ("sent_neg", "sent_neu", "sent_pos"),
        ("neg", "neu", "pos"),
    ]

    lower_to_col = {c.lower(): c for c in df.columns}
    used_triplet = None

    for a, b, c in prob_triplets:
        ra, rb, rc = lower_to_col.get(a.lower()), lower_to_col.get(b.lower()), lower_to_col.get(c.lower())
        if ra and rb and rc:
            used_triplet = (ra, rb, rc)
            break

    prob_cols: list[str] = []
    if used_triplet:
        # Si ya existen p_* no duplicamos; si no, creamos copias normalizadas
        canon = (lower_to_col.get("p_neg"), lower_to_col.get("p_neu"), lower_to_col.get("p_pos"))
        has_canon = all(canon)

        if not has_canon:
            df["p_neg"] = df[used_triplet[0]].astype(float)
            df["p_neu"] = df[used_triplet[1]].astype(float)
            df["p_pos"] = df[used_triplet[2]].astype(float)

        prob_cols = [c for c in ("p_neg", "p_neu", "p_pos") if c in df.columns]

    # Si hay probas, aseguramos confidence y labels derivados (para que RBM no filtre todo a vacío)
    if len(prob_cols) == 3:
        if "sentiment_conf" not in df.columns:
            df["sentiment_conf"] = df[prob_cols].astype(float).max(axis=1)

        # Derivar etiqueta si no existe ninguna etiqueta “hard”
        if not any(c in df.columns for c in ("sentiment_label_teacher", "sentiment_label", "y_sentimiento", "label")):
            lab = (
                df[prob_cols]
                .astype(float)
                .idxmax(axis=1)
                .map({"p_neg": "neg", "p_neu": "neu", "p_pos": "pos"})
                .fillna("")
            )
            df["sentiment_label"] = lab

        # Asegurar y_sentimiento (usado por RBMRestringida)
        if "y_sentimiento" not in df.columns:
            if "sentiment_label_teacher" in df.columns:
                df["y_sentimiento"] = df["sentiment_label_teacher"].astype(str)
            elif "sentiment_label" in df.columns:
                df["y_sentimiento"] = df["sentiment_label"].astype(str)

    sentiment_cols = [c for c in ("p_neg", "p_neu", "p_pos", "sentiment_conf") if c in df.columns]

    one_hot_cols = [c for c in df.columns if c.startswith("score_q_")]

    # Columnas que usan los RBM como features
    calif_cols = [c for c in df.columns if c.startswith("calif_") and c.split("_", 1)[-1].isdigit()]
    pregunta_cols = [c for c in df.columns if c.startswith("pregunta_") and c.split("_", 1)[-1].isdigit()]

    base_cols = [c for c in ["periodo", "teacher_id", "materia_id", "score_0_50", "score_q"] if c in df.columns]
    extra_cols = [
        c for c in [
            "accepted_by_teacher",
            "sentiment_label_teacher",
            "sentiment_label",
            "y_sentimiento",
        ]
        if c in df.columns
    ]

    keep_cols = base_cols + calif_cols + pregunta_cols + extra_cols + sentiment_cols + text_feat_cols + one_hot_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # dedup por si acaso
    train = df[keep_cols].copy()

    train_path = out_dir / "train_matrix.parquet"
    train.to_parquet(train_path, index=False)

    (out_dir / "teacher_index.json").write_text(json.dumps(teacher_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "materia_index.json").write_text(json.dumps(materia_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bins.json").write_text(
        json.dumps({"score_bins": list(cfg.score_bins), "score_q_labels": list(cfg.score_q_labels)}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "input_uri": input_uri,
                "output_dir": str(out_dir),
                "teacher_col": teacher_col,
                "materia_col": materia_col,
                "score_col": score_col,
                "n_rows": int(len(train)),
                "columns": train.columns.tolist(),
                "text_feat_cols": text_feat_cols,
                "text": text_meta,
                "sentiment_cols": sentiment_cols,
                "one_hot_cols": one_hot_cols,
                "score_debug": score_debug,
                "score_source": (score_debug or {}).get("source"),
                "derived_score": bool((score_debug or {}).get("created_columns")),
                "blocks": {
                    "sentiment": bool(sentiment_cols) and ("y_sentimiento" in train.columns),
                    "text_feats": bool(text_feat_cols),
                    "one_hot": bool(one_hot_cols),
                    "pair_matrix": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------
    # Pair-level (Ruta 2): artifacts/features/<dataset_id>/pair_matrix.parquet
    # -------------------------------------------------------------------
    pair_df, pair_meta = _build_pair_matrix(
        df=df,
        dataset_id=dataset_id,
        input_uri=input_uri,
        teacher_col=teacher_col,
        materia_col=materia_col,
        score_col=score_col,
    )

    pair_path = out_dir / "pair_matrix.parquet"
    pair_df.to_parquet(pair_path, index=False)

    (out_dir / "pair_meta.json").write_text(
        json.dumps(pair_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
        except Exception:
            return str(p)

    return {
        "train_matrix": _rel(train_path),
        "teacher_index": _rel(out_dir / "teacher_index.json"),
        "materia_index": _rel(out_dir / "materia_index.json"),
        "bins": _rel(out_dir / "bins.json"),
        "meta": _rel(out_dir / "meta.json"),
        "pair_matrix": _rel(pair_path),
        "pair_meta": _rel(out_dir / "pair_meta.json"),
    }

def _resolve_artifacts_root(*, artifacts_root: Path | None = None) -> Path:
    """Resuelve la raíz de artifacts de forma compatible con el backend.

    Prioridad:
    1) artifacts_root explícito (si se pasa)
    2) neurocampus.utils.paths.artifacts_dir() (respeta NC_ARTIFACTS_DIR)
    3) Path.cwd() / "artifacts" como fallback

    Returns
    -------
    Path
        Ruta absoluta a la carpeta raíz de artifacts.
    """
    if artifacts_root is not None:
        return artifacts_root.expanduser().resolve()

    # Preferir el helper centralizado si existe (P1/P2 ya lo usan en otros módulos).
    try:
        from neurocampus.utils.paths import artifacts_dir as _artifacts_dir  # type: ignore
        return Path(_artifacts_dir()).expanduser().resolve()
    except Exception:
        # Fallback razonable: asumir ejecución desde el root del repo.
        return (Path.cwd() / "artifacts").expanduser().resolve()


def load_feature_pack(
    *,
    dataset_id: str,
    kind: str = "train",
    artifacts_root: Path | None = None,
    load_meta: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Carga un feature pack desde artifacts/features/<dataset_id>.

    Esta función es el complemento "lector" de :func:`prepare_feature_pack`.

    Parameters
    ----------
    dataset_id:
        Identificador del dataset (ej. "2025-1").
    kind:
        - "train": carga ``train_matrix.parquet`` (row-level).
        - "pair":  carga ``pair_matrix.parquet`` (teacher-materia level).
    artifacts_root:
        Override de la raíz de artifacts. Si es None, intenta resolver con:
        `neurocampus.utils.paths.artifacts_dir()` y luego fallback a `./artifacts`.
    load_meta:
        Si True, intenta cargar ``meta.json`` (train) o ``pair_meta.json`` (pair).

    Returns
    -------
    (pd.DataFrame, Optional[dict])
        DataFrame cargado y el meta asociado (si existe y load_meta=True).

    Raises
    ------
    ValueError
        Si dataset_id o kind son inválidos.
    FileNotFoundError
        Si no existe el parquet esperado.
    """
    ds = str(dataset_id or "").strip()
    if not ds:
        raise ValueError("dataset_id vacío")

    k = str(kind or "").strip().lower()
    if k not in ("train", "pair"):
        raise ValueError("kind inválido: use 'train' o 'pair'")

    root = _resolve_artifacts_root(artifacts_root=artifacts_root)
    feat_dir = (root / "features" / ds).resolve()

    if k == "train":
        parquet_path = feat_dir / "train_matrix.parquet"
        meta_path = feat_dir / "meta.json"
    else:
        parquet_path = feat_dir / "pair_matrix.parquet"
        meta_path = feat_dir / "pair_meta.json"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No existe feature pack '{k}' para dataset_id={ds}. "
            f"Falta: {parquet_path}. "
            "Ejecuta primero /modelos/feature-pack/prepare."
        )

    df = pd.read_parquet(parquet_path)

    meta: dict[str, Any] | None = None
    if load_meta and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            # Meta es best-effort: no rompemos por JSON corrupto.
            meta = None

    return df, meta
