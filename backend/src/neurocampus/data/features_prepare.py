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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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
) -> Dict[str, str]:
    """
    Genera feature-pack desde un parquet/csv etiquetado.

    Args:
        base_dir: raíz del proyecto.
        dataset_id: periodo o id lógico.
        input_uri: ruta relativa (ej. 'data/labeled/2024-2_beto.parquet' o 'historico/unificado_labeled.parquet')
        output_dir: ruta relativa/absoluta (ej. 'artifacts/features/2024-2')
        cfg: bins estables.

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
