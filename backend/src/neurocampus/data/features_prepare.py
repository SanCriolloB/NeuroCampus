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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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
    return _pick_first(df, ["rating", "score_0_50", "calificacion", "score"])


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

    teacher_col = _detect_teacher_col(df)
    materia_col = _detect_materia_col(df)
    score_col = _detect_score_col(df)

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
    sentiment_cols = [c for c in ["p_neg", "p_neu", "p_pos", "sentiment_conf"] if c in df.columns]
    one_hot_cols = [c for c in df.columns if c.startswith("score_q_")]

    base_cols = [c for c in ["periodo", "teacher_id", "materia_id", "score_0_50", "score_q"] if c in df.columns]
    extra_cols = [c for c in ["accepted_by_teacher", "sentiment_label_teacher"] if c in df.columns]

    keep_cols = base_cols + extra_cols + sentiment_cols + text_feat_cols + one_hot_cols
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
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    }
