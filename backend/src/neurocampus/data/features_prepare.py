# backend/src/neurocampus/data/features_prepare.py
"""neurocampus.data.features_prepare

Este módulo construye el **feature-pack** persistente para el entrenamiento de modelos.

Motivación
----------
En NeuroCampus, el dataset etiquetado (BETO/teacher) debe permanecer lo más "crudo"
posible: probabilidades, flags de aceptación y (opcional) embeddings de texto.
Los artefactos derivados como one-hot/hashing, bins e índices deben vivir en un
directorio de *features* por dataset_id para que:

- sean reproducibles;
- no "contaminen" los datos etiquetados;
- puedan versionarse/rehacerse sin reprocesar BETO;
- el entrenamiento consuma una matriz lista para modelos.

Artefactos esperados
--------------------
Dentro de ``artifacts/features/<dataset_id>/``:

- ``train_matrix.parquet``
- ``teacher_index.json``
- ``materia_index.json``
- ``bins.json``

Este módulo está pensado para ser invocado desde un job (/jobs/data/features/prepare)
o desde un CLI propio en el futuro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[4]  # <repo_root>
_WS_RE = re.compile(r"\s+")


def _ensure_dir(p: Path) -> Path:
    """Crea el directorio (parents=True) si no existe y devuelve el mismo Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_key(value: Any) -> str:
    """Normaliza una llave textual para crear índices estables.

    - Convierte a string.
    - Quita acentos.
    - Baja a minúsculas.
    - Colapsa espacios.
    - Devuelve "" si es nulo.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def _build_index(values: Sequence[str]) -> Dict[str, int]:
    """Construye un mapping valor->id con orden determinista."""
    uniq = sorted({v for v in values if v != ""})
    return {v: i for i, v in enumerate(uniq)}


def _hash_bucket(s: str, dim: int) -> int:
    """Hash estable (md5) para asignar un string a un bucket [0, dim)."""
    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h, 16) % dim


def _encode_onehot_or_hash(
    series: pd.Series,
    *,
    prefix: str,
    max_onehot: int = 200,
    hash_dim: int = 128,
    include_na: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Codifica una serie categórica.

    Estrategia:
    - Si la cardinalidad <= max_onehot: one-hot con ``pd.get_dummies``.
    - Si no: feature hashing a ``hash_dim`` columnas.

    Devuelve (df_features, meta)
    """
    s = series.fillna("").astype(str)
    uniq = s.nunique(dropna=False)

    if uniq <= max_onehot:
        d = pd.get_dummies(
            s if include_na else s.replace({"": np.nan}),
            prefix=prefix,
            dummy_na=include_na,
        )
        meta = {"kind": "onehot", "unique": int(uniq), "n_features": int(d.shape[1])}
        return d.astype(np.float32), meta

    # Hashing
    X = np.zeros((len(s), hash_dim), dtype=np.float32)
    for i, v in enumerate(s.tolist()):
        if not v and not include_na:
            continue
        b = _hash_bucket(v or "<NA>", hash_dim)
        X[i, b] = 1.0

    cols = [f"{prefix}_h_{i}" for i in range(hash_dim)]
    meta = {"kind": "hash", "unique": int(uniq), "hash_dim": int(hash_dim), "n_features": int(hash_dim)}
    return pd.DataFrame(X, columns=cols), meta


def _find_calif_cols(df: pd.DataFrame, n: int = 10) -> List[str]:
    """Identifica columnas de calificación (calif_*, pregunta_*, p*)."""
    for base in ("calif_", "pregunta_"):
        cols = [f"{base}{i}" for i in range(1, n + 1)]
        if all(c in df.columns for c in cols):
            return cols

    cols = [f"p{i}" for i in range(1, n + 1)]
    if all(c in df.columns for c in cols):
        return cols

    pat = re.compile(r"^(calif|pregunta)_?(\d+)$", re.IGNORECASE)
    candidates: List[Tuple[int, str]] = []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            num = int(m.group(2))
            if 1 <= num <= n:
                candidates.append((num, c))
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return [c for _, c in candidates[:n]]

    return []


def _scale_to_50(x: pd.Series) -> pd.Series:
    """Asegura escala 0..50 (si max<=5.5 asume 0..5 y multiplica por 10)."""
    s = pd.to_numeric(x, errors="coerce")
    mx = float(np.nanmax(s.to_numpy())) if len(s) else float("nan")
    if np.isfinite(mx) and mx <= 5.5:
        return s * 10.0
    return s


def _compute_bins(
    df: pd.DataFrame,
    cols: List[str],
    *,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Crea bins por columna (quantiles) y devuelve (df_bins, bins_json)."""
    bins_json: Dict[str, Any] = {"n_bins": n_bins, "cols": {}}
    out = pd.DataFrame(index=df.index)

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0][: n_bins + 1]
        else:
            try:
                qs = np.linspace(0, 1, n_bins + 1)
                edges = np.unique(np.quantile(valid.to_numpy(), qs)).tolist()
                if len(edges) < n_bins + 1:
                    edges = np.linspace(float(valid.min()), float(valid.max()), n_bins + 1).tolist()
            except Exception:
                edges = np.linspace(float(valid.min()), float(valid.max()), n_bins + 1).tolist()

        edges = [float(e) for e in edges]
        if len(edges) < 2 or not all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)):
            edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0][: n_bins + 1]

        out[f"{c}_bin"] = pd.cut(s, bins=edges, labels=False, include_lowest=True).astype("Int64")
        bins_json["cols"][c] = {"edges": edges}

    return out, bins_json


@dataclass(frozen=True)
class FeaturePackResult:
    """Resultado de la preparación de feature-pack (para meta de jobs/UI)."""

    dataset_id: str
    out_dir: Path
    train_matrix_path: Path
    n_rows: int
    n_features: int
    teacher_encoding: Dict[str, Any]
    materia_encoding: Dict[str, Any]
    bins: Dict[str, Any]


def prepare_feature_pack(
    *,
    dataset_id: str,
    labeled_path: Path,
    out_dir: Optional[Path] = None,
    max_onehot: int = 200,
    teacher_hash_dim: int = 128,
    materia_hash_dim: int = 128,
    include_bins: bool = True,
    n_bins: int = 5,
    use_accepted_only: bool = False,
) -> FeaturePackResult:
    """Construye y persiste la matriz de entrenamiento (feature-pack).

    Parameters
    ----------
    dataset_id:
        Identificador lógico del dataset (p.ej., "2025-1" o "historico").

    labeled_path:
        Ruta al parquet etiquetado (BETO). Ej:
        - data/labeled/<periodo>_beto.parquet
        - historico/unificado_labeled.parquet

    use_accepted_only:
        Si True, filtra a filas con ``accepted_by_teacher==1`` y ``has_text==1``.
    """
    if out_dir is None:
        out_dir = BASE_DIR / "artifacts" / "features" / str(dataset_id)
    out_dir = _ensure_dir(out_dir)

    if not labeled_path.exists():
        raise FileNotFoundError(f"No existe labeled_path: {labeled_path}")

    df = pd.read_parquet(labeled_path)
    if df.empty:
        raise ValueError(f"El dataset etiquetado está vacío: {labeled_path}")

    calif_cols = _find_calif_cols(df, n=10)
    if not calif_cols:
        raise ValueError(
            "No se encontraron columnas de calificación (calif_1..10 o pregunta_1..10)."
        )

    for c in calif_cols:
        df[c] = _scale_to_50(df[c])

    df["score_q"] = df[calif_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    if "has_text" not in df.columns:
        df["has_text"] = 0
    if "accepted_by_teacher" not in df.columns:
        df["accepted_by_teacher"] = 0

    df["has_text"] = pd.to_numeric(df["has_text"], errors="coerce").fillna(0).astype(int)
    df["accepted_by_teacher"] = pd.to_numeric(df["accepted_by_teacher"], errors="coerce").fillna(0).astype(int)

    if use_accepted_only:
        df = df[(df["has_text"] == 1) & (df["accepted_by_teacher"] == 1)].copy()

    teacher_col = next((c for c in ("docente", "profesor", "teacher") if c in df.columns), None)
    materia_col = next((c for c in ("materia", "asignatura", "subject", "codigo_materia") if c in df.columns), None)

    df["teacher_key"] = df[teacher_col].map(normalize_key) if teacher_col else ""
    df["materia_key"] = df[materia_col].map(normalize_key) if materia_col else ""

    teacher_index = _build_index(df["teacher_key"].tolist())
    materia_index = _build_index(df["materia_key"].tolist())
    df["teacher_id"] = df["teacher_key"].map(lambda v: teacher_index.get(v, -1)).astype(int)
    df["materia_id"] = df["materia_key"].map(lambda v: materia_index.get(v, -1)).astype(int)

    numeric_parts: List[pd.DataFrame] = []

    base_numeric = df[calif_cols + ["score_q"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_parts.append(base_numeric.astype(np.float32))

    p_cols = [c for c in ("p_neg", "p_neu", "p_pos") if c in df.columns]
    if p_cols:
        numeric_parts.append(df[p_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32))

    embed_cols = sorted(
        [c for c in df.columns if isinstance(c, str) and c.startswith("feat_t_")],
        key=lambda c: int(re.findall(r"\d+", c)[0]) if re.findall(r"\d+", c) else 10**9,
    )
    if embed_cols:
        numeric_parts.append(df[embed_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32))

    teacher_feats, teacher_meta = _encode_onehot_or_hash(
        df["teacher_key"], prefix="teacher", max_onehot=max_onehot, hash_dim=teacher_hash_dim, include_na=True
    )
    materia_feats, materia_meta = _encode_onehot_or_hash(
        df["materia_key"], prefix="materia", max_onehot=max_onehot, hash_dim=materia_hash_dim, include_na=True
    )

    bins_json: Dict[str, Any] = {"enabled": False}
    bins_df = pd.DataFrame(index=df.index)
    if include_bins:
        bins_df, bins_json_cols = _compute_bins(df, calif_cols, n_bins=n_bins)
        bins_json = {"enabled": True, **bins_json_cols}

    id_cols = [c for c in ("id", "periodo", "codigo_materia", "grupo", "cedula_profesor") if c in df.columns]
    id_part = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    id_part["teacher_id"] = df["teacher_id"].astype(int)
    id_part["materia_id"] = df["materia_id"].astype(int)
    id_part["has_text"] = df["has_text"].astype(int)
    id_part["accepted_by_teacher"] = df["accepted_by_teacher"].astype(int)

    X = pd.concat([id_part] + numeric_parts + [teacher_feats, materia_feats, bins_df], axis=1, copy=False)

    train_matrix_path = out_dir / "train_matrix.parquet"
    X.to_parquet(train_matrix_path, index=False)

    (out_dir / "teacher_index.json").write_text(json.dumps(teacher_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "materia_index.json").write_text(json.dumps(materia_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bins.json").write_text(json.dumps(bins_json, ensure_ascii=False, indent=2), encod
