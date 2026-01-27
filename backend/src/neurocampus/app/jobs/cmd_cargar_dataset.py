# backend/src/neurocampus/app/jobs/cmd_cargar_dataset.py
"""neurocampus.app.jobs.cmd_cargar_dataset

Normaliza un dataset crudo (CSV/Parquet) a un layout estándar que alimenta el
pipeline de **Datos**.

Salidas esperadas (por convención del proyecto)
----------------------------------------------
- data/processed/<dataset_id>.parquet

Garantías principales
---------------------
- Columnas de calificación normalizadas como: calif_1..calif_N
- Escala numérica normalizada a **0..50** (clamp defensivo)
- Columna `periodo` por fila (si no venía, se infiere)
- Columna `comentario` (texto original) y `has_text` (bandera 0/1)

Notas
-----
- Este job no ejecuta PLN. Solo prepara el dataset para que luego
  `cmd_preprocesar_beto.py` pueda etiquetar y/o generar embeddings.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

TEXT_CANDIDATES = ["comentario", "comentarios", "observaciones", "obs", "texto", "review", "opinion"]

EXCLUDE_NAME_PATTERNS = [
    r"\bid\b",
    r"\bcod(igo)?\b",
    r"\bgrupo\b",
    r"\bmateria\b",
    r"\basignatura\b",
    r"\bdocumento\b",
    r"\bidentificaci(o|ó)n\b",
    r"\bsemestre\b",
    r"\bperiodo\b",
    r"\ba(ñ|n)o\b",
    r"\bfecha\b",
    r"\bedad\b",
    r"\b(telefono|tel|celular)\b",
    r"\bcorreo\b",
    r"\bemail\b",
    r"\bdni\b",
    r"\b(nit|rut)\b",
]


def _normalize(s: str) -> str:
    """lower, remove accents, collapse spaces/hyphens/underscores -> '_'"""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)  # space or hyphen -> underscore
    s = re.sub(r"_+", "_", s).strip("_")  # collapse duplicates
    return s


def _try_read_csv(path: str):
    """Intenta múltiples separadores/encodings para maximizar compatibilidad."""
    attempts = [
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep=";", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8"),
        dict(sep=";", encoding="utf-8"),
        dict(sep="\t", encoding="utf-8-sig"),
        dict(sep="|", encoding="utf-8-sig"),
        dict(sep=",", encoding="latin1"),
        dict(sep=";", encoding="latin1"),
    ]
    last_err = None
    for kw in attempts:
        try:
            df = pd.read_csv(path, **kw)
            return df, kw
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No se pudo leer el CSV. Último error: {last_err}")


def _find_text_col(cols):
    """Encuentra la columna que contiene el texto/comentario."""
    norm = {c: _normalize(c) for c in cols}
    inv = {}
    for k, v in norm.items():
        inv.setdefault(v, k)

    for cand in TEXT_CANDIDATES:
        if cand in inv:
            return inv[cand]

    for c, v in norm.items():
        if re.search(r"(coment|observa|text|review|opini)", v):
            return c
    return None


def _is_excluded(name_norm: str) -> bool:
    """True si el nombre de columna parece metadato, no calificación."""
    return any(re.search(p, name_norm) for p in EXCLUDE_NAME_PATTERNS)


def _within_scale(series: pd.Series, lo=0.0, hi=5.0, min_ratio=0.8) -> bool:
    """Heurística: % de valores en [lo,hi]."""
    s = pd.to_numeric(series, errors="coerce")
    ok = s.between(lo, hi).mean()
    return ok >= min_ratio


def _select_calif_cols(df: pd.DataFrame, args) -> list[str]:
    """Selecciona columnas de calificación (pregunta_1..N, p1..N, etc.)."""
    cols = list(df.columns)
    norm_map = {_normalize(c): c for c in cols}

    if args.calif_list:
        raw = [x.strip() for x in args.calif_list.split(",") if x.strip()]
        chosen = []
        for col in raw:
            key = _normalize(col)
            if key not in norm_map:
                raise ValueError(f"Columna '{col}' no existe en el CSV.")
            chosen.append(norm_map[key])
        return chosen

    if args.calif_prefix:
        pref = _normalize(args.calif_prefix)
        chosen = []
        for i in range(1, args.calif_n + 1):
            want = f"{pref}_{i}"
            if want in norm_map:
                chosen.append(norm_map[want])
            else:
                candidates = [orig for norm, orig in norm_map.items() if norm == want]
                if not candidates:
                    raise ValueError(f"No se encontró columna para '{args.calif_prefix}{i}'.")
                chosen.append(candidates[0])
        return chosen

    patterns = [
        (r"^pregunta_(?:[1-9]|10)$", 10),
        (r"^p(?:[1-9]|10)$", 10),
        (r"^item_(?:[1-9]|10)$", 10),
        (r"^calif_(?:[1-9]|10)$", 10),
        (r"^nota_(?:[1-9]|10)$", 10),
    ]
    norm_cols = {_normalize(c): c for c in cols}
    for pat, _ in patterns:
        matched = []
        for norm, orig in norm_cols.items():
            if re.fullmatch(pat, norm):
                num = int(re.search(r"(?:[1-9]|10)$", norm).group(0))
                matched.append((num, orig))
        if len(matched) >= 5:
            matched.sort(key=lambda t: t[0])
            return [orig for _, orig in matched[: args.calif_n]]

    candidates = []
    for c in cols:
        n = _normalize(c)
        if _is_excluded(n):
            continue
        if _within_scale(df[c], lo=0, hi=5, min_ratio=0.8):
            candidates.append(c)
    if len(candidates) >= args.calif_n:
        return candidates[: args.calif_n]

    candidates = []
    for c in cols:
        n = _normalize(c)
        if _is_excluded(n):
            continue
        if _within_scale(df[c], lo=0, hi=100, min_ratio=0.8):
            candidates.append(c)
    if len(candidates) >= args.calif_n:
        return candidates[: args.calif_n]

    raise ValueError(
        "No se pudieron identificar columnas de calificación válidas. "
        "Usa --calif-prefix pregunta --calif-n 10 o --calif-list 'pregunta 1,...,pregunta 10'."
    )


def _infer_dataset_id(src: str, dst: str, dataset_id_arg: Optional[str]) -> str:
    """Infere un dataset_id/periodo estable.

    Orden:
    1) --dataset-id (si es válido)
    2) stem del archivo de salida (data/processed/<stem>.parquet)
    3) stem del archivo de entrada

    Nota:
    - Algunos frontends/envíos pueden mandar `"None"`, `"null"` o `"nan"` como string.
      Esos valores NO se consideran dataset_id válidos.
    """

    def _is_valid(x: Optional[str]) -> bool:
        if x is None:
            return False
        s = str(x).strip()
        if not s:
            return False
        return s.lower() not in {"none", "null", "nan", "<na>"}

    if _is_valid(dataset_id_arg):
        return str(dataset_id_arg).strip()

    dst_stem = Path(dst).stem
    if _is_valid(dst_stem):
        return dst_stem

    src_stem = Path(src).stem
    if _is_valid(src_stem):
        return src_stem

    raise ValueError("No se pudo inferir un dataset_id válido (periodo).")



def _scale_to_0_50(s: pd.Series) -> pd.Series:
    """Normaliza una serie numérica a escala 0..50 (con clamp defensivo)."""
    x = pd.to_numeric(s, errors="coerce")
    vmax = float(x.max(skipna=True) or 0.0)

    if vmax <= 5.5:
        y = x * 10.0
    elif vmax <= 50.0:
        y = x
    elif vmax <= 100.0:
        y = x / 2.0
    else:
        y = (x / vmax * 50.0) if vmax > 0 else x

    return y.clip(lower=0.0, upper=50.0)


def main() -> None:
    """CLI entrypoint."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="Ruta CSV/parquet de entrada.")
    ap.add_argument("--out", dest="dst", required=True, help="Ruta parquet de salida.")
    ap.add_argument("--dataset-id", dest="dataset_id", default=None, help="Dataset logical id (periodo).")

    ap.add_argument("--calif-prefix", dest="calif_prefix", default=None)
    ap.add_argument("--calif-n", dest="calif_n", type=int, default=10)
    ap.add_argument("--calif-list", dest="calif_list", default=None)

    ap.add_argument("--meta-list", dest="meta_list", default=None)
    args = ap.parse_args()

    dataset_id = _infer_dataset_id(args.src, args.dst, args.dataset_id)

    if args.src.lower().endswith(".parquet"):
        df = pd.read_parquet(args.src)
        read_kw = {"format": "parquet"}
    else:
        df, read_kw = _try_read_csv(args.src)

    text_col = _find_text_col(df.columns)
    if text_col is None:
        raise ValueError("No se encontró columna de comentario (ej: 'comentario'/'observaciones').")

    s = df[text_col].fillna("").astype(str).str.strip()
    s_lower = s.str.lower()
    s = s.mask(s_lower.isin({"nan", "none", "null"}), "")
    df[text_col] = s
    df["has_text"] = (df[text_col].str.len() > 0).astype(int)

    df = df.reset_index(drop=True)

    califs = _select_calif_cols(df, args)
    if len(califs) > args.calif_n:
        califs = califs[: args.calif_n]

    out = pd.DataFrame()
    out["periodo"] = str(dataset_id)
    out["comentario"] = df[text_col].astype(str).to_numpy()
    out["has_text"] = df["has_text"].astype(int).to_numpy()

    for i, c in enumerate(califs, start=1):
        out[f"calif_{i}"] = _scale_to_0_50(df[c])

    calif_out = [c for c in out.columns if c.startswith("calif_")]
    if calif_out:
        out["rating"] = out[calif_out].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    for cand in ["y", "label", "sentimiento", "y_sentimiento", "target"]:
        if cand in df.columns:
            out["y_sentimiento"] = df[cand].astype(str)
            break

    meta_kept = []
    default_meta = [
        "id", "profesor", "docente", "teacher",
        "materia", "asignatura", "subject",
        "codigo_materia", "grupo", "cedula_profesor",
    ]
    want = list(default_meta)
    if args.meta_list:
        want.extend([w.strip() for w in args.meta_list.split(",") if w.strip()])

    seen = set()
    want = [x for x in want if not (x.lower() in seen or seen.add(x.lower()))]

    norm_map = {_normalize(c): c for c in df.columns}
    for m in want:
        key = _normalize(m)
        if key in norm_map:
            col_orig = norm_map[key]
            out[m] = df[col_orig].astype(str)
            meta_kept.append(m)

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.dst, index=False)

    print({
        "dataset_id": dataset_id,
        "read_kwargs": read_kw,
        "text_col": text_col,
        "n_rows": int(len(out)),
        "calif_from": califs,
        "meta_kept": meta_kept,
        "out_cols": out.columns.tolist(),
    })


if __name__ == "__main__":
    main()
