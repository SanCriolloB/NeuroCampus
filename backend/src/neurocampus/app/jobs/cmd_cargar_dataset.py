# backend/src/neurocampus/app/jobs/cmd_cargar_dataset.py
import argparse
import pandas as pd
import re, unicodedata
from pathlib import Path

TEXT_CANDIDATES = ["comentario","comentarios","observaciones","obs","texto","review","opinion"]

EXCLUDE_NAME_PATTERNS = [
    r"\bid\b", r"\bcod(igo)?\b", r"\bgrupo\b", r"\bmateria\b", r"\basignatura\b",
    r"\bdocumento\b", r"\bidentificaci(o|ó)n\b", r"\bsemestre\b", r"\bperiodo\b",
    r"\ba(ñ|n)o\b", r"\bfecha\b", r"\bedad\b", r"\b(telefono|tel|celular)\b",
    r"\bcorreo\b", r"\bemail\b", r"\bdni\b", r"\b(nit|rut)\b"
]

def _normalize(s: str) -> str:
    """lower, remove accents, collapse spaces/hyphens/underscores -> '_'"""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)       # space or hyphen -> underscore
    s = re.sub(r"_+", "_", s).strip("_") # collapse duplicates
    return s

def _try_read_csv(path):
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
    norm = {c: _normalize(c) for c in cols}
    inv = {}
    for k, v in norm.items():
        # si hay colisiones, respetamos la primera
        inv.setdefault(v, k)
    for cand in TEXT_CANDIDATES:
        if cand in inv:
            return inv[cand]
    # fallback por regex normalizada
    for c, v in norm.items():
        if re.search(r"(coment|observa|text|review|opini)", v):
            return c
    return None

def _is_excluded(name_norm: str) -> bool:
    return any(re.search(p, name_norm) for p in EXCLUDE_NAME_PATTERNS)

def _within_scale(series: pd.Series, lo=0.0, hi=5.0, min_ratio=0.8) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    ok = s.between(lo, hi).mean()
    return ok >= min_ratio

def _select_calif_cols(df: pd.DataFrame, args) -> list[str]:
    cols = list(df.columns)
    norm_map = {_normalize(c): c for c in cols}

    # 1) Lista explícita (normalizada)
    if args.calif_list:
        raw = [x.strip() for x in args.calif_list.split(",") if x.strip()]
        chosen = []
        for col in raw:
            key = _normalize(col)
            if key not in norm_map:
                raise ValueError(f"Columna '{col}' no existe en el CSV.")
            chosen.append(norm_map[key])
        return chosen

    # 2) Prefijo + N (normalizado). Ej: --calif-prefix pregunta --calif-n 10
    if args.calif_prefix:
        pref = _normalize(args.calif_prefix)
        chosen = []
        for i in range(1, args.calif_n + 1):
            want = f"{pref}_{i}"
            # acepta variantes con/ sin guion/espacio porque todo se normaliza a '_'
            if want in norm_map:
                chosen.append(norm_map[want])
            else:
                # por si hay pequeñas variaciones
                candidates = [orig for norm, orig in norm_map.items()
                              if norm == want or re.fullmatch(rf"{pref}_{i}", norm)]
                if not candidates:
                    raise ValueError(f"No se encontró columna para '{args.calif_prefix}{i}'.")
                chosen.append(candidates[0])
        return chosen

    # 3) Auto: patrones comunes normalizados (pregunta_1..10, p1..10, item_1..10, calif_1..10, nota_1..10)
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
            m = re.fullmatch(pat, norm)
            if m:
                # extraer número
                num = int(re.search(r"(?:[1-9]|10)$", norm).group(0))
                matched.append((num, orig))
        if len(matched) >= 5:
            matched.sort(key=lambda t: t[0])
            return [orig for _, orig in matched[:args.calif_n]]

    # 4) Fallback: numéricas 0-5 y NO excluidas
    candidates = []
    for c in cols:
        n = _normalize(c)
        if _is_excluded(n):
            continue
        if _within_scale(df[c], lo=0, hi=5, min_ratio=0.8):
            candidates.append(c)
    if len(candidates) >= args.calif_n:
        return candidates[:args.calif_n]

    # 5) Último recurso: escala 0-100
    candidates = []
    for c in cols:
        n = _normalize(c)
        if _is_excluded(n):
            continue
        if _within_scale(df[c], lo=0, hi=100, min_ratio=0.8):
            candidates.append(c)
    if len(candidates) >= args.calif_n:
        return candidates[:args.calif_n]

    raise ValueError("No se pudieron identificar columnas de calificación válidas. "
                     "Usa --calif-prefix pregunta --calif-n 10 o --calif-list 'pregunta 1,...,pregunta 10'.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="Ruta CSV/parquet de entrada (p.ej., examples/Evaluacion.csv)")
    ap.add_argument("--out", dest="dst", required=True,
                    help="Ruta parquet de salida (p.ej., data/processed/evaluaciones_2025.parquet)")
    # Selección de calificaciones
    ap.add_argument("--calif-prefix", dest="calif_prefix", default=None,
                    help="Prefijo de preguntas (ej: 'pregunta' detecta 'pregunta 1'/'pregunta_1').")
    ap.add_argument("--calif-n", dest="calif_n", type=int, default=10,
                    help="Número de preguntas a tomar (default: 10).")
    ap.add_argument("--calif-list", dest="calif_list", default=None,
                    help="Lista separada por coma con nombres EXACTOS (se normalizan) en orden.")
    # Metadatos a preservar
    ap.add_argument("--meta-list", dest="meta_list", default=None,
                    help="Columnas meta a conservar como texto (se normalizan nombres).")
    args = ap.parse_args()

    # lee CSV o parquet
    if args.src.lower().endswith(".parquet"):
        df = pd.read_parquet(args.src)
        read_kw = {"format": "parquet"}
    else:
        df, read_kw = _try_read_csv(args.src)

    # detectar columna de texto
    text_col = _find_text_col(df.columns)
    if text_col is None:
        raise ValueError("No se encontró columna de comentario (ej: 'comentario'/'observaciones').")

    # normalizar texto y filtrar vacíos
    s = df[text_col].fillna("").astype(str).str.strip()
    s_lower = s.str.lower()
    s = s.mask(s_lower.isin({"nan", "none", "null"}), "")
    df[text_col] = s
    # NO filtrar; solo marcar
    df["has_text"] = (df[text_col].str.len() > 0).astype(int)

    # reset index para evitar alineación
    df = df.reset_index(drop=True)

    # seleccionar columnas de calificación
    califs = _select_calif_cols(df, args)
    if len(califs) > args.calif_n:
        califs = califs[:args.calif_n]

    # construir salida estándar
    out = pd.DataFrame()
    out["comentario"] = df[text_col].astype(str).to_numpy()
    out["has_text"] = df["has_text"].astype(int).to_numpy()
    
    for i, c in enumerate(califs, start=1):
        out[f"calif_{i}"] = pd.to_numeric(df[c], errors="coerce").to_numpy()

    # etiqueta humana (si existiera)
    for cand in ["y","label","sentimiento","y_sentimiento","target"]:
        if cand in df.columns:
            out["y_sentimiento"] = df[cand].astype(str)
            break

    # preservar metadatos (mapeo normalizado)
    meta_kept = []
    if args.meta_list:
        want = [w.strip() for w in args.meta_list.split(",") if w.strip()]
        norm_map = {_normalize(c): c for c in df.columns}
        for m in want:
            key = _normalize(m)
            if key in norm_map:
                col_orig = norm_map[key]
                out[m] = df[col_orig].astype(str).to_numpy()  # conserva con el nombre solicitado
                meta_kept.append(m)
    # preservar metadatos (mapeo normalizado)
    meta_kept = []
    want = []

    # 1) metadatos por defecto necesarios para la UI (Fase 4)
    default_meta = [
        "id", "profesor", "materia", "asignatura",
        "codigo_materia", "grupo", "cedula_profesor",
    ]
    want.extend(default_meta)

    # 2) metadatos solicitados por parámetro (si vienen)
    if args.meta_list:
        want.extend([w.strip() for w in args.meta_list.split(",") if w.strip()])

    # única lista sin duplicados manteniendo orden
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
        "read_kwargs": read_kw,
        "text_col": text_col,
        "n_rows": int(len(out)),
        "calif_from": califs,
        "meta_kept": meta_kept,
        "out_cols": out.columns.tolist()
    })

if __name__ == "__main__":
    main()
