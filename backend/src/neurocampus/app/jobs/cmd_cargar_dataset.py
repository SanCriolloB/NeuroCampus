# backend/src/neurocampus/app/jobs/cmd_cargar_dataset.py
import argparse
import pandas as pd
import re
from pathlib import Path

TEXT_CANDIDATES = ["comentario","comentarios","observaciones","obs","texto","review","opinion"]

def _try_read_csv(path):
    # Intentos ordenados por probabilidad en Windows/Excel/OneDrive
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
    raise RuntimeError(f"No se pudo leer el CSV con separadores/codificaciones comunes. Último error: {last_err}")

def _find_text_col(cols):
    lc = [c.lower().strip() for c in cols]
    for cand in TEXT_CANDIDATES:
        if cand in lc:
            return cols[lc.index(cand)]
    # fallback: alguna col con nombre que sugiera texto
    for i, name in enumerate(lc):
        if re.search(r"coment|observa|text|review|opini", name):
            return cols[i]
    return None

def _numeric_columns(df):
    nums = []
    for c in df.columns:
        # intenta convertir a numérico sin reventar: si muchas filas se convierten, lo tomamos como numérico
        s = pd.to_numeric(df[c], errors="coerce")
        ratio = s.notna().mean()
        if ratio >= 0.8:  # 80% numéricas → consideramos que es calificación
            nums.append(c)
    return nums

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="Ruta CSV de entrada (p.ej., examples/Evaluacion.csv)")
    ap.add_argument("--out", dest="dst", required=True, help="Ruta parquet de salida (p.ej., data/processed/evaluaciones_2025.parquet)")
    # Mini-ajuste: preservar metadatos como texto
    ap.add_argument("--meta-list", dest="meta_list", default=None,
                    help="Lista separada por coma de columnas a preservar como metadatos (ej: 'codigo_materia,docente,grupo,periodo')")
    args = ap.parse_args()

    df, read_kw = _try_read_csv(args.src)

    # Detectar comentario
    text_col = _find_text_col(df.columns)
    if text_col is None:
        raise ValueError("No se encontró columna de comentario. Renombra en el CSV alguna columna a 'comentario'.")

    # Normalizar texto
    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    df = df[df[text_col].str.len() > 0].copy()

    # Detectar calificaciones numéricas
    num_cols = _numeric_columns(df)
    # evita tomar ID u otros; si tienes una columna id con muy pocos no-nulos, el threshold lo descartará
    # Orden estable por nombre original
    num_cols_sorted = sorted(num_cols, key=lambda x: df.columns.get_loc(x))

    # Construir dataframe estándar
    out = pd.DataFrame()
    out["comentario"] = df[text_col].astype(str)

    # Renombrar a calif_1..N
    for i, c in enumerate(num_cols_sorted, start=1):
        out[f"calif_{i}"] = pd.to_numeric(df[c], errors="coerce")

    # Si existe etiqueta humana, presérvala con nombre estándar
    for cand in ["y","label","sentimiento","y_sentimiento","target"]:
        if cand in df.columns:
            out["y_sentimiento"] = df[cand].astype(str)
            break

    # Mini-ajuste: preservar metadatos (como texto) si se pasan por --meta-list
    meta_kept = []
    if args.meta_list:
        wanted = [c.strip() for c in args.meta_list.split(",") if c.strip()]
        for m in wanted:
            if m in df.columns:
                # Fuerza a texto para evitar que entren como features numéricos
                out[m] = df[m].astype(str)
                meta_kept.append(m)

    # Guardar parquet
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.dst, index=False)

    # Reporte básico
    print({
        "read_kwargs": read_kw,
        "text_col": text_col,
        "n_rows": len(out),
        "n_calif_cols": len(num_cols_sorted),
        "has_y_sentimiento": "y_sentimiento" in out.columns,
        "meta_kept": meta_kept,
        "out_cols": out.columns.tolist()
    })

if __name__ == "__main__":
    main()
