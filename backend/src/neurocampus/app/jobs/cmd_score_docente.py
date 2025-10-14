# backend/src/neurocampus/app/jobs/cmd_score_docente.py
# Agrega por (docente, codigo_materia[, periodo]) y calcula % "le irá bien".
# Usa p_pos o etiqueta 'pos' para el conteo; aplica intervalo Jeffreys para robustez.

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import beta  # si no tienes scipy, implemento a mano abajo

def jeffreys_interval(k: int, n: int, alpha: float = 0.05):
    # Beta(0.5,0.5) posterior → intervalo central al 1-alpha
    if n == 0:
        return (0.0, 1.0)
    lo = beta.ppf(alpha/2, k + 0.5, (n - k) + 0.5)
    hi = beta.ppf(1 - alpha/2, k + 0.5, (n - k) + 0.5)
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="Parquet/CSV con columnas: docente, codigo_materia, (opcional periodo), p_pos y/o sentiment_label_teacher")
    ap.add_argument("--out", dest="dst", required=True, help="Ruta parquet/csv de salida")
    ap.add_argument("--group-cols", default="docente,codigo_materia,periodo",
                    help="Columnas para agrupar, separadas por coma. Default: docente,codigo_materia,periodo")
    ap.add_argument("--pos-th", type=float, default=0.55, help="Umbral p_pos para considerar 'pos' (si no hay label binaria).")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha para intervalo (1-alpha de confianza).")
    ap.add_argument("--mix-w", type=float, default=0.4, help="Peso del sentimiento en score combinado (resto calificaciones).")
    args = ap.parse_args()

    # 1) Carga
    df = pd.read_parquet(args.src) if args.src.lower().endswith(".parquet") else pd.read_csv(args.src)

    # 2) Columnas de agrupación existentes
    group_cols = [c for c in [c.strip() for c in args.group_cols.split(",")] if c and c in df.columns]
    if not group_cols:
        raise ValueError("Ninguna de las columnas de agrupación existe en el dataset.")

    # 3) Deriva POS binario
    if "sentiment_label_teacher" in df.columns:
        y_pos = (df["sentiment_label_teacher"].astype(str).str.lower() == "pos").astype(int)
    elif "p_pos" in df.columns:
        y_pos = (df["p_pos"].astype(float) >= args.pos_th).astype(int)
    else:
        raise ValueError("Se requiere sentiment_label_teacher o p_pos para derivar 'pos'.")

    df["_is_pos"] = y_pos

    # 4) Agregaciones numéricas (calif_1..10 si existen)
    calif_cols = [c for c in df.columns if c.startswith("calif_")]
    agg = { "_is_pos": ["sum","count"] }
    for c in calif_cols:
        agg[c] = "mean"
    if "p_pos" in df.columns:
        agg["p_pos"] = "mean"

    g = df.groupby(group_cols, dropna=False).agg(agg)
    # aplanar columnas multiíndice
    g.columns = ["_".join([c] if isinstance(c, str) else [c[0], c[1]]) for c in g.columns.values]
    g = g.reset_index()

    # 5) Métricas de “le irá bien”
    g.rename(columns={"_is_pos_sum":"pos_count", "_is_pos_count":"n"}, inplace=True)
    g["pct_pos"] = (g["pos_count"] / g["n"]).fillna(0.0)

    # Jeffreys CI
    ci = g.apply(lambda r: jeffreys_interval(int(r["pos_count"]), int(r["n"]), alpha=args.alpha), axis=1)
    g["pct_pos_lo"] = [x[0] for x in ci]
    g["pct_pos_hi"] = [x[1] for x in ci]

    # Score combinado (ejemplo 60% calif + 40% sentimiento)
    if calif_cols:
        g["calif_mean"] = g[[f"{c}_mean" for c in calif_cols]].mean(axis=1)
        g["calif_mean_0_1"] = (g["calif_mean"] / 5.0).clip(0,1)
    else:
        g["calif_mean_0_1"] = np.nan

    w = float(args.mix_w)  # peso del sentimiento
    g["score_combinado_0_1"] = (1-w)*g["calif_mean_0_1"].fillna(0.0) + w*g["pct_pos"]
    g["prob_bueno_pct"] = (g["score_combinado_0_1"] * 100).round(1)

    # 6) Guardar
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if args.dst.lower().endswith(".parquet"):
        g.to_parquet(args.dst, index=False)
    else:
        g.to_csv(args.dst, index=False)

    # 7) Resumen consola
    print({
        "out": args.dst,
        "groups": group_cols,
        "n_groups": len(g),
        "cols": list(g.columns)[:12] + (["..."] if len(g.columns) > 12 else []),
    })

if __name__ == "__main__":
    main()
