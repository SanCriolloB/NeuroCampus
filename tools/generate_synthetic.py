#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genera un dataset tabular sintético compatible con el pipeline NeuroCampus.

Columnas:
- calif_1..calif_10  (numéricas, rango 0–5)
- p_neg/p_neu/p_pos  (probabilidades que ~suman 1)
- sentiment_label_teacher (neg/neu/pos) consistente con p_* y media(calif)

Uso:
  python tools/sim/generate_synthetic.py --n 5000 --out data/simulated/evals_sim_5k.parquet
  python tools/sim/generate_synthetic.py --n 3000 --out data/simulated/evals_sim_3k.csv
"""
import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="Filas a generar")
    ap.add_argument("--out", required=True, help="Ruta de salida (.parquet o .csv)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    N = args.n
    # calificaciones base con clusters suaves
    cluster_means = np.random.choice([1.5, 2.5, 3.8], size=N, p=[0.25, 0.35, 0.40])
    X = np.clip(
        np.random.normal(loc=cluster_means[:, None], scale=0.8, size=(N, 10)),
        0,
        5,
    ).astype(np.float32)
    df = pd.DataFrame({f"calif_{i+1}": X[:, i] for i in range(10)})

    # probas suaves con sesgo según media de califs
    avg = X.mean(axis=1)
    p_pos = 1 / (1 + np.exp(-(avg - 2.5)))   # mayor media -> más pos
    p_neg = 1 - p_pos
    p_neu = 0.25 * np.ones_like(p_neg)
    s = p_neg + p_neu + p_pos
    p_neg, p_neu, p_pos = p_neg / s, p_neu / s, p_pos / s

    df["p_neg"] = p_neg.astype(np.float32)
    df["p_neu"] = p_neu.astype(np.float32)
    df["p_pos"] = p_pos.astype(np.float32)

    # label teacher consistente con p_* + un poco de ruido
    idx_max = np.stack([p_neg, p_neu, p_pos], axis=1).argmax(axis=1)
    labels = np.array(["neg", "neu", "pos"])[idx_max]
    flip = np.random.rand(N) < 0.05
    flip_to = np.random.choice(["neg", "neu", "pos"], size=N)
    labels = np.where(flip, flip_to, labels)
    df["sentiment_label_teacher"] = labels

    # escribir
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ext = os.path.splitext(args.out)[1].lower()
    if ext == ".parquet":
        df.to_parquet(args.out, index=False)
    elif ext == ".csv":
        df.to_csv(args.out, index=False)
    else:
        raise ValueError("Usa .parquet o .csv en --out")

    print(f"[OK] Dataset sintético escrito en: {args.out} (N={N})")


if __name__ == "__main__":
    main()
