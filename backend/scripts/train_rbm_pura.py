#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena la RBM Pura (no supervisada) directamente sobre un dataset tabular.
Guarda artefactos mínimos en artifacts/jobs/<timestamp>.

Uso:
  PYTHONPATH=backend/src python backend/scripts/train_rbm_pura.py --data data/labeled/evaluaciones_2025_teacher.parquet --out artifacts/jobs/rbm_pura_demo
"""
import argparse, os, time, json
import numpy as np
import pandas as pd
import torch

from neurocampus.models.strategies.rbm_pura import RBM

def _load(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet": return pd.read_parquet(path)
    if ext == ".csv":     return pd.read_csv(path)
    if ext in (".xlsx",".xls"): return pd.read_excel(path)
    raise ValueError(f"Extensión no soportada: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_hidden", type=int, default=64)
    ap.add_argument("--cd_k", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = _load(args.data)
    # Selecciona solo columnas numéricas para RBM no supervisada
    X = df.select_dtypes(include=[np.number]).astype(np.float32)

    model = RBM(
        n_hidden=args.n_hidden, cd_k=args.cd_k, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed
    )
    model.fit(X)

    out_dir = args.out or os.path.join("artifacts","jobs", f"rbm_pura_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    # Guardar “representaciones” de 256 filas como muestra
    H = model.transform(X.iloc[:256])
    np.save(os.path.join(out_dir,"H_sample.npy"), H)

    with open(os.path.join(out_dir,"train_meta.json"),"w",encoding="utf-8") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_hidden": int(args.n_hidden),
            "cd_k": int(args.cd_k),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
        }, f, indent=2)

    print(f"[OK] RBM Pura entrenada. Artefactos en: {out_dir}")

if __name__ == "__main__":
    main()
