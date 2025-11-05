#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena la RBM General (bipartita + cabeza supervisada mínima) con (X,y).
Guarda artefactos (rbm.pt, head.pt, vectorizer.json, job_meta.json).

Uso:
  PYTHONPATH=backend/src python backend/scripts/train_rbm_general.py --data data/labeled/evaluaciones_2025_teacher.parquet --target sentiment_label_teacher --out artifacts/jobs/rbm_general_demo
"""
import argparse, os, time, json
import numpy as np
import pandas as pd
import torch

from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral

def _load(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet": return pd.read_parquet(path)
    if ext == ".csv":     return pd.read_csv(path)
    if ext in (".xlsx",".xls"): return pd.read_excel(path)
    raise ValueError(f"Extensión no soportada: {ext}")

def _pick_target(df: pd.DataFrame, target: str|None) -> str:
    if target and target in df.columns: return target
    for c in ["sentiment_label_teacher","y","label","target","y_sentimiento"]:
        if c in df.columns: return c
    raise ValueError("No se encontró columna de target.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default=None)

    ap.add_argument("--n_hidden", type=int, default=64)
    ap.add_argument("--cd_k", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epochs_rbm", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_rbm", type=float, default=1e-2)
    ap.add_argument("--lr_head", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--scale_mode", default="minmax", choices=["minmax","scale_0_5"])
    ap.add_argument("--use_text_probs", action="store_true")
    ap.add_argument("--use_text_embeds", action="store_true")
    ap.add_argument("--max_calif", type=int, default=10)
    ap.add_argument("--text_embed_prefix", default="x_text_")

    args = ap.parse_args()

    df = _load(args.data).copy()
    y_col = _pick_target(df, args.target)
    y_raw = df[y_col].astype("string").str.lower().fillna("")
    mapping = {"neg":0,"neu":1,"pos":2}
    # normalizar etiquetas
    y = y_raw.map(lambda s: "neg" if "neg" in s else ("pos" if "pos" in s else ("neu" if "neu" in s else "")))
    mask = y.isin(["neg","neu","pos"])
    df = df[mask].reset_index(drop=True)
    y = y[mask].map(mapping).to_numpy(dtype=np.int64)

    # X: se dejará que el modelo construya feat_cols a partir del DF completo
    model = RBMGeneral(
        n_hidden=args.n_hidden, cd_k=args.cd_k, seed=args.seed
    )
    info = model.fit(
        df,
        scale_mode=args.scale_mode,
        lr_rbm=args.lr_rbm, lr_head=args.lr_head,
        epochs=args.epochs, epochs_rbm=args.epochs_rbm,
        batch_size=args.batch_size,
        use_text_probs=args.use_text_probs,
        use_text_embeds=args.use_text_embeds,
        max_calif=args.max_calif,
        text_embed_prefix=args.text_embed_prefix,
    )

    out_dir = args.out or os.path.join("artifacts","jobs", f"rbm_general_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)
    with open(os.path.join(out_dir,"job_score.json"),"w",encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[OK] RBM General entrenada. Artefactos en: {out_dir}")
    print(f"Scores: {info}")

if __name__ == "__main__":
    main()
