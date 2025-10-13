# backend/src/neurocampus/app/jobs/cmd_preprocesar_beto.py
# - Limpia + lematiza texto
# - Etiqueta sentimiento con BETO (dos modos):
#    * probs (recomendado): p_neg/p_neu/p_pos + gating (threshold, margin, neu_min)
#    * simple: replica pipeline del notebook y etiqueta top1 con score
# - Deja 'comentario' estandarizado y columnas para el Student (RBM).

import argparse, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
from neurocampus.services.nlp.preprocess import limpiar_texto, tokenizar_y_lematizar_batch
from neurocampus.services.nlp.teacher import run_transformer, accept_mask
from transformers import pipeline  # para el modo simple del notebook

TEXT_CANDIDATES = ["Sugerencias","sugerencias","comentario","comentarios","observaciones","obs","texto","review","opinion"]

def _pick_text_col(df: pd.DataFrame, prefer: str|None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    norm = {c: re.sub(r"\s+","",c).lower() for c in df.columns}
    for cand in TEXT_CANDIDATES:
        if cand in df.columns: return cand
        n_cand = re.sub(r"\s+","",cand).lower()
        for raw, n in norm.items():
            if n == n_cand: return raw
    raise ValueError("No se encontró columna de texto. Usa --text-col para forzar.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="CSV/Parquet estandarizado (calif_1..10 + texto).")
    ap.add_argument("--out", dest="dst", required=True, help="Salida etiquetada (parquet/csv).")
    ap.add_argument("--text-col", default=None, help="Nombre exacto de la columna de texto si quieres forzarlo.")
    ap.add_argument("--beto-model", default="finiteautomata/beto-sentiment-analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--neu-min", type=float, default=0.75)
    ap.add_argument("--beto-mode", choices=["probs","simple"], default="probs",
                    help="probs: p_neg/p_neu/p_pos + gating; simple: pipeline tipo notebook (top1 + score)")
    args = ap.parse_args()

    # 1) Cargar datos
    df = pd.read_parquet(args.src) if args.src.lower().endswith(".parquet") else pd.read_csv(args.src)

    # 2) Seleccionar col de texto + limpiar + lematizar
    text_col = _pick_text_col(df, args.text_col)
    df["_texto_clean"]  = df[text_col].astype(str).map(limpiar_texto)
    df["_texto_lemmas"] = tokenizar_y_lematizar_batch(df["_texto_clean"].tolist(), batch_size=512)

    # 3) BETO
    if args.beto-mode == "simple":
        # --- MODO SIMPLE (como el notebook) ---
        sentiment_analyzer = pipeline("sentiment-analysis", model=args.beto_model)
        df['Sugerencias_lemmatizadas'] = df['_texto_lemmas'].fillna('').astype(str)
        results = sentiment_analyzer(df['Sugerencias_lemmatizadas'].tolist())
        # etiqueta top1 + score
        df['sentimiento'] = [r['label'] for r in results]
        lbl_map = {"NEG": "neg", "NEU": "neu", "POS": "pos",
                   "negative": "neg", "neutral": "neu", "positive": "pos"}
        df["sentiment_label_teacher"] = df["sentimiento"].map(lbl_map).fillna(df["sentimiento"].str.lower())
        df["sentiment_conf"] = [r.get("score", 1.0) for r in results]
        df["accepted_by_teacher"] = (df["sentiment_conf"] >= args.threshold).astype(int)
        # columnas p_* quedan NaN (no hay probs completas en modo simple)
        df["p_neg"], df["p_neu"], df["p_pos"] = np.nan, np.nan, np.nan
    else:
        # --- MODO PROBS (recomendado) ---
        P = run_transformer(df["_texto_lemmas"].astype(str).tolist(), args.beto_model, batch_size=args.batch_size)
        df["p_neg"], df["p_neu"], df["p_pos"] = P[:,0], P[:,1], P[:,2]
        idx = P.argmax(axis=1)
        labels = np.array(["neg","neu","pos"], dtype=object)[idx]
        conf = P.max(axis=1)
        df["sentiment_label_teacher"] = labels
        df["sentiment_conf"] = conf
        acc = accept_mask(P, labels, threshold=args.threshold, margin=args.margin, neu_min=args.neu_min)
        df["accepted_by_teacher"] = acc.astype(int)

    # 4) Asegurar columna 'comentario' (trazabilidad humana)
    if "comentario" not in df.columns:
        df["comentario"] = df["_texto_clean"]

    # 5) Guardar dataset + meta
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if args.dst.lower().endswith(".parquet"):
        df.to_parquet(args.dst, index=False)
    else:
        df.to_csv(args.dst, index=False)

    meta = {
        "model": args.beto_model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(len(df)),
        "accepted_count": int(df["accepted_by_teacher"].fillna(0).astype(int).sum()),
        "threshold": float(args.threshold),
        "margin": float(args.margin),
        "neu_min": float(args.neu_min),
        "text_col": text_col
    }
    with open(args.dst + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print({"out": args.dst, "n_rows": meta["n_rows"],
           "accept_rate": float(df["accepted_by_teacher"].fillna(0).astype(int).mean()),
           "text_col": text_col})

if __name__ == "__main__":
    main()
