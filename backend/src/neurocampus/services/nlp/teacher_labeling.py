# backend/src/neurocampus/services/nlp/teacher_labeling.py
# Uso:
# python -m neurocampus.services.nlp.teacher_labeling --in data/processed/evaluaciones_2025.parquet \
#   --out data/labeled/evaluaciones_2025_teacher.parquet --model pysentimiento/robertuito-sentiment-analysis \
#   --label-map 3class_neg-neu-pos --threshold 0.80
#
# Salida: mismo DF + columnas p_neg, p_neu, p_pos, sentiment_label_teacher, sentiment_conf, accepted_by_teacher
# y un archivo meta JSON con métricas de aceptación.

import argparse, json, time
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--model", default="pysentimiento/robertuito-sentiment-analysis")
    ap.add_argument("--label-map", default="3class_neg-neu-pos")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    df = pd.read_parquet(args.src) if args.src.endswith(".parquet") else pd.read_csv(args.src)
    assert "comentario" in df.columns, "Falta columna 'comentario'"

    # Carga del modelo (placeholder rápido sin internet): en proyecto real usar transformers pipeline
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    # tok = AutoTokenizer.from_pretrained(args.model)
    # mdl = AutoModelForSequenceClassification.from_pretrained(args.model)
    # nlp = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)

    # Simulación mínima: asigna probabilidades dummy si no hay acceso al modelo.
    # Reemplaza por pipeline real en entorno con pesos descargados.
    import numpy as np
    rng = np.random.default_rng(7)
    P = rng.random((len(df), 3))
    P = P / P.sum(axis=1, keepdims=True)

    df["p_neg"], df["p_neu"], df["p_pos"] = P[:,0], P[:,1], P[:,2]
    labels = np.array(["neg","neu","pos"])[P.argmax(axis=1)]
    conf = P.max(axis=1)
    df["sentiment_label_teacher"] = labels
    df["sentiment_conf"] = conf
    df["accepted_by_teacher"] = (conf >= args.threshold).astype(int)

    # Guarda dataset
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if args.dst.endswith(".parquet"):
        df.to_parquet(args.dst, index=False)
    else:
        df.to_csv(args.dst, index=False)

    # Meta con tasa de aceptación
    meta = {
        "model": args.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(len(df)),
        "accepted_count": int(df["accepted_by_teacher"].sum()),
        "threshold": args.threshold,
    }
    with open(args.dst + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
