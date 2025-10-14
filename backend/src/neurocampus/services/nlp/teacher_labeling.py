# backend/src/neurocampus/services/nlp/teacher_labeling.py
# Uso:
# python -m neurocampus.services.nlp.teacher_labeling --in data/processed/evaluaciones_2025.parquet \
#   --out data/labeled/evaluaciones_2025_teacher.parquet --model pysentimiento/robertuito-sentiment-analysis \
#   --label-map 3class_neg-neu-pos --threshold 0.80 --batch-size 32
#
# Salida: mismo DF + columnas p_neg, p_neu, p_pos, sentiment_label_teacher, sentiment_conf, accepted_by_teacher
# y un archivo meta JSON con métricas de aceptación.

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_label(s: str) -> str:
    """Normaliza etiquetas del modelo a {neg, neu, pos}."""
    s = str(s).strip().lower()
    if s.startswith("neg"):    # NEG, negative, negativo, LABEL_0 (si algún modelo la usa)
        return "neg"
    if s.startswith("neu"):    # NEU, neutral
        return "neu"
    if s.startswith("pos"):    # POS, positive, positivo
        return "pos"
    # Algunos modelos usan LABEL_0/1/2; mapeo típico NEG/NEU/POS
    if s in {"label_0"}:
        return "neg"
    if s in {"label_1"}:
        return "neu"
    if s in {"label_2"}:
        return "pos"
    return s


def _auto_device():
    """Devuelve 0 si hay CUDA disponible, si no -1 (CPU). No falla si torch no está instalado."""
    try:
        import torch  # noqa
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _run_teacher(df: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    """
    Ejecuta pipeline de transformers y devuelve matriz P de shape (N,3)
    con columnas en orden [p_neg, p_neu, p_pos].
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=_auto_device(),
        return_all_scores=True,
        truncation=True,
    )

    texts = df["comentario"].astype(str).tolist()
    preds = pipe(texts, batch_size=batch_size)  # list of list[{'label': 'NEG', 'score': 0.12}, ...]
    P = np.zeros((len(preds), 3), dtype=float)
    order = ["neg", "neu", "pos"]

    for i, row in enumerate(preds):
        # row: [{'label': 'NEG', 'score': x}, {'label': 'NEU', ...}, {'label': 'POS', ...}]
        scores = { _norm_label(d["label"]): float(d["score"]) for d in row }
        # Asegura las 3 entradas; si falta alguna, pon 0
        pneg = scores.get("neg", 0.0)
        pneu = scores.get("neu", 0.0)
        ppos = scores.get("pos", 0.0)
        s = pneg + pneu + ppos
        if s > 0:
            pneg, pneu, ppos = pneg/s, pneu/s, ppos/s
        P[i, 0], P[i, 1], P[i, 2] = pneg, pneu, ppos

    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--model", default="pysentimiento/robertuito-sentiment-analysis")
    ap.add_argument("--label-map", default="3class_neg-neu-pos")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # Carga de datos
    if args.src.endswith(".parquet"):
        df = pd.read_parquet(args.src)
    else:
        df = pd.read_csv(args.src)
    assert "comentario" in df.columns, "Falta columna 'comentario'"

    # Ejecutar Teacher real (Transformers)
    try:
        P = _run_teacher(df, args.model, args.batch_size)
    except ModuleNotFoundError as e:
        # Mensaje claro si faltan dependencias
        raise RuntimeError(
            "Faltan dependencias para el Teacher. Instala:\n"
            "  pip install 'transformers>=4.41' 'torch>=2.2'\n"
            "o usa un entorno compatible (recomendado Python 3.11)."
        ) from e

    # Construye columnas de salida
    df["p_neg"], df["p_neu"], df["p_pos"] = P[:, 0], P[:, 1], P[:, 2]
    idx = P.argmax(axis=1)
    lbl_order = np.array(["neg", "neu", "pos"], dtype=object)
    labels = lbl_order[idx]
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
        "label_map": args.label_map,
        "mode": "real",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(len(df)),
        "accepted_count": int(df["accepted_by_teacher"].sum()),
        "threshold": float(args.threshold),
        "batch_size": int(args.batch_size),
        "label_distribution": {
            "neg": int((labels == "neg").sum()),
            "neu": int((labels == "neu").sum()),
            "pos": int((labels == "pos").sum()),
        },
    }
    with open(args.dst + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Resumen en stdout (útil en CI)
    print({
        "out": args.dst,
        "n_rows": meta["n_rows"],
        "accepted": meta["accepted_count"],
        "accept_rate": round(meta["accepted_count"]/max(1, meta["n_rows"]), 4),
        "labels": meta["label_distribution"],
    })


if __name__ == "__main__":
    main()
