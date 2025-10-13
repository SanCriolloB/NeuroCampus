# backend/src/neurocampus/services/nlp/teacher.py
# Infere sentimiento con un modelo Hugging Face y arma gating de aceptación.
from typing import List
import numpy as np

def _norm_lbl(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("neg") or s == "label_0": return "neg"
    if s.startswith("neu") or s == "label_1": return "neu"
    if s.startswith("pos") or s == "label_2": return "pos"
    return s

def run_transformer(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Devuelve matriz P (N,3) en orden [neg, neu, pos] usando top_k=None."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, truncation=True, device=-1)
    preds = pipe(texts, batch_size=batch_size)
    P = np.zeros((len(preds), 3), dtype=float)
    for i, row in enumerate(preds):
        scores = { _norm_lbl(d["label"]): float(d["score"]) for d in row }
        v = [scores.get("neg",0.0), scores.get("neu",0.0), scores.get("pos",0.0)]
        s = sum(v) or 1.0
        P[i,:] = np.array(v)/s
    return P

def accept_mask(P: np.ndarray, labels: np.ndarray, threshold: float=0.75, margin: float=0.15, neu_min: float=0.75) -> np.ndarray:
    """Aceptación si top >= threshold y (top-second) >= margin; NEU exige p_neu >= neu_min."""
    top = P.max(axis=1)
    second = np.partition(P, -2, axis=1)[:,-2]
    acc = (top >= threshold) & ((top - second) >= margin)
    acc = acc & (~(labels=="neu") | (P[:,1] >= neu_min))
    return acc
