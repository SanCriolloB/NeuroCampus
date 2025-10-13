# backend/src/neurocampus/prediction/chain/posprocesado.py
from typing import Dict, Tuple

def calibrate(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Punto único para calibración (Platt/Isotónica o simple normalización).
    Por ahora dejamos identidad (stub), manteniendo interfaz preparada.
    """
    # TODO: reemplazar por calibración real
    total = sum(scores.values()) or 1.0
    return {k: v/total for k, v in scores.items()}

def pick_label(scores: Dict[str, float]) -> Tuple[str, float]:
    label = max(scores, key=scores.get)
    return label, scores[label]

def format_output(raw: Dict[str, dict]) -> Tuple[str, Dict[str,float], Dict[str,float], float]:
    """
    raw: {"materia_scores": {...}, "sentiment_scores": {...}}
    Devuelve (label_top, materia_scores_calibradas, sentiment_scores, confidence)
    """
    mat = calibrate(raw.get("materia_scores", {}))
    sent = raw.get("sentiment_scores", {"pos":0.33,"neu":0.33,"neg":0.34})
    top, conf = pick_label(mat)
    return top, mat, sent, conf
