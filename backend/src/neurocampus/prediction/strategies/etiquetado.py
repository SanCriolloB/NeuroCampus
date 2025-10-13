# backend/src/neurocampus/prediction/strategies/etiquetado.py
from typing import Dict, Any

def vectorize_simple(texto: str, califs: dict) -> dict:
    """
    Vectorización mínima de ejemplo (stub).
    Reemplazar por: TF-IDF/embeddings + features cuantitativas normalizadas.
    """
    x = {f"pregunta_{i}": float(califs.get(f"pregunta_{i}", 0)) for i in range(1, 11)}
    x["len_comentario"] = float(len(texto or ""))
    return x

def infer_stub(artifacts: dict, X: dict) -> Dict[str, Any]:
    """
    Inferencia de ejemplo: devuelve probabilidades *coherentes*.
    Sustituir por carga de vectorizer + modelo (campeón o job_id) y predicción real.
    """
    # Heurística inocua: más longitud -> más 'Álgebra', etc. (solo stub)
    base = 0.5 if X.get("len_comentario", 0) > 40 else 0.3
    materia_scores = {"Álgebra": base, "Cálculo": 0.4, "Física": 0.3}
    # Normaliza a probas
    s = sum(materia_scores.values())
    materia_scores = {k: v/s for k, v in materia_scores.items()}
    sentiment_scores = {"pos": 0.6, "neu": 0.3, "neg": 0.1}
    return {"materia_scores": materia_scores, "sentiment_scores": sentiment_scores}
