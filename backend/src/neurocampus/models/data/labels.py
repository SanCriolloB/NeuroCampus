# backend/src/neurocampus/models/data/labels.py
import pandas as pd
from typing import Optional

_ALLOWED = {"neg","neu","pos"}
_MAP = {
    "neg":"neg","negative":"neg","negativo":"neg",
    "neu":"neu","neutral":"neu",
    "pos":"pos","positive":"pos","positivo":"pos",
}

def _norm_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return _MAP.get(s, None)

def resolve_sentiment_labels(
    df: pd.DataFrame,
    require_teacher_accept: bool = True,
    accept_threshold: float = 0.80
) -> pd.Series:
    """
    Devuelve la etiqueta final por fila siguiendo la regla:
    humano (y_sentimiento) > teacher (sentiment_label_teacher).
    - Normaliza etiquetas a {neg, neu, pos}.
    - Si require_teacher_accept=True, solo usa teacher cuando:
        accepted_by_teacher == 1
      Ó, si no existe esa columna:
        sentiment_conf >= accept_threshold
      Si ninguna de las dos está, acepta todo (modo retrocompatible).
    - Si no hay etiqueta válida, retorna NaN (entrenamiento debe filtrar).
    """
    n = len(df)
    y_h = df["y_sentimiento"].map(_norm_label) if "y_sentimiento" in df else pd.Series([None]*n, index=df.index)

    y_t = df["sentiment_label_teacher"].map(_norm_label) if "sentiment_label_teacher" in df else pd.Series([None]*n, index=df.index)

    if require_teacher_accept:
        if "accepted_by_teacher" in df:
            ok_t = df["accepted_by_teacher"].fillna(0).astype(int).astype(bool)
        elif "sentiment_conf" in df:
            ok_t = df["sentiment_conf"].fillna(0) >= accept_threshold
        else:
            # Si no tenemos ningún indicador de aceptación, por retrocompatibilidad aceptamos todo
            ok_t = pd.Series([True]*n, index=df.index)
    else:
        ok_t = pd.Series([True]*n, index=df.index)

    # humano > teacher (solo si teacher_ok)
    y = y_h.copy()
    use_teacher = y.isna() & y_t.notna() & ok_t
    y.loc[use_teacher] = y_t.loc[use_teacher]

    # Asegura pertenencia al alfabeto permitido
    y = y.where(y.isin(_ALLOWED), other=pd.NA)
    return y
