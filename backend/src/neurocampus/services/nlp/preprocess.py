# backend/src/neurocampus/services/nlp/preprocess.py
from __future__ import annotations

import re
import pandas as pd
from typing import List, Iterable, Optional

# --- Dependencia opcional: emoji ---
try:
    import emoji  # type: ignore
except Exception:
    emoji = None

# Patrones básicos (se preservan del código actual)
URL_PATTERN     = r'(https?://\S+|www\.\S+)'
EMAIL_PATTERN   = r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'
PHONE_PATTERN   = r'\b(?:\+?\d[\d\s\-\(\)]{6,}\d)\b'
MENTION_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#\w+'

# -------------------------
# Limpieza y emoji handling
# -------------------------

def _replace_emojis(text: str) -> str:
    """Transforma emojis en tokens 'emoji_*' (si el paquete emoji está disponible)."""
    if emoji is None:
        return text
    def desc(e):
        name = emoji.demojize(e, delimiters=("", "")).replace(":", "").replace(" ", "_")
        return f"emoji_{name}"
    return emoji.replace_emoji(text, replace=lambda c: desc(c))

def limpiar_texto(texto: str) -> str:
    """
    Limpieza básica: URL/EMAIL/PHONE/MENTION/HASHTAG → tokens; emojis; espacios.
    (Se respeta el comportamiento original.)
    """
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    texto = re.sub(URL_PATTERN,     " <URL> ",     texto)
    texto = re.sub(EMAIL_PATTERN,   " <EMAIL> ",   texto)
    texto = re.sub(PHONE_PATTERN,   " <PHONE> ",   texto)
    texto = re.sub(MENTION_PATTERN, " <MENTION> ", texto)
    texto = re.sub(HASHTAG_PATTERN, lambda m: " <HASHTAG_"+m.group(0)[1:]+"> ", texto)
    texto = _replace_emojis(texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# ------------------------------------
# Tokenización / lematización con spaCy
# ------------------------------------

def _try_load_spacy():
    """
    Intenta cargar spaCy en español. Si spaCy NO está instalado, o no hay modelo,
    devolvemos None y luego caeremos a un fallback ligero que conserva tokens <...>.
    Orden:
      1) es_core_news_sm (si está instalado)
      2) spacy.blank('es') + componente de lemas de respaldo
    """
    try:
        import importlib
        import spacy  # type: ignore
    except Exception:
        return None  # spaCy no está instalado

    # Intentar modelo pequeño si está disponible
    try:
        importlib.import_module("es_core_news_sm")
        return spacy.load("es_core_news_sm")
    except Exception:
        pass

    # Si no hay modelo, usar blank('es') y un pipe de lemas de respaldo
    try:
        nlp = spacy.blank("es")
        @spacy.language.Language.component("fallback_lemma")
        def fallback_lemma(doc):
            for t in doc:
                # Igual que el código original: usar lower como "lemma" si no hay modelo
                t.lemma_ = t.text.lower()
            return doc
        nlp.add_pipe("fallback_lemma")
        return nlp
    except Exception:
        return None

def _lemmatize_with_spacy(nlp, texts: List[str], batch_size: int = 512) -> List[str]:
    """
    Lematiza con spaCy (o usa el fallback_lemma si no hay lematizador real).
    Mantiene tokens especiales <URL>/<EMAIL>/<HASHTAG_*>/<MENTION> y descarta
    puntuación/espacios. Solo admite tokens con letras (como el código original).
    """
    disable = [p for p in ("parser", "ner", "textcat", "tok2vec") if p in getattr(nlp, "pipe_names", [])]
    out: List[str] = []
    import re as _re

    with nlp.select_pipes(disable=disable) if hasattr(nlp, "select_pipes") else nullcontext():
        for doc in nlp.pipe(texts, batch_size=batch_size):
            toks = []
            for token in doc:
                txt = token.text
                # Mantener tokens especiales generados por limpiar_texto
                if txt.startswith("<") and txt.endswith(">"):
                    toks.append(txt)
                    continue
                if token.is_punct or token.is_space:
                    continue
                # Mantener solo tokens con letras (incluyendo acentos), igual que antes
                if not _re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]", txt):
                    continue
                lemma = (token.lemma_ or txt).lower().strip()
                if lemma == "-pron-":
                    lemma = txt.lower()
                toks.append(lemma)
            out.append(" ".join(toks))
    return out

# Context manager nulo para compatibilidad en caso de spaCy < 3
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# --------------------------
# Fallback sin spaCy (robusto)
# --------------------------

_PUNCT_CHARS = set(".,;:!?()[]{}\"'`«»“”’–—-_/\\|~^+=*<>")

_LETTERS_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]")

def _lemmatize_fallback(texts: List[str]) -> List[str]:
    """
    Fallback ligero si no hay spaCy:
    - Asume que los textos ya pasaron por limpiar_texto (así conserva <URL>, etc.)
    - Tokeniza por espacios
    - Mantiene tokens <...>
    - Descarta tokens que sean solo puntuación/espacios
    - Mantiene solo tokens con letras (como el original)
    - "Lemma" = lower del token
    """
    out: List[str] = []
    for s in texts:
        if not s:
            out.append("")
            continue
        toks = []
        for tok in s.split():
            if tok.startswith("<") and tok.endswith(">"):
                toks.append(tok)
                continue
            # descartar pura puntuación
            if all(ch in _PUNCT_CHARS for ch in tok):
                continue
            if not _LETTERS_RE.search(tok):
                continue
            toks.append(tok.lower())
        out.append(" ".join(toks))
    return out

def tokenizar_y_lematizar_batch(texts: List[str], batch_size: int = 512) -> List[str]:
    """
    Mantiene tokens especiales <URL>/<EMAIL>/<HASHTAG_*>/<MENTION> y descarta
    pura puntuación/espacios. Si spaCy/modelo no está disponible, cae a fallback.
    """
    texts = list(texts or [])
    if not texts:
        return []
    nlp = _try_load_spacy()
    if nlp is None:
        return _lemmatize_fallback(texts)
    try:
        return _lemmatize_with_spacy(nlp, texts, batch_size=batch_size)
    except Exception:
        # Si spaCy falla por cualquier motivo, garantizamos continuidad
        return _lemmatize_fallback(texts)
