# backend/src/neurocampus/services/nlp/preprocess.py
import re
import pandas as pd
from typing import List

try:
    import emoji
except Exception:
    emoji = None

# Patrones básicos
URL_PATTERN     = r'(https?://\S+|www\.\S+)'
EMAIL_PATTERN   = r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'
PHONE_PATTERN   = r'\b(?:\+?\d[\d\s\-\(\)]{6,}\d)\b'
MENTION_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#\w+'

def _replace_emojis(text: str) -> str:
    """Transforma emojis en tokens 'emoji_*' (si el paquete emoji está disponible)."""
    if emoji is None:
        return text
    def desc(e):
        name = emoji.demojize(e, delimiters=("","")).replace(":", "").replace(" ", "_")
        return f"emoji_{name}"
    return emoji.replace_emoji(text, replace=lambda c: desc(c))

def limpiar_texto(texto: str) -> str:
    """Limpieza básica: URL/EMAIL/PHONE/MENTION/HASHTAG → tokens; emojis; espacios."""
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

# ---- Lematización (spaCy) ----
def _try_load_spacy():
    import importlib, spacy
    try:
        importlib.import_module("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")
    except Exception:
        nlp = spacy.blank("es")
        @spacy.language.Language.component("fallback_lemma")
        def fallback_lemma(doc):
            for t in doc:
                t.lemma_ = t.text.lower()
            return doc
        nlp.add_pipe("fallback_lemma")
    return nlp

def tokenizar_y_lematizar_batch(texts: List[str], batch_size: int = 512) -> List[str]:
    """
    Mantiene tokens especiales <URL>/<EMAIL>/<HASHTAG_*>/<MENTION> y descarta pura puntuación/espacios.
    """
    import re as _re
    nlp = _try_load_spacy()
    disable = [p for p in ("parser","ner","textcat","tok2vec") if p in nlp.pipe_names]
    out = []
    with nlp.select_pipes(disable=disable):
        for doc in nlp.pipe(texts, batch_size=batch_size):
            toks = []
            for token in doc:
                txt = token.text
                if txt.startswith("<") and txt.endswith(">"):
                    toks.append(txt); continue
                if token.is_punct or token.is_space:
                    continue
                if not _re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]", txt):
                    continue
                lemma = (token.lemma_ or txt).lower().strip()
                if lemma == "-pron-": lemma = txt.lower()
                toks.append(lemma)
            out.append(" ".join(toks))
    return out
