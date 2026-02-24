"""
feature_selectors — Selección unificada de columnas de entrada para modelos NeuroCampus.
========================================================================================

Este módulo centraliza la lógica de selección de features (columnas numéricas,
probabilidades de texto y embeddings de texto) que antes estaba duplicada en
``modelo_rbm_general`` y ``modelo_rbm_restringida``.

Proporciona trazabilidad completa del feature set seleccionado para cada run,
incluyendo conteos y nombres de columnas de texto, lo cual es necesario para
la documentación de reproducibilidad y auditoría de modelos.

Uso típico::

    from neurocampus.models.utils.feature_selectors import pick_feature_cols

    result = pick_feature_cols(
        df,
        max_calif=10,
        include_text_probs=True,
        include_text_embeds=True,
    )
    X_np = df[result.feature_cols].to_numpy(dtype="float32")
    meta = result.traceability_dict()  # → para meta.json / predictor.json

.. versionadded:: P2.6
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

__all__ = [
    "FeatureSelectionResult",
    "pick_feature_cols",
    "auto_detect_embed_prefix",
    "CANDIDATE_EMBED_PREFIXES",
    "META_EXCLUDE_COLS",
]


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Prefijos candidatos para autodetección de columnas de embeddings de texto.
#: Se prueban en orden; el primero que coincida con alguna columna del DF gana.
CANDIDATE_EMBED_PREFIXES: List[str] = [
    "x_text_",
    "text_emb_",
    "emb_",
    "feat_text_",
    "feat_t_",
    "mean_feat_t_",
    "std_feat_t_",
]

#: Columnas de metadatos que NO deben incluirse como features numéricas,
#: incluso si son de tipo numérico.  Se comparan en minúsculas.
META_EXCLUDE_COLS: set[str] = {
    "y",
    "y_sentimiento",
    "y_label",
    "label",
    "target",
    "periodo",
    "semestre",
    "alumno_id",
    "teacher_id",
    "materia_id",
    "docente_id",
    "n_par",
    "student_id",
    "id",
    "row_id",
    "index",
    "unnamed: 0",
    "sentiment_label_teacher",
    "accept_teacher",
    "p_neg",
    "p_neu",
    "p_pos",
}

#: Patrones regex para columnas numéricas aceptadas (clasificación).
_NUMERIC_PATTERNS: List[str] = [
    r"^calif_\d+$",
    r"^pregunta_\d+$",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_lower(s: Any) -> str:
    """Convierte a minúsculas de forma segura."""
    try:
        return str(s).strip().lower()
    except Exception:
        return ""


def _suffix_index(name: str, prefix: str) -> int:
    """Extrae el índice numérico del sufijo de una columna.

    Parameters
    ----------
    name : str
        Nombre completo de la columna (e.g. ``"x_text_42"``).
    prefix : str
        Prefijo a eliminar (e.g. ``"x_text_"``).

    Returns
    -------
    int
        Índice numérico extraído, o ``0`` si no es parseable.

    Examples
    --------
    >>> _suffix_index("x_text_5", "x_text_")
    5
    """
    try:
        return int(name[len(prefix):])
    except (ValueError, IndexError):
        return 0

_EMBED_SUFFIX_RE = re.compile(r"^\d+$")

def _is_embed_column(name: str, prefix: str) -> bool:
    """True si `name` representa una dimensión de embedding para `prefix`.

    Se exige que el sufijo después del prefijo sea numérico (e.g. `feat_t_0`).
    Esto evita falsos positivos en columnas como `text_coverage`.
    """
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix):]
    return bool(_EMBED_SUFFIX_RE.match(suffix))



def auto_detect_embed_prefix(columns: Sequence[str]) -> Optional[str]:
    """Detecta automáticamente el prefijo de columnas de embeddings de texto.

    Recorre :data:`CANDIDATE_EMBED_PREFIXES` en orden y devuelve el primero
    que coincida con al menos una columna del listado proporcionado.

    Parameters
    ----------
    columns : Sequence[str]
        Lista de nombres de columnas del DataFrame.

    Returns
    -------
    Optional[str]
        El prefijo detectado, o ``None`` si ninguno coincide.
    """
    cols_list = list(columns)
    for prefix in CANDIDATE_EMBED_PREFIXES:
        if any(_is_embed_column(c, prefix) for c in cols_list):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Resultado tipado
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    """Resultado de la selección de features con metadata de trazabilidad.

    Attributes
    ----------
    feature_cols : List[str]
        Lista ordenada de columnas seleccionadas para la entrada del modelo.
    calif_cols : List[str]
        Subconjunto de ``feature_cols`` que corresponden a calificaciones
        numéricas (e.g. ``calif_1``, ``calif_2``).
    text_prob_cols : List[str]
        Subconjunto de columnas de probabilidades de texto incluidas
        (``p_neg``, ``p_neu``, ``p_pos``), vacío si no se activaron.
    text_embed_cols : List[str]
        Subconjunto de columnas de embeddings de texto incluidas
        (e.g. ``x_text_0`` .. ``x_text_63``), vacío si no se activaron.
    text_embed_prefix : Optional[str]
        Prefijo utilizado para las columnas de embeddings de texto.
        ``None`` si no se usaron embeddings.
    """

    feature_cols: List[str] = field(default_factory=list)
    calif_cols: List[str] = field(default_factory=list)
    text_prob_cols: List[str] = field(default_factory=list)
    text_embed_cols: List[str] = field(default_factory=list)
    text_embed_prefix: Optional[str] = None

    @property
    def n_features(self) -> int:
        """Número total de features seleccionadas."""
        return len(self.feature_cols)

    @property
    def n_text_features(self) -> int:
        """Número de features derivadas de texto (probs + embeds)."""
        return len(self.text_prob_cols) + len(self.text_embed_cols)

    @property
    def has_text_features(self) -> bool:
        """``True`` si se incluyó al menos una feature de texto."""
        return self.n_text_features > 0

    def traceability_dict(self) -> Dict[str, Any]:
        """Devuelve un diccionario con metadata de trazabilidad para
        persistir en ``meta.json`` o ``predictor.json``.

        Returns
        -------
        Dict[str, Any]
            Diccionario con claves:
            ``n_features``, ``n_text_features``, ``has_text_features``,
            ``text_embed_prefix``, ``text_feat_cols``, ``text_prob_cols``.
        """
        return {
            "n_features": self.n_features,
            "n_calif_features": len(self.calif_cols),
            "n_text_features": self.n_text_features,
            "has_text_features": self.has_text_features,
            "text_embed_prefix": self.text_embed_prefix,
            "text_feat_cols": self.text_embed_cols,
            "text_prob_cols": self.text_prob_cols,
        }


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def pick_feature_cols(
    df: pd.DataFrame,
    *,
    max_calif: int = 10,
    include_text_probs: bool = False,
    include_text_embeds: bool = False,
    text_embed_prefix: Optional[str] = None,
    auto_detect_prefix: bool = True,
) -> FeatureSelectionResult:
    """Selecciona y ordena las columnas de entrada para modelos de clasificación.

    Orden de prioridad de columnas:

    1. **Calificaciones** (``calif_1`` .. ``calif_{max_calif}``), o en su
       defecto columnas numéricas que no sean metadatos.
    2. **Probabilidades de texto** (``p_neg``, ``p_neu``, ``p_pos``) — solo
       si ``include_text_probs=True`` y existen en el DataFrame.
    3. **Embeddings de texto** (``x_text_0`` .. ``x_text_N``) — solo si
       ``include_text_embeds=True``.  El prefijo se autodetecta si no se
       especifica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas candidatas.
    max_calif : int, default 10
        Máximo de columnas de calificación a incluir.
    include_text_probs : bool, default False
        Si ``True``, incluye ``p_neg/p_neu/p_pos`` como features.
    include_text_embeds : bool, default False
        Si ``True``, incluye columnas de embeddings de texto.
    text_embed_prefix : Optional[str], default None
        Prefijo explícito para columnas de embeddings.  Si es ``None`` y
        ``auto_detect_prefix=True``, se autodetecta.
    auto_detect_prefix : bool, default True
        Si ``True`` y ``text_embed_prefix`` es ``None``, intenta detectar
        automáticamente el prefijo de embeddings.

    Returns
    -------
    FeatureSelectionResult
        Resultado con listas de columnas seleccionadas y metadata.

    Examples
    --------
    >>> result = pick_feature_cols(df, include_text_embeds=True)
    >>> X = df[result.feature_cols].to_numpy(dtype="float32")
    >>> print(result.n_text_features)
    64
    """
    cols = list(df.columns)

    # ------------------------------------------------------------------
    # 1) Calificaciones numéricas
    # ------------------------------------------------------------------
    califs = [c for c in cols if c.startswith("calif_")]
    if califs:
        # Ordenar por índice numérico: calif_1, calif_2, ..., calif_N
        def _idx(c: str) -> int:
            try:
                return int(c.split("_")[1])
            except (ValueError, IndexError):
                return 10**9
        califs = sorted(califs, key=_idx)[:max_calif]
    else:
        # Fallback: columnas numéricas que NO sean metadatos
        # Se aceptan también patrones como pregunta_N
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        califs = [
            c for c in nums
            if _safe_lower(c) not in META_EXCLUDE_COLS
        ][:max_calif]

    features: List[str] = list(califs)

    # ------------------------------------------------------------------
    # 2) Probabilidades del teacher (p_neg, p_neu, p_pos)
    # ------------------------------------------------------------------
    text_prob_selected: List[str] = []
    _prob_cols = ["p_neg", "p_neu", "p_pos"]
    if include_text_probs and all(k in df.columns for k in _prob_cols):
        text_prob_selected = list(_prob_cols)
        features += text_prob_selected

    # ------------------------------------------------------------------
    # 3) Embeddings de texto (x_text_0 .. x_text_N)
    # ------------------------------------------------------------------
    text_embed_selected: List[str] = []
    resolved_prefix: Optional[str] = text_embed_prefix

    if include_text_embeds:
        # Autodetección de prefijo si no se proporcionó
        if resolved_prefix is None and auto_detect_prefix:
            resolved_prefix = auto_detect_embed_prefix(cols)

        if resolved_prefix is None:
            # Default del proyecto si no se detectó nada
            resolved_prefix = "x_text_"

        embed_cols = [c for c in cols if _is_embed_column(c, resolved_prefix)]
        if embed_cols:
            # Ordenar por sufijo numérico: x_text_0, x_text_1, ...
            embed_cols = sorted(
                embed_cols,
                key=lambda c: _suffix_index(c, resolved_prefix),
            )
            text_embed_selected = embed_cols
            features += text_embed_selected

    return FeatureSelectionResult(
        feature_cols=features,
        calif_cols=califs,
        text_prob_cols=text_prob_selected,
        text_embed_cols=text_embed_selected,
        text_embed_prefix=resolved_prefix if text_embed_selected else None,
    )
