# backend/src/neurocampus/data/utils/headers.py
"""
Utilidades para normalización de encabezados de columnas.

Este módulo NO debe importar estrategias ni jobs, para evitar import cycles.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


def _strip_accents(s: str) -> str:
    """Remueve acentos y normaliza unicode."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def _clean_col(name: str) -> str:
    """
    Normaliza un nombre de columna a snake_case simple:
    - lower
    - sin acentos
    - separadores a '_'
    - elimina caracteres raros
    """
    s = str(name).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^\w]+", "_", s)         # todo lo no alfanum -> _
    s = re.sub(r"_+", "_", s).strip("_")  # colapsar __ y trim
    return s


def normalizar_encabezados(cols: Iterable[str]) -> List[str]:
    """
    Normaliza lista de encabezados.

    Args:
        cols: iterable de nombres de columnas.

    Returns:
        Lista de columnas normalizadas.
    """
    return [_clean_col(c) for c in cols]
