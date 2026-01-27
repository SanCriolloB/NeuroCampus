# backend/src/neurocampus/data/strategies/unificacion.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
import unicodedata
from pathlib import Path

# Adapter de almacenamiento (localfs://, etc.)
from ..adapters.almacen_adapter import AlmacenAdapter

# Lectura multi-formato y helpers de DF (con las firmas reales de tus adapters)
from ..adapters.formato_adapter import read_file            # (fileobj, filename) -> DataFrame/like
from ..adapters.dataframe_adapter import as_df              # (obj) -> DF normalizado al engine
from ..strategies.unificacion import UnificacionStrategy as _UnificacionStrategy

import pandas as pd                                         # escritura parquet, manipulación tabular

# ---------------------------------------------------------------------------
# Normalización local de encabezados (para evitar dependencia de validadores.py)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9:_\s]")  # conservamos ':' para Sugerencias:
_MULTI_WS_RE = re.compile(r"\s+")

# Sinónimos/canon mínimo para nuestros datasets
_CANON_MAP = {
    "codigo materia": "codigo_materia",
    "codigomateria": "codigo_materia",
    "cod_materia": "codigo_materia",
    "codigo_asignatura": "codigo_materia",
    "codigo asignatura": "codigo_materia",
    "cedula profesor": "cedula_profesor",
    "cedula_docente": "cedula_profesor",
    "docente_id": "cedula_profesor",
    "grupo_id": "grupo",
    "sugerencias": "Sugerencias:",
    "sugerencias_": "Sugerencias:",
    "observaciones": "Sugerencias:",
    "comentarios": "Sugerencias:",
    "pregunta 1": "pregunta_1",
    "pregunta 2": "pregunta_2",
    "pregunta 3": "pregunta_3",
    "pregunta 4": "pregunta_4",
    "pregunta 5": "pregunta_5",
    "pregunta 6": "pregunta_6",
    "pregunta 7": "pregunta_7",
    "pregunta 8": "pregunta_8",
    "pregunta 9": "pregunta_9",
    "pregunta 10": "pregunta_10",
    "periodo academico": "periodo",
    "periodo_academico": "periodo",
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _slug(s: str) -> str:
    s = _strip_accents(str(s)).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    s = s.replace(" ", "_")
    return s

def normalizar_encabezados(cols: List[str]) -> List[str]:
    """
    Normaliza encabezados:
      - lower + sin acentos
      - elimina puntuación salvo ':' (para 'Sugerencias:')
      - espacios→'_' y colapsa '_'
      - aplica mapa de sinónimos → nombre canónico
    """
    norm: List[str] = []
    for c in cols:
        s = _slug(c)
        # reintentar variantes con/ sin '_' final
        s2 = s.rstrip("_")
        canon = _CANON_MAP.get(s, _CANON_MAP.get(s2, s))
        # Regla especial: si termina en '_' quítalo
        canon = canon.rstrip("_")
        norm.append(canon)
    return norm

# ---------------------------------------------------------------------------

DEDUP_KEYS = ["periodo", "codigo_materia", "grupo", "cedula_profesor"]
PERIODO_RE = re.compile(r"^\d{4}-(1|2)$")  # AAAA-SEM (e.g., 2024-1, 2024-2)

class UnificacionStrategy:
    """
    Wrapper retrocompatible para evitar duplicidad y *circular imports*.

    Motivo:
    - `neurocampus.data.strategies.unificacion` necesita `normalizar_encabezados`
      (definido en este módulo).
    - Si este módulo importa `UnificacionStrategy` desde strategies en top-level,
      se crea un ciclo de importación.

    Solución:
    - Importar la implementación real solo en runtime (lazy import) al instanciar.
    """

    def __init__(self, *args, **kwargs):
        from ..strategies.unificacion import UnificacionStrategy as _Impl  # lazy import
        self._impl = _Impl(*args, **kwargs)

    def __getattr__(self, name):
        # Delegación transparente a la implementación real
        return getattr(self._impl, name)

