# backend/src/neurocampus/models/strategies/metodologia.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple, Optional

from ...data.strategies.unificacion import UnificacionStrategy

class Metodo(Enum):
    PERIODO_ACTUAL = "PeriodoActual"
    ACUMULADO = "Acumulado"
    VENTANA = "Ventana"

@dataclass
class MetodoParams:
    ultimos: Optional[int] = None
    desde: Optional[str] = None   # "2024-2"
    hasta: Optional[str] = None   # "2025-1"

class DatasetResolver:
    """
    Devuelve (uri, meta) del dataset según la metodología requerida.
    """
    def __init__(self, base_uri: str = "localfs://."):
        self.uni = UnificacionStrategy(base_uri=base_uri)

    def resolve(self, metodo: Metodo, params: MetodoParams | None = None) -> Tuple[str, Dict[str, Any]]:
        params = params or MetodoParams()
        if metodo == Metodo.PERIODO_ACTUAL:
            return self.uni.periodo_actual()
        if metodo == Metodo.ACUMULADO:
            return self.uni.acumulado()
        if metodo == Metodo.VENTANA:
            return self.uni.ventana(ultimos=params.ultimos, desde=params.desde, hasta=params.hasta)
        raise ValueError(f"Metodología no soportada: {metodo}")
