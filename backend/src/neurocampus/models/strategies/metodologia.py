# backend/src/neurocampus/models/strategies/metodologia.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import re
import pandas as pd

# -------------------------------------------------------------------
# Normaliza el formato del periodo: "YYYY-SEM" (e.g., "2024-2")
# -------------------------------------------------------------------
PERIODO_RE = re.compile(r"^(?P<y>\d{4})[-_](?P<s>\d{1,2})$")

def _parse_periodo(value: str) -> tuple[int, int]:
    m = PERIODO_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"Periodo inválido: {value!r}. Use 'YYYY-SEM' p.ej. '2024-2'.")
    return int(m.group("y")), int(m.group("s"))

def _periodo_key(value: str) -> int:
    y, s = _parse_periodo(value)
    return y * 10 + s  # orden correcto para comparar periodos

# -------------------------------------------------------------------
# Base y estrategias
# -------------------------------------------------------------------
@dataclass
class SeleccionConfig:
    periodo_actual: Optional[str] = None  # 'YYYY-SEM'
    ventana_n: int = 4                    # para la Ventana

class BaseMetodologia:
    name = "base"

    def seleccionar(self, df: pd.DataFrame, cfg: SeleccionConfig) -> pd.DataFrame:
        """Devuelve el subconjunto de df según la estrategia."""
        raise NotImplementedError

class PeriodoActualMetodologia(BaseMetodologia):
    name = "periodo_actual"

    def seleccionar(self, df: pd.DataFrame, cfg: SeleccionConfig) -> pd.DataFrame:
        # Si no llega periodo_actual, usamos el máximo presente en df['periodo']
        if not cfg.periodo_actual:
            # Suponemos columna 'periodo' (estandarizada en el pipeline de datos)
            periodo_top = max(df["periodo"].dropna().map(_periodo_key))
            # Reconvertir a "YYYY-SEM"
            y, s = divmod(periodo_top, 10)
            objetivo = f"{y}-{s}"
        else:
            objetivo = cfg.periodo_actual
        return df[df["periodo"].astype(str).str.strip().eq(objetivo)].copy()

class AcumuladoMetodologia(BaseMetodologia):
    name = "acumulado"

    def seleccionar(self, df: pd.DataFrame, cfg: SeleccionConfig) -> pd.DataFrame:
        if not cfg.periodo_actual:
            # usar máximo disponible
            periodo_top = max(df["periodo"].dropna().map(_periodo_key))
        else:
            periodo_top = _periodo_key(cfg.periodo_actual)
        # todos los periodos <= actual
        mask = df["periodo"].dropna().map(_periodo_key) <= periodo_top
        return df[mask].copy()

class VentanaMetodologia(BaseMetodologia):
    name = "ventana"

    def seleccionar(self, df: pd.DataFrame, cfg: SeleccionConfig) -> pd.DataFrame:
        if not cfg.periodo_actual:
            periodo_top = max(df["periodo"].dropna().map(_periodo_key))
        else:
            periodo_top = _periodo_key(cfg.periodo_actual)
        # construir las últimas N llaves
        claves: List[int] = sorted(df["periodo"].dropna().map(_periodo_key).unique())
        # incluir periodo_top aunque no exista en claves (si viene por parámetro)
        if periodo_top not in claves:
            claves.append(periodo_top)
            claves = sorted(set(claves))
        # tomar los últimos N <= periodo_top
        claves_filtradas = [k for k in claves if k <= periodo_top][-cfg.ventana_n:]
        mask = df["periodo"].dropna().map(_periodo_key).isin(claves_filtradas)
        return df[mask].copy()

# -------------------------------------------------------------------
# Registro simple para resolver por nombre
# -------------------------------------------------------------------
REGISTRY = {
    PeriodoActualMetodologia.name: PeriodoActualMetodologia(),
    AcumuladoMetodologia.name: AcumuladoMetodologia(),
    VentanaMetodologia.name: VentanaMetodologia(),
}

def resolver_metodologia(nombre: Optional[str]) -> BaseMetodologia:
    if not nombre:
        return REGISTRY["periodo_actual"]
    nombre = str(nombre).strip().lower()
    if nombre not in REGISTRY:
        raise ValueError(f"Metodologia desconocida: {nombre}. "
                         f"Opciones: {', '.join(REGISTRY.keys())}")
    return REGISTRY[nombre]
