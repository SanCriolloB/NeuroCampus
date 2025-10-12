# backend/src/neurocampus/data/strategies/unificacion.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
from pathlib import Path

# Adapter de almacenamiento (localfs://, etc.)
from ..adapters.almacen_adapter import AlmacenAdapter

# Lectura multi-formato y helpers de DF (con las firmas reales de tus adapters)
from ..adapters.formato_adapter import read_file            # (fileobj, filename) -> DataFrame/like
from ..adapters.dataframe_adapter import as_df              # (obj) -> DF normalizado al engine

import pandas as pd                                         # escritura parquet, manipulación tabular

# Normalización/validación usada en D2–D4
from ..chain.validadores import normalizar_encabezados

# ---------------------------------------------------------------------------

DEDUP_KEYS = ["periodo", "codigo_materia", "grupo", "cedula_profesor"]
PERIODO_RE = re.compile(r"^\d{4}-(1|2)$")  # AAAA-SEM (e.g., 2024-1, 2024-2)

class UnificacionStrategy:
    """
    Unifica datasets históricos bajo tres metodologías:
      - PeriodoActual: último periodo disponible
      - Acumulado: concat + dedupe
      - Ventana: últimos N periodos o rango [desde,hasta]
    Salidas en /historico como .parquet (columnas normalizadas).
    """

    def __init__(self, base_uri: str = "localfs://."):
        self.store = AlmacenAdapter(base_uri)

    # -------- Descubrimiento de periodos --------
    def listar_periodos(self, prefix: str = "datasets/") -> List[str]:
        """
        Lista carpetas de la forma AAAA-SEM dentro de datasets/.
        Se basa en self.store.ls(prefix) que devuelve rutas (strings).
        """
        items = self.store.ls(prefix)
        periodos: List[str] = []
        for it in items:
            name = Path(it).name  # último segmento de la ruta
            if PERIODO_RE.match(name):
                periodos.append(name)
        periodos.sort()
        return periodos

    # -------- Lectura y normalización por período --------
    def _leer_periodo(self, periodo: str) -> pd.DataFrame:
        """
        Lee el dataset de un período dado priorizando parquet > csv > xlsx.
        - Usa read_file(fileobj, filename) del formato_adapter
        - Convierte a DF del engine vía as_df y, si no es pandas, lo pasa a pandas
        - Normaliza encabezados y asegura columna 'periodo'
        """
        folder = f"datasets/{periodo}"
        candidatos = ("data.parquet", "data.csv", "data.xlsx")

        for candidate in candidatos:
            uri = f"{folder}/{candidate}"
            if self.store.exists(uri):
                # Abrimos en binario; el adapter de formatos decide cómo leer según 'filename'
                mode = "rb"
                with self.store.open(uri, mode) as fh:
                    df_like = read_file(fh, uri)   # adapter decide por extensión
                pdf = as_df(df_like)

                # Si el engine subyacente no es pandas (p.ej. polars), convertir a pandas
                if hasattr(pdf, "to_pandas"):
                    try:
                        pdf = pdf.to_pandas()
                    except Exception:
                        # Si to_pandas falla por cualquier razón, forzamos a DataFrame
                        pdf = pd.DataFrame(pdf)

                # Normalización de encabezados (D2–D4)
                pdf.columns = normalizar_encabezados(list(pdf.columns))

                # Asegurar 'periodo'
                if "periodo" not in pdf.columns:
                    pdf["periodo"] = periodo

                return pdf

        raise FileNotFoundError(f"No se encontró dataset para periodo {periodo} en {folder}/")

    # -------- Utilitarios --------
    def _dedupe_concat(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatena y elimina duplicados por claves canónicas cuando existan.
        """
        if not frames:
            raise ValueError("No hay frames para unificar")
        big = pd.concat(frames, ignore_index=True, copy=False)
        keys = [k for k in DEDUP_KEYS if k in big.columns]
        if keys:
            big = big.drop_duplicates(subset=keys)
        else:
            big = big.drop_duplicates()
        return big

    def _write_parquet(self, df: pd.DataFrame, out_uri: str) -> None:
        """
        Escribe en parquet usando el file handle del store. Se abre en 'wb'.
        """
        parent = str(Path(out_uri).parent).replace("\\", "/")
        if parent and parent not in ("", "."):
            self.store.makedirs(parent)
        with self.store.open(out_uri, "wb") as out_fh:
            df.to_parquet(out_fh, index=False)

    # -------- Metodologías --------
    def periodo_actual(self) -> Tuple[str, Dict[str, Any]]:
        """
        Toma el último período (lexicográficamente mayor), normaliza y escribe
        historico/periodo_actual/<AAAA-SEM>.parquet
        """
        periodos = self.listar_periodos()
        if not periodos:
            raise RuntimeError("No hay periodos en datasets/")
        ultimo = periodos[-1]
        pdf = self._leer_periodo(ultimo)

        out_uri = f"historico/periodo_actual/{ultimo}.parquet"
        self._write_parquet(pdf, out_uri)

        return out_uri, {"periodo": ultimo, "rows": int(len(pdf))}

    def acumulado(self) -> Tuple[str, Dict[str, Any]]:
        """
        Concatena todos los períodos disponibles, deduplica y escribe
        historico/unificado.parquet
        """
        periodos = self.listar_periodos()
        frames = [self._leer_periodo(p) for p in periodos]
        pdf = self._dedupe_concat(frames)

        out_uri = "historico/unificado.parquet"
        self._write_parquet(pdf, out_uri)

        return out_uri, {"periodos": len(frames), "rows": int(len(pdf))}

    def ventana(
        self,
        ultimos: Optional[int] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Toma una ventana temporal por:
          - 'ultimos' N periodos, o
          - rango inclusivo [desde, hasta] (strings AAAA-SEM).
        Escribe historico/ventanas/unificado_<desde>_<hasta>.parquet
        """
        periodos = self.listar_periodos()
        if ultimos:
            sel = periodos[-ultimos:]
        else:
            if not (desde and hasta):
                raise ValueError("Se requiere 'ultimos' o bien ('desde' y 'hasta')")
            sel = [p for p in periodos if desde <= p <= hasta]

        frames = [self._leer_periodo(p) for p in sel]
        pdf = self._dedupe_concat(frames)

        tag = f"{sel[0]}_{sel[-1]}" if sel else "vacia"
        out_uri = f"historico/ventanas/unificado_{tag}.parquet"
        self._write_parquet(pdf, out_uri)

        return out_uri, {"periodos": sel, "rows": int(len(pdf))}
