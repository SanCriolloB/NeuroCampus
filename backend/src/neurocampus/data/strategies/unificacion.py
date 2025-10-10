# backend/src/neurocampus/data/strategies/unificacion.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import re
from pathlib import Path

# Adapter acordado
from ..adapters.almacen_adapter import AlmacenAdapter
# Lectura multi-formato y helpers de DF (como en días previos)
from ..adapters.formato_adapter import leer_archivo_generico  # csv/xlsx/parquet
from ..adapters.dataframe_adapter import to_pandas, to_parquet
# Normalización/validación usada en D2–D4
from ..chain.validadores import normalizar_encabezados

DEDUP_KEYS = ["periodo", "codigo_materia", "grupo", "cedula_profesor"]
PERIODO_RE = re.compile(r"^\d{4}-(1|2)$")  # AAAA-SEM

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
        items = self.store.ls(prefix)
        periodos = []
        for it in items:
            name = Path(it).name
            if PERIODO_RE.match(name):
                periodos.append(name)
        periodos.sort()
        return periodos

    def _leer_periodo(self, periodo: str):
        folder = f"datasets/{periodo}"
        # prioridad: parquet > csv > xlsx
        for candidate in ("data.parquet", "data.csv", "data.xlsx"):
            uri = f"{folder}/{candidate}"
            if self.store.exists(uri):
                df = leer_archivo_generico(self.store.open(uri, "rb" if uri.endswith(".parquet") or uri.endswith(".xlsx") else "r"))
                pdf = to_pandas(df)
                pdf.columns = normalizar_encabezados(list(pdf.columns))
                # si no trae 'periodo', fijarlo por robustez
                if "periodo" not in pdf.columns:
                    pdf["periodo"] = periodo
                return pdf
        raise FileNotFoundError(f"No se encontró dataset para periodo {periodo}")

    def _dedupe_concat(self, frames):
        import pandas as pd
        if not frames:
            raise ValueError("No hay frames para unificar")
        big = pd.concat(frames, ignore_index=True, copy=False)
        keys = [k for k in DEDUP_KEYS if k in big.columns]
        if keys:
            big = big.drop_duplicates(subset=keys)
        return big

    # -------- Metodologías --------
    def periodo_actual(self) -> Tuple[str, Dict[str, Any]]:
        periodos = self.listar_periodos()
        if not periodos:
            raise RuntimeError("No hay periodos en datasets/")
        ultimo = periodos[-1]
        pdf = self._leer_periodo(ultimo)
        out_uri = f"historico/periodo_actual/{ultimo}.parquet"
        self.store.makedirs("historico/periodo_actual")
        to_parquet(pdf, self.store.open(out_uri, "wb"))
        return out_uri, {"periodo": ultimo, "rows": int(len(pdf))}

    def acumulado(self) -> Tuple[str, Dict[str, Any]]:
        frames = [self._leer_periodo(p) for p in self.listar_periodos()]
        pdf = self._dedupe_concat(frames)
        out_uri = "historico/unificado.parquet"
        self.store.makedirs("historico")
        to_parquet(pdf, self.store.open(out_uri, "wb"))
        return out_uri, {"periodos": len(frames), "rows": int(len(pdf))}

    def ventana(self, ultimos: Optional[int] = None,
                desde: Optional[str] = None, hasta: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        periodos = self.listar_periodos()
        if ultimos:
            sel = periodos[-ultimos:]
        else:
            if not (desde and hasta):
                raise ValueError("Se requiere ultimos o (desde y hasta)")
            sel = [p for p in periodos if desde <= p <= hasta]
        frames = [self._leer_periodo(p) for p in sel]
        pdf = self._dedupe_concat(frames)
        tag = f"{sel[0]}_{sel[-1]}" if sel else "vacia"
        out_uri = f"historico/ventanas/unificado_{tag}.parquet"
        self.store.makedirs("historico/ventanas")
        to_parquet(pdf, self.store.open(out_uri, "wb"))
        return out_uri, {"periodos": sel, "rows": int(len(pdf))}
