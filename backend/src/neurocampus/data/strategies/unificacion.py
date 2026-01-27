# backend/src/neurocampus/data/strategies/unificacion.py
from __future__ import annotations

"""Estrategia de unificación histórica.

Este módulo pertenece al dominio **Datos** y su responsabilidad es crear
artefactos históricos reproducibles en disco, para consumo posterior por la
pestaña **Modelos**.

Cambios clave (DataTab)
-----------------------
- Soporta el layout actualizado: `datasets/<periodo>.parquet|csv|xlsx` (archivo plano)
  en lugar del layout anterior por carpeta `datasets/<periodo>/data.*`.
- Mantiene compatibilidad retroactiva con el layout por carpeta.
- Puede unificar tanto datasets "raw/processed" como datasets ya etiquetados
  (BETO) para producir `historico/unificado_labeled.parquet`.
"""

from typing import List, Optional, Dict, Any, Tuple
import re
from pathlib import Path

import pandas as pd

from ..adapters.almacen_adapter import AlmacenAdapter
from ..adapters.formato_adapter import read_file
from ..adapters.dataframe_adapter import as_df
from ..chain.validadores import normalizar_encabezados

DEDUP_KEYS = ["periodo", "codigo_materia", "grupo", "cedula_profesor"]
PERIODO_RE = re.compile(r"^\d{4}-(1|2|3)$")


class UnificacionStrategy:
    """Unifica datasets históricos y genera artefactos en /historico."""

    def __init__(self, base_uri: str = "localfs://."):
        self.store = AlmacenAdapter(base_uri)

    def listar_periodos_labeled(self, labeled_dir: str = "data/labeled") -> list[str]:
        """
        Lista datasets etiquetados (layout nuevo):
        - data/labeled/<dataset_id>_beto.parquet

        Retorna dataset_id (por ejemplo: 2025-1, 2024-2, evaluaciones_2025, etc.)
        """
        import re
        from pathlib import Path

        p = Path(labeled_dir)
        if not p.exists():
            return []

        out: list[str] = []
        for f in p.glob("*_beto.parquet"):
            name = f.name
            # si es periodo tipo 2025-1_beto.parquet
            m = re.match(r"^(?P<id>.+)_beto\.parquet$", name)
            if m:
                out.append(m.group("id"))

        return sorted(set(out))


    def _resolve_dataset_uri(self, periodo: str) -> str:
        """Resuelve la URI del dataset crudo (datasets/) para un periodo."""
        flat = [
            f"datasets/{periodo}.parquet",
            f"datasets/{periodo}.csv",
            f"datasets/{periodo}.xlsx",
        ]
        for uri in flat:
            if self.store.exists(uri):
                return uri

        folder = f"datasets/{periodo}"
        legacy = (f"{folder}/data.parquet", f"{folder}/data.csv", f"{folder}/data.xlsx")
        for uri in legacy:
            if self.store.exists(uri):
                return uri

        raise FileNotFoundError(f"No se encontró dataset para periodo {periodo} en datasets/")

    def _resolve_labeled_uri(self, periodo: str) -> str:
        """Resuelve la URI del dataset etiquetado (data/labeled) para un periodo."""
        candidates = [
            f"data/labeled/{periodo}_beto.parquet",
            f"data/labeled/{periodo}_teacher.parquet",
            f"data/labeled/{periodo}_beto.csv",
            f"data/labeled/{periodo}_teacher.csv",
        ]
        for uri in candidates:
            if self.store.exists(uri):
                return uri
        raise FileNotFoundError(f"No se encontró labeled para periodo {periodo} en data/labeled/")

    def _read_any(self, uri: str) -> pd.DataFrame:
        """Lee cualquier archivo (csv/xlsx/parquet) usando adapters y lo convierte a pandas."""
        with self.store.open(uri, "rb") as fh:
            df_like = read_file(fh, uri)
        pdf = as_df(df_like)

        if hasattr(pdf, "to_pandas"):
            try:
                pdf = pdf.to_pandas()
            except Exception:
                pdf = pd.DataFrame(pdf)

        pdf.columns = normalizar_encabezados(list(pdf.columns))
        return pdf

    def _leer_periodo(self, periodo: str) -> pd.DataFrame:
        """Lee dataset crudo de un periodo y asegura columna periodo."""
        uri = self._resolve_dataset_uri(periodo)
        pdf = self._read_any(uri)
        if "periodo" not in pdf.columns:
            pdf["periodo"] = periodo
        return pdf

    def _leer_labeled_periodo(self, periodo: str) -> pd.DataFrame:
        """Lee dataset etiquetado de un periodo y asegura columna periodo."""
        uri = self._resolve_labeled_uri(periodo)
        pdf = self._read_any(uri)
        if "periodo" not in pdf.columns:
            pdf["periodo"] = periodo
        return pdf

    def _dedupe_concat(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatena y elimina duplicados por claves canónicas cuando existan."""
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
        """Escribe parquet usando el store."""
        parent = str(Path(out_uri).parent).replace("\\", "/")
        if parent and parent not in ("", "."):
            self.store.makedirs(parent)
        with self.store.open(out_uri, "wb") as out_fh:
            df.to_parquet(out_fh, index=False)

    def periodo_actual(self) -> Tuple[str, Dict[str, Any]]:
        """Último periodo lexicográfico → historico/periodo_actual/<periodo>.parquet"""
        periodos = self.listar_periodos_labeled()
        if not periodos:
            raise RuntimeError("No hay periodos en datasets/")
        ultimo = periodos[-1]
        pdf = self._leer_periodo(ultimo)

        out_uri = f"historico/periodo_actual/{ultimo}.parquet"
        self._write_parquet(pdf, out_uri)
        return out_uri, {"periodo": ultimo, "rows": int(len(pdf))}

    def acumulado(self) -> Tuple[str, Dict[str, Any]]:
        """Concatena todos los periodos → historico/unificado.parquet"""
        periodos = self.listar_periodos_labeled()
        if not periodos:
            raise RuntimeError("No hay datasets etiquetados ...")
        frames = [self._leer_periodo(p) for p in periodos]
        pdf = self._dedupe_concat(frames)

        out_uri = "historico/unificado.parquet"
        self._write_parquet(pdf, out_uri)
        return out_uri, {"periodos": len(frames), "rows": int(len(pdf))}

    def ventana(
        self,
        ultimos: Optional[int] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Unifica una ventana de periodos → historico/ventanas/unificado_<tag>.parquet"""
        periodos = self.listar_periodos_labeled()
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

    def acumulado_labeled(self) -> Tuple[str, Dict[str, Any]]:
        """Unifica labeled disponibles → historico/unificado_labeled.parquet"""
        periodos = self.listar_periodos_labeled()
        frames: List[pd.DataFrame] = []
        skipped: List[str] = []

        for p in periodos:
            try:
                frames.append(self._leer_labeled_periodo(p))
            except FileNotFoundError:
                skipped.append(p)

        if not frames:
            raise RuntimeError(
                "No hay datasets etiquetados para unificar. "
                "Asegúrate de correr BETO al menos en un periodo."
            )

        pdf = self._dedupe_concat(frames)
        out_uri = "historico/unificado_labeled.parquet"
        self._write_parquet(pdf, out_uri)

        return out_uri, {"periodos": len(frames), "rows": int(len(pdf)), "skipped": skipped}
