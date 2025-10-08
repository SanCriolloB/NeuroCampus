"""
FormatoAdapter: lectura robusta de CSV / XLSX / Parquet hacia el engine configurado.
- Detecta por extensión o parámetro explícito.
- No guarda; solo lee a memoria para validar.
"""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, Optional
import os

from .dataframe_adapter import as_df, _ENGINE

def infer_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt"): return "csv"
    if ext in (".xlsx", ".xls"): return "xlsx"
    if ext in (".parquet", ".pq"): return "parquet"
    # fallback seguro
    return "csv"

def read_file(fileobj: BinaryIO, filename: str, explicit: Optional[str]=None):
    fmt = (explicit or infer_format(filename)).lower()
    fileobj.seek(0)

    if _ENGINE == "polars":
        import polars as pl
        if fmt == "csv":
            return pl.read_csv(fileobj)
        if fmt == "xlsx":
            # polars no lee xlsx nativamente → usar pandas y convertir
            import pandas as pd
            df = pd.read_excel(fileobj)
            return pl.from_pandas(df)
        if fmt == "parquet":
            return pl.read_parquet(fileobj)
        raise ValueError(f"Formato no soportado: {fmt}")

    # pandas
    import pandas as pd
    if fmt == "csv":
        return pd.read_csv(fileobj)
    if fmt == "xlsx":
        return pd.read_excel(fileobj)
    if fmt == "parquet":
        return pd.read_parquet(fileobj)
    raise ValueError(f"Formato no soportado: {fmt}")