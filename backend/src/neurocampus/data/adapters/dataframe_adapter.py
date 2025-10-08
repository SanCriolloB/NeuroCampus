"""
DataFrameAdapter: capa delgada para trabajar indistintamente con pandas o polars.
- Selección vía env: NC_DF_ENGINE ∈ {"pandas","polars"} (default: "pandas")
- Devuelve SIEMPRE un DataFrame del engine seleccionado.
- Provee utilidades mínimas para dtypes y conteos nulos.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

_ENGINE = os.getenv("NC_DF_ENGINE", "pandas").lower()

if _ENGINE == "polars":
    import polars as pl
    DF = pl.DataFrame
else:
    import pandas as pd
    DF = pd.DataFrame

def as_df(obj: Any) -> DF:
    """Convierte 'obj' a DF del engine configurado."""
    if _ENGINE == "polars":
        if isinstance(obj, pl.DataFrame):
            return obj
        return pl.DataFrame(obj)  # best effort
    else:
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame(obj)

def columns(df: DF) -> List[str]:
    return list(df.columns)

def row_count(df: DF) -> int:
    return df.height if _ENGINE == "polars" else len(df)

def null_counts(df: DF) -> Dict[str, int]:
    if _ENGINE == "polars":
        return {c: int(df[c].null_count()) for c in df.columns}
    else:
        return df.isna().sum().to_dict()

def dtype_of(df: DF, col: str) -> str:
    """Retorna un string estable del tipo (simplificado)."""
    if _ENGINE == "polars":
        return str(df.schema[col])
    else:
        return str(df[col].dtype)