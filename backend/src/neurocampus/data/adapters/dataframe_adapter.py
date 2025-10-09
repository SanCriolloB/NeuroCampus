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
    """Retorna un tipo lógico estable: string|integer|number|boolean|date|datetime."""
    if _ENGINE == "polars":
        t = str(df.schema[col]).lower()  # ej: 'i64', 'f64', 'utf8', 'bool', 'date', 'datetime'
        if "utf8" in t: return "string"
        if t.startswith(("i", "u")): return "integer"
        if t.startswith("f"): return "number"
        if "bool" in t: return "boolean"
        if "date" == t: return "date"
        if "datetime" in t or "time" in t: return "datetime"
        return t
    else:
        t = str(df[col].dtype).lower()  # ej: 'int64', 'float64', 'string[python]', 'object', 'boolean'
        if "string" in t or "object" in t: return "string"
        if "int" in t and "uint" not in t: return "integer"
        if "float" in t or "decimal" in t: return "number"
        if "bool" in t: return "boolean"
        if "datetime" in t or "date" in t or "time" in t: return "datetime"
        return t