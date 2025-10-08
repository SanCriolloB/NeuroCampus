"""
Cadena de validación del dataset:
  1) columnas requeridas
  2) tipos (coherencia vs schema)
  3) dominio/valores permitidos (si aplica)
  4) duplicados (filas exactas)
  5) calidad: nulos, blancos

Entrada:
  - df: DataFrame (pandas o polars), ya cargado por FormatoAdapter
  - schema: dict cargado desde schemas/plantilla_dataset.schema.json

Salida:
  - dict con summary y issues detallados (para mapear al Pydantic de la API)
"""
from __future__ import annotations
from typing import Dict, List, Any
import json
from pathlib import Path

from ..adapters.dataframe_adapter import columns, null_counts, row_count, dtype_of, _ENGINE

def _load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _expected_from_schema(schema: Dict[str, Any]):
    """Extrae columnas y metadatos mínimos del json de esquema."""
    cols = []
    domains = {}
    types = {}
    for col in schema.get("columns", []):
        name = col["name"]
        cols.append(name)
        if "domain" in col:
            domains[name] = col["domain"]  # e.g., {"allowed": [...]} o rangos
        if "dtype" in col:
            types[name] = col["dtype"]
    return cols, types, domains

def check_required_columns(df, expected_cols: List[str]) -> List[Dict[str, Any]]:
    present = set(columns(df))
    issues = []
    for c in expected_cols:
        if c not in present:
            issues.append({
                "code": "MISSING_COLUMN",
                "severity": "error",
                "column": c,
                "row": None,
                "message": f"Columna requerida ausente: {c}"
            })
    return issues

def check_types(df, expected_types: Dict[str, str]) -> List[Dict[str, Any]]:
    issues = []
    for col, exp in expected_types.items():
        if col not in columns(df):
            continue
        seen = dtype_of(df, col)
        if str(exp).lower() not in str(seen).lower():
            issues.append({
                "code": "BAD_TYPE",
                "severity": "warning",
                "column": col,
                "row": None,
                "message": f"Tipo esperado {exp} vs observado {seen}"
            })
    return issues

def check_domains(df, domains: Dict[str, dict]) -> List[Dict[str, Any]]:
    """Valida dominios discretos (allowed) o rangos min/max si se definieron."""
    issues = []
    if not domains:
        return issues
    if _ENGINE == "polars":
        import polars as pl
    else:
        import pandas as pd

    for col, meta in domains.items():
        if col not in columns(df):
            continue
        allowed = meta.get("allowed")
        min_v = meta.get("min")
        max_v = meta.get("max")

        # iteración eficiente por valores únicos
        uniques = df[col].unique()
        if _ENGINE != "polars":
            uniques = uniques.dropna().tolist()
        else:
            uniques = [u for u in uniques if u is not None]

        if allowed:
            bad = [u for u in uniques if u not in allowed]
            for v in bad:
                issues.append({
                    "code": "DOMAIN_VIOLATION",
                    "severity": "error",
                    "column": col,
                    "row": None,
                    "message": f"Valor fuera de dominio en {col}: {v}"
                })
        if min_v is not None or max_v is not None:
            # Muestreo rápido de filas infractoras (hasta 20 para no saturar respuesta)
            if _ENGINE == "polars":
                mask = None
                if min_v is not None:
                    mask = (df[col] < min_v) if mask is None else (mask | (df[col] < min_v))
                if max_v is not None:
                    mask = (df[col] > max_v) if mask is None else (mask | (df[col] > max_v))
                infr = df.filter(mask).head(20)
                rows = [] if infr.is_empty() else list(range(0, len(infr)))
            else:
                s = df[col]
                mask = False
                if min_v is not None: mask |= (s < min_v)
                if max_v is not None: mask |= (s > max_v)
                rows = df[mask].head(20).index.tolist()
            for r in rows:
                issues.append({
                    "code": "RANGE_VIOLATION",
                    "severity": "error",
                    "column": col,
                    "row": int(r),
                    "message": f"Valor fuera de rango en {col}"
                })
    return issues

def check_duplicates(df) -> List[Dict[str, Any]]:
    """Detecta duplicados por fila completa (snapshot sencillo para D3)."""
    issues = []
    if _ENGINE == "polars":
        import polars as pl
        dup_mask = df.is_duplicated()
        idxs = [i for i, d in enumerate(dup_mask) if d]
    else:
        import pandas as pd
        dup_mask = df.duplicated(keep=False)
        idxs = [int(i) for i, v in dup_mask.items() if v]
    for r in idxs[:50]:  # limitar cantidad reportada
        issues.append({
            "code": "DUPLICATE_ROW",
            "severity": "warning",
            "column": None,
            "row": r,
            "message": "Fila duplicada detectada"
        })
    return issues

def check_quality(df) -> List[Dict[str, Any]]:
    """Reporta columnas con alta tasa de nulos como warning."""
    issues = []
    n = row_count(df)
    if n == 0:
        return issues
    nc = null_counts(df)
    for col, cnt in nc.items():
        ratio = cnt / max(n, 1)
        if ratio >= 0.2:  # umbral inicial
            issues.append({
                "code": "HIGH_NULL_RATIO",
                "severity": "warning",
                "column": col,
                "row": None,
                "message": f"{ratio:.1%} nulos en {col}"
            })
    return issues

def validate(df, schema_path: str) -> Dict[str, Any]:
    schema = _load_schema(schema_path)
    expected_cols, expected_types, domains = _expected_from_schema(schema)

    issues: List[Dict[str, Any]] = []
    issues += check_required_columns(df, expected_cols)
    issues += check_types(df, expected_types)
    issues += check_domains(df, domains)
    issues += check_duplicates(df)
    issues += check_quality(df)

    errors = sum(1 for i in issues if i["severity"] == "error")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    return {
        "summary": {
            "rows": row_count(df),
            "errors": errors,
            "warnings": warnings,
            "engine": _ENGINE
        },
        "issues": issues
    }