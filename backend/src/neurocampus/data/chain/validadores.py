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
    * Formato nativo admitido: {"columns":[{"name","dtype","domain"...}, ...]}
    * Formato JSON Schema admitido: {"properties": {...}, "required": [...]}

Salida:
  - dict con summary e issues detallados (para mapear al Pydantic de la API)
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
    """
    Extrae columnas, tipos y dominios desde:
      A) Formato nativo {"columns":[{name,dtype,domain{allowed|min|max}}]}
      B) JSON Schema {"properties":{col:{type,enum,minimum,maximum,...}}, "required":[...]}
         - Si 'required' está presente, esas columnas son las requeridas.
         - Si no, se asume que todas las claves de 'properties' son esperadas.
    """
    # --- (A) Formato nativo
    if "columns" in schema and isinstance(schema["columns"], list):
        cols: List[str] = []
        types: Dict[str, str] = {}
        domains: Dict[str, dict] = {}
        for col in schema["columns"]:
            name = col["name"]
            cols.append(name)
            if "dtype" in col:
                types[name] = col["dtype"]
            if "domain" in col:
                # Normalizamos a {allowed:[], min:?, max:?} si aplica
                d = col["domain"]
                nd = {}
                if isinstance(d, dict):
                    if "allowed" in d:
                        nd["allowed"] = d["allowed"]
                    if "min" in d:
                        nd["min"] = d["min"]
                    if "max" in d:
                        nd["max"] = d["max"]
                if nd:
                    domains[name] = nd
        return cols, types, domains

    # --- (B) JSON Schema
    if "properties" in schema and isinstance(schema["properties"], dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        cols = list(required) if required else list(props.keys())

        types: Dict[str, str] = {}
        domains: Dict[str, dict] = {}
        for name, spec in props.items():
            # type → mapeo directo a categorías lógicas
            js_type = spec.get("type", "string")
            types[name] = js_type  # 'number' | 'integer' | 'boolean' | 'string' | ...

            # dominios: enum / minimum / maximum
            d: Dict[str, Any] = {}
            if "enum" in spec and isinstance(spec["enum"], list):
                d["allowed"] = list(spec["enum"])
            if "minimum" in spec:
                d["min"] = spec["minimum"]
            if "maximum" in spec:
                d["max"] = spec["maximum"]
            if d:
                domains[name] = d

        return cols, types, domains

    # --- Fallback sin restricciones
    return [], {}, {}


def _type_matches(expected: str, seen: str) -> bool:
    """
    Compara 'expected' (p.ej., JSON Schema: number/integer/string/boolean/date)
    con el dtype observado del engine (pandas/polars) como texto.
    Heurístico pero estable para D3.
    """
    e = (expected or "").lower()
    s = (seen or "").lower()

    if e in ("number", "numeric", "float", "double"):
        return ("float" in s) or ("int" in s) or ("decimal" in s)
    if e in ("integer", "int"):
        return "int" in s and "uint" not in s  # int/Int64, evita confusión con unsigned
    if e in ("boolean", "bool"):
        return "bool" in s
    if e in ("string", "str"):
        # pandas: object/string ; polars: 'utf8'
        return ("object" in s) or ("string" in s) or ("str" in s) or ("utf8" in s)
    if e in ("date", "datetime", "timestamp"):
        return ("date" in s) or ("datetime" in s) or ("timestamp" in s)
    # Si no reconocemos, hacemos una comparación relajada
    return e in s


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
    if not expected_types:
        return issues
    cols_list = columns(df)
    for col, exp in expected_types.items():
        if col not in cols_list:
            continue
        seen = dtype_of(df, col)
        if not _type_matches(str(exp), str(seen)):
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

    for col, meta in domains.items():
        if col not in columns(df):
            continue
        allowed = meta.get("allowed")
        min_v = meta.get("min")
        max_v = meta.get("max")

        # valores únicos no nulos (ajuste pandas/polars)
        if _ENGINE == "polars":
            uniques = df[col].unique()
            # polars.Series -> iterables; filtramos None
            uniques = [u for u in uniques if u is not None]
        else:
            s = df[col]
            uniques = s.dropna().unique().tolist()

        if allowed is not None:
            bad = [u for u in uniques if u not in allowed]
            for v in bad:
                issues.append({
                    "code": "DOMAIN_VIOLATION",
                    "severity": "error",
                    "column": col,
                    "row": None,
                    "message": f"Valor fuera de dominio en {col}: {v}"
                })

        if (min_v is not None) or (max_v is not None):
            # Muestreo de filas infractoras (hasta 20)
            rows: List[int] = []
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
                if min_v is not None:
                    mask |= (s < min_v)
                if max_v is not None:
                    mask |= (s > max_v)
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
        dup_mask = df.is_duplicated()
        idxs = [i for i, d in enumerate(dup_mask) if d]
    else:
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
