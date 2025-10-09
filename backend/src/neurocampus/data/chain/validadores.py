"""
Cadena de validación del dataset (robusta con normalización y coerción de tipos):
  0) normalización de encabezados (espacios <-> '_', acentos, ':'), sinónimos básicos
  1) columnas requeridas
  2) tipos (coherencia vs schema) con coerción previa
  3) dominio/valores permitidos (si aplica)
  4) duplicados (filas exactas)
  5) calidad: nulos, blancos

Entrada:
  - df: DataFrame (pandas o polars)
  - schema_path: ruta a JSON con formato nativo {"columns": [...]} o JSON Schema
Salida:
  - dict con summary (rows, errors, warnings, engine) e issues[]
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Union
import json
import re
import unicodedata
from pathlib import Path

from ..adapters.dataframe_adapter import columns, null_counts, row_count, dtype_of, _ENGINE

# ----------------------------
# Utilidades de schema
# ----------------------------

def _load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _expected_from_schema(schema: Dict[str, Any]) -> Tuple[List[str], Dict[str, Union[str, List[str]]], Dict[str, dict]]:
    """
    Extrae columnas, tipos y dominios desde:
      A) Formato nativo {"columns":[{name,dtype,domain{allowed|min|max}}]}
      B) JSON Schema {"properties":{col:{type,enum,minimum,maximum}}, "required":[...]}
         - Si 'required' está presente → esas son requeridas; si no, todas las keys.
         - 'type' puede ser string o lista (p.ej. ["string","null"]).
    """
    # --- (A) Formato nativo
    if "columns" in schema and isinstance(schema["columns"], list):
        cols: List[str] = []
        types: Dict[str, Union[str, List[str]]] = {}
        domains: Dict[str, dict] = {}
        for col in schema["columns"]:
            name = col["name"]
            cols.append(name)
            if "dtype" in col:
                types[name] = col["dtype"]
            if "domain" in col and isinstance(col["domain"], dict):
                d = col["domain"]
                nd: Dict[str, Any] = {}
                if "allowed" in d: nd["allowed"] = d["allowed"]
                if "min" in d: nd["min"] = d["min"]
                if "max" in d: nd["max"] = d["max"]
                if nd: domains[name] = nd
        return cols, types, domains

    # --- (B) JSON Schema
    if "properties" in schema and isinstance(schema["properties"], dict):
        props = schema["properties"]
        required = schema.get("required", [])
        cols = list(required) if required else list(props.keys())

        types: Dict[str, Union[str, List[str]]] = {}
        domains: Dict[str, dict] = {}
        for name, spec in props.items():
            js_type = spec.get("type", "string")
            types[name] = js_type  # puede ser str o list

            d: Dict[str, Any] = {}
            if "enum" in spec and isinstance(spec["enum"], list):
                d["allowed"] = list(spec["enum"])
            if "minimum" in spec: d["min"] = spec["minimum"]
            if "maximum" in spec: d["max"] = spec["maximum"]
            if d: domains[name] = d
        return cols, types, domains

    # --- Fallback sin restricciones
    return [], {}, {}

# ----------------------------
# Normalización de encabezados
# ----------------------------

_PUNCT_RE = re.compile(r"[:;,\.\(\)\[\]\{\}/\\]+")
_MULTI_WS_RE = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _slug_name(name: str) -> str:
    """
    Normaliza un nombre para comparación:
    - lower
    - quita acentos
    - elimina puntuación (: ; , . () [] {} / \)
    - reemplaza espacios por '_'
    - colapsa múltiples '_' consecutivos
    """
    s = _strip_accents(str(name)).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

# Sinónimos básicos comunes en nuestros datasets
_SYNONYMS = {
    "codigo_materia": {"codigo materia", "codigomateria", "cod_materia", "codigo_asignatura", "codigo asignatura"},
    "cedula_profesor": {"cedula profesor", "cedula", "cedula_docente", "cedula docente", "docente_id"},
    "grupo": {"grupo", "grupo_id"},
    "sugerencias": {"sugerencias", "observaciones", "comentarios", "sugerencias_"},
    "periodo": {"periodo", "periodo academico", "periodo_academico"},
}

def _build_alias_index(expected_cols: List[str]) -> Dict[str, str]:
    """
    Devuelve un índice {slug → nombre_canonico} para:
    - Cada esperado en varias formas (espacios <-> '_', con/ sin signos)
    - Sinónimos en _SYNONYMS
    """
    idx: Dict[str, str] = {}
    for canon in expected_cols:
        # nombre base
        idx[_slug_name(canon)] = canon
        # variante espacios <-> guiones bajos (ya cubierto por slug)
        # sinónimos
        key = _slug_name(canon)
        for syn in _SYNONYMS.get(key, set()):
            idx[_slug_name(syn)] = canon
        # quitar dos puntos finales (ej. Sugerencias:)
        if canon.endswith(":"):
            idx[_slug_name(canon[:-1])] = canon
        else:
            idx[_slug_name(canon + ":")] = canon
    return idx

def _rename_columns_by_alias(df, expected_cols: List[str]):
    """
    Renombra columnas del df al nombre canónico si el slug coincide
    con algún slug esperado o sinónimo; además descarta columnas 'Unnamed: N'.
    """
    cols_now = columns(df)
    alias_index = _build_alias_index(expected_cols)
    rename_map: Dict[str, str] = {}

    # descartar 'Unnamed: N' explícitamente
    drop_candidates = [c for c in cols_now if re.match(r"unnamed:\s*\d+", c.strip(), flags=re.I)]

    for c in cols_now:
        if c in drop_candidates:
            continue
        slug = _slug_name(c)
        if slug in alias_index:
            rename_map[c] = alias_index[slug]
        else:
            # también intentar sin ':' al final
            if slug.endswith("_"):
                slug2 = slug.rstrip("_")
                if slug2 in alias_index:
                    rename_map[c] = alias_index[slug2]

    # Aplicar renombre
    if _ENGINE == "polars":
        import polars as pl
        new_names = {}
        for c in cols_now:
            if c in drop_candidates:
                continue
            new_names[c] = rename_map.get(c, c)
        df = df.rename(new_names)
        # eliminar columnas a descartar
        for c in drop_candidates:
            if c in columns(df):
                df = df.drop(c)
    else:
        df = df.drop(columns=drop_candidates, errors="ignore").rename(columns=rename_map)

    return df

# ----------------------------
# Coerción de tipos
# ----------------------------

def _type_matches_one(expected: str, seen: str) -> bool:
    e = (expected or "").lower()
    s = (seen or "").lower()
    if e in ("number", "numeric", "float", "double"):
        return ("float" in s) or ("int" in s) or ("decimal" in s)
    if e in ("integer", "int"):
        return "int" in s and "uint" not in s
    if e in ("boolean", "bool"):
        return "bool" in s
    if e in ("string", "str"):
        return ("object" in s) or ("string" in s) or ("str" in s) or ("utf8" in s)
    if e in ("date", "datetime", "timestamp"):
        return ("date" in s) or ("datetime" in s) or ("timestamp" in s)
    return e in s

def _type_matches(expected: Union[str, List[str]], seen: str) -> bool:
    if isinstance(expected, list):
        # ignorar 'null' en la comparación
        ex = [x for x in expected if str(x).lower() != "null"]
        return any(_type_matches_one(str(e), seen) for e in ex) or (len(ex) == 0)
    return _type_matches_one(str(expected), seen)

def _coerce_types(df, expected_types: Dict[str, Union[str, List[str]]]):
    """
    Coacciona al tipo esperado cuando sea posible:
      - string: a 'string' (pandas) o pl.Utf8 (polars)
      - integer: a pandas 'Int64' / polars Int64
      - number: a float
      - boolean: a boolean
      - date/datetime: a datetime (si aplica)
    """
    if not expected_types:
        return df

    if _ENGINE == "polars":
        import polars as pl
        exprs = []
        for col, t in expected_types.items():
            if col not in columns(df):
                continue
            tgt = t[0] if isinstance(t, list) else t
            if str(tgt).lower() in ("string", "str"):
                exprs.append(pl.col(col).cast(pl.Utf8).alias(col))
            elif str(tgt).lower() in ("integer", "int"):
                exprs.append(pl.col(col).cast(pl.Int64).alias(col))
            elif str(tgt).lower() in ("number", "numeric", "float", "double"):
                exprs.append(pl.col(col).cast(pl.Float64).alias(col))
            elif str(tgt).lower() in ("boolean", "bool"):
                exprs.append(pl.col(col).cast(pl.Boolean).alias(col))
            # fechas: se puede añadir con pl.col(col).str.strptime(pl.Datetime, strict=False)
        if exprs:
            df = df.with_columns(exprs)
        return df

    # pandas
    import pandas as pd
    for col, t in expected_types.items():
        if col not in columns(df):
            continue
        tgt = t[0] if isinstance(t, list) else t
        tl = str(tgt).lower()
        try:
            if tl in ("string", "str"):
                df[col] = df[col].astype("string")
            elif tl in ("integer", "int"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif tl in ("number", "numeric", "float", "double"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif tl in ("boolean", "bool"):
                # mapear strings "true"/"false"/"1"/"0" automáticamente
                df[col] = df[col].astype("boolean")
            elif tl in ("date", "datetime", "timestamp"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            # si falla coerción, dejamos el dtype original (lo reportará BAD_TYPE)
            pass
    return df

# ----------------------------
# Checks
# ----------------------------

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

def check_types(df, expected_types: Dict[str, Union[str, List[str]]]) -> List[Dict[str, Any]]:
    issues = []
    if not expected_types:
        return issues
    cols_list = columns(df)
    for col, exp in expected_types.items():
        if col not in cols_list:
            continue
        seen = dtype_of(df, col)
        if not _type_matches(exp, str(seen)):
            issues.append({
                "code": "BAD_TYPE",
                "severity": "warning",
                "column": col,
                "row": None,
                "message": f"Tipo esperado {exp} vs observado {seen}"
            })
    return issues

def check_domains(df, domains: Dict[str, dict]) -> List[Dict[str, Any]]:
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
    issues = []
    if _ENGINE == "polars":
        dup_mask = df.is_duplicated()
        idxs = [i for i, d in enumerate(dup_mask) if d]
    else:
        dup_mask = df.duplicated(keep=False)
        idxs = [int(i) for i, v in dup_mask.items() if v]
    for r in idxs[:50]:
        issues.append({
            "code": "DUPLICATE_ROW",
            "severity": "warning",
            "column": None,
            "row": r,
            "message": "Fila duplicada detectada"
        })
    return issues

def check_quality(df) -> List[Dict[str, Any]]:
    issues = []
    n = row_count(df)
    if n == 0:
        return issues
    nc = null_counts(df)
    for col, cnt in nc.items():
        ratio = cnt / max(n, 1)
        if ratio >= 0.2:
            issues.append({
                "code": "HIGH_NULL_RATIO",
                "severity": "warning",
                "column": col,
                "row": None,
                "message": f"{ratio:.1%} nulos en {col}"
            })
    return issues

# ----------------------------
# Orquestación
# ----------------------------

def validate(df, schema_path: str) -> Dict[str, Any]:
    schema = _load_schema(schema_path)
    expected_cols, expected_types, domains = _expected_from_schema(schema)

    # 0) Normalizar encabezados (espacios <-> '_', acentos, ':', sinónimos)
    df = _rename_columns_by_alias(df, expected_cols)

    # 0.1) Coerción de tipos previa para reducir BAD_TYPE (ej. int64 -> string)
    df = _coerce_types(df, expected_types)

    # 1..5) Checks
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
