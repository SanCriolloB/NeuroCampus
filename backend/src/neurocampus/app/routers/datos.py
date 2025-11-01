# backend/src/neurocampus/app/routers/datos.py
"""
Router del contexto 'datos'.

Incluye:
- GET  /datos/esquema   → expone el esquema de la plantilla (lee JSON si existe o usa fallback).
- POST /datos/validar   → valida un archivo (csv/xlsx/parquet) con el wrapper unificado y devuelve 'sample'.
- POST /datos/upload    → ingesta real del archivo (a parquet por defecto) en datasets/{periodo}.parquet,
                          con control de sobrescritura.

Notas:
- Gating de formato (400) para extensiones no soportadas (.txt, etc.).
- Respuesta de /validar mantiene compat: añade `dataset_id` y `sample` (primeras filas).
- /upload escribe en <repo_root>/datasets/. Si ya existe y overwrite=False → 409.
- Si no hay motor parquet disponible, hace fallback a CSV (mismo nombre con .csv) y lo indica en stored_as.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import os

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

# Esquemas de respuesta del dominio
from ..schemas.datos import (
    DatosUploadResponse,
    EsquemaCol,
    EsquemaResponse,
)

# Facade que conecta con el wrapper unificado de validación
from ...data.facades.datos_facade import validar_archivo

router = APIRouter(tags=["datos"])


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _repo_root_from_here() -> Path:
    """Detecta la raíz del repo subiendo niveles desde este archivo."""
    here = Path(__file__).resolve()
    # [0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=repo_root
    return here.parents[5]


def _to_bool(x) -> bool:
    """Convierte valores sueltos de formularios a bool."""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "on", "t", "y"}


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

@router.get("/ping")
def ping() -> dict:
    return {"datos": "pong"}


# ---------------------------------------------------------------------------
# Esquema de plantilla
# ---------------------------------------------------------------------------

_FALLBACK_SCHEMA: Dict[str, Any] = {
    "version": "v0.3.0",
    "columns": [
        {"name": "periodo", "dtype": "string", "required": True},
        {"name": "codigo_materia", "dtype": "string", "required": True},
        {"name": "grupo", "dtype": "integer", "required": True},
        {"name": "pregunta_1", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_2", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_3", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_4", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_5", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_6", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_7", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_8", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_9", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "pregunta_10", "dtype": "number", "required": True, "range": [0, 50]},
        {"name": "Sugerencias:", "dtype": "string", "required": False, "max_len": 5000},
    ],
}

@router.get("/esquema", response_model=EsquemaResponse)
def get_esquema(version: Optional[str] = None) -> EsquemaResponse:
    """
    Lee schemas/plantilla_dataset.schema.json en la raíz del repo si existe;
    si no, devuelve un esquema mínimo de fallback.
    """
    import json

    schema_file = _repo_root_from_here() / "schemas" / "plantilla_dataset.schema.json"

    if schema_file.exists():
        try:
            data = json.loads(schema_file.read_text(encoding="utf-8"))
            props = data.get("properties", {})
            required = set(data.get("required", []))
            columns: List[EsquemaCol] = []

            for name, spec in props.items():
                js_type = spec.get("type", "string")
                if js_type == "number":
                    dtype = "number"
                elif js_type == "integer":
                    dtype = "integer"
                elif js_type == "boolean":
                    dtype = "boolean"
                else:
                    dtype = "string"

                col: Dict[str, Any] = {
                    "name": name,
                    "dtype": dtype,
                    "required": name in required,
                }

                if "minimum" in spec and "maximum" in spec:
                    col["range"] = [spec["minimum"], spec["maximum"]]

                if "maxLength" in spec:
                    col["max_len"] = int(spec["maxLength"])

                if isinstance(spec.get("enum"), list):
                    col["domain"] = [str(v) for v in spec["enum"]]

                columns.append(EsquemaCol(**col))

            return EsquemaResponse(version=str(data.get("version", "v0.3.0")), columns=columns)
        except Exception:
            # Cualquier error leyendo el JSON → usar fallback
            pass

    return EsquemaResponse(
        version=_FALLBACK_SCHEMA["version"],
        columns=[EsquemaCol(**c) for c in _FALLBACK_SCHEMA["columns"]],
    )


# ---------------------------------------------------------------------------
# Validación — wrapper unificado + compat con tests (sample) + gating de formato
# ---------------------------------------------------------------------------

def _first_rows_sample(raw: bytes, name: str, forced_fmt: Optional[str]) -> List[Dict[str, Any]]:
    """
    Lee mínimamente el archivo solo para construir 'sample' (primeras 5 filas),
    sin interferir con la validación del facade/wrapper.
    """
    fmt = (forced_fmt or "").strip().lower()
    lname = (name or "").lower()

    try:
        if fmt == "csv" or (not fmt and lname.endswith(".csv")):
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
        elif fmt == "xlsx" or (not fmt and lname.endswith(".xlsx")):
            df = pd.read_excel(io.BytesIO(raw))
        elif fmt == "parquet" or (not fmt and lname.endswith(".parquet")):
            df = pd.read_parquet(io.BytesIO(raw))
        else:
            return []
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.head(5).to_dict(orient="records")
    except Exception:
        pass
    return []


@router.post("/validar")
async def validar_datos(
    file: UploadFile = File(..., description="CSV/XLSX/Parquet"),
    dataset_id: str = Form(..., description="Identificador lógico del dataset (p. ej. 'docentes')"),
    fmt: Optional[str] = Form(None, description="Forzar lector: 'csv' | 'xlsx' | 'parquet' (opcional)"),
) -> Dict[str, Any]:
    """
    Valida un archivo de datos contra el validador unificado.
    - Gatea formato para responder 400 si no es csv/xlsx/parquet.
    - Enriquecer respuesta con `dataset_id` y `sample` (compat tests/herramientas).
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Archivo vacío o no leído.")

        name = file.filename or "upload"
        lower = name.lower()
        forced = (fmt or "").strip().lower()

        # 1) Gating de formato para retornar 400 en no soportados
        allowed = {"csv", "xlsx", "parquet"}
        ext_ok = (
            (forced in allowed) or
            lower.endswith(".csv") or
            lower.endswith(".xlsx") or
            lower.endswith(".parquet")
        )
        if not ext_ok:
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Use csv/xlsx/parquet o especifique 'fmt'.",
            )

        # 2) Construir 'sample' (compat con tests/herramientas que lo esperan)
        sample = _first_rows_sample(raw, name, forced)

        # 3) Delegar validación al facade (wrapper unificado)
        report = validar_archivo(
            fileobj=io.BytesIO(raw),
            filename=name,
            fmt=forced or None,
            dataset_id=dataset_id,
        )

        # 4) Enriquecer respuesta con dataset_id y sample por compatibilidad
        if isinstance(report, dict):
            report.setdefault("dataset_id", dataset_id)
            report.setdefault("sample", sample)

        return report

    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer el CSV con UTF-8. Intente especificar 'fmt' o convertir la codificación.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al validar: {e}")


# ---------------------------------------------------------------------------
# Ingesta (upload) real a datasets/{periodo}.parquet con control de overwrite
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=DatosUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV/XLSX/Parquet"),
    periodo: str = Form(..., description="Identificador de periodo (p. ej. '2024-2')"),
    dataset_id: str = Form(..., description="Compat alias (se ignora; usamos 'periodo')"),
    overwrite: bool = Form(False, description="Sobrescribir si existe"),
) -> DatosUploadResponse:
    """
    Ingesta real del dataset:
    - Lee CSV/XLSX/Parquet y lo escribe en <repo_root>/datasets/{periodo}.parquet (por defecto).
    - Si el fichero ya existe y overwrite=False → 409.
    - Si falta motor parquet (pyarrow/fastparquet), cae a CSV y lo informa en stored_as.
    """
    if not periodo:
        raise HTTPException(status_code=400, detail="periodo es requerido")

    # Gateo de formato por extensión del filename (simple y suficiente para el flujo actual)
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".parquet")):
        raise HTTPException(status_code=400, detail="Formato no soportado en upload. Use csv/xlsx/parquet.")

    # Directorio de destino: <repo_root>/datasets
    repo_root = _repo_root_from_here()
    outdir = repo_root / "datasets"
    outdir.mkdir(parents=True, exist_ok=True)

    # Rutas de salida
    parquet_path = outdir / f"{periodo}.parquet"
    csv_fallback_path = outdir / f"{periodo}.csv"

    # Control de sobrescritura
    if parquet_path.exists() or csv_fallback_path.exists():
        if not _to_bool(overwrite):
            raise HTTPException(
                status_code=409,
                detail=f"El dataset '{periodo}' ya existe. Activa 'overwrite' para reemplazarlo."
            )

    # Leer el archivo en memoria y a DataFrame (que el facade ya usa)
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Archivo vacío o no leído.")

        # Valernos del mismo lector que usa el facade (si lo prefieres)
        from ...data.adapters.formato_adapter import read_file
        df = read_file(io.BytesIO(raw), file.filename or "upload", explicit=None)

        if not isinstance(df, pd.DataFrame) or df.empty:
            # Permitimos subir aunque esté vacío, pero indicamos filas 0
            rows = 0
            # Aún así creamos un parquet vacío para mantener consistencia de pipeline
            try:
                df.to_parquet(parquet_path, index=False)  # podría fallar sin motor parquet
                stored_uri = f"localfs://neurocampus/datasets/{periodo}.parquet"
            except Exception:
                df.to_csv(csv_fallback_path, index=False)
                stored_uri = f"localfs://neurocampus/datasets/{periodo}.csv"
            return DatosUploadResponse(dataset_id=periodo, rows_ingested=rows, stored_as=stored_uri, warnings=[])

        # Escribir parquet; si no hay motor parquet, caer a CSV
        try:
            df.to_parquet(parquet_path, index=False)  # requiere pyarrow o fastparquet
            stored_uri = f"localfs://neurocampus/datasets/{periodo}.parquet"
        except (ImportError, ValueError, RuntimeError) as _e:
            # Fallback sin romper el flujo: persistimos como CSV
            df.to_csv(csv_fallback_path, index=False)
            stored_uri = f"localfs://neurocampus/datasets/{periodo}.csv"

        return DatosUploadResponse(
            dataset_id=periodo,
            rows_ingested=int(len(df)),
            stored_as=stored_uri,
            warnings=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir dataset: {e}")
