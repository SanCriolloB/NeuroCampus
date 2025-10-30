# backend/src/neurocampus/app/routers/datos.py
"""
Router del contexto 'datos'.

Días previos:
- GET  /datos/esquema  → expone el esquema de la plantilla (lee JSON o usa fallback).
- POST /datos/upload   → mock de carga (valida campos mínimos y responde metadatos).

Día 6 (paso 2/3): /datos/validar debe usar el facade que conecta con el wrapper unificado,
devolviendo el contrato estándar: {"ok", "summary", "issues", ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..schemas.datos import (
    DatosUploadResponse,
    EsquemaCol,
    EsquemaResponse,
)

# ✅ Nuevo: usamos el facade que llama al wrapper unificado
from ...data.facades.datos_facade import validar_archivo

router = APIRouter(tags=["datos"])


@router.get("/ping")
def ping() -> dict:
    return {"datos": "pong"}


# ---------------------------------------------------------------------------
# Esquema de plantilla (igual que antes)
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


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    # [0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=repo_root
    return here.parents[5]


@router.get("/esquema", response_model=EsquemaResponse)
def get_esquema(version: Optional[str] = None) -> EsquemaResponse:
    import json

    repo_root = _repo_root_from_here()
    schema_file = repo_root / "schemas" / "plantilla_dataset.schema.json"

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

                if "enum" in spec and isinstance(spec["enum"], list):
                    col["domain"] = [str(v) for v in spec["enum"]]

                columns.append(EsquemaCol(**col))

            return EsquemaResponse(version=str(data.get("version", "v0.3.0")), columns=columns)
        except Exception:
            pass

    return EsquemaResponse(
        version=_FALLBACK_SCHEMA["version"],
        columns=[EsquemaCol(**c) for c in _FALLBACK_SCHEMA["columns"]],
    )


# ---------------------------------------------------------------------------
# Mock de carga (igual que antes)
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=DatosUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    periodo: str = Form(...),
    overwrite: bool = Form(False),
) -> DatosUploadResponse:
    if not periodo:
        raise HTTPException(status_code=400, detail="periodo es requerido")

    stored_uri = f"localfs://neurocampus/datasets/{periodo}.parquet"
    return DatosUploadResponse(
        dataset_id=periodo,
        rows_ingested=0,
        stored_as=stored_uri,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Validación (Día 6) — ahora con wrapper unificado (summary + issues)
# ---------------------------------------------------------------------------

@router.post("/validar")
async def validar_datos(
    file: UploadFile = File(..., description="CSV/XLSX/Parquet"),
    dataset_id: str = Form(..., description="Identificador lógico del dataset (p. ej. 'docentes')"),
    fmt: Optional[str] = Form(None, description="Forzar lector: 'csv' | 'xlsx' | 'parquet' (opcional)"),
) -> Dict[str, Any]:
    """
    Lee el archivo y delega en el facade (que llama al wrapper unificado).
    Devuelve el contrato estándar esperado por los tests:
      { "ok": bool, "summary": {...}, "issues": [...] }
    """
    try:
        raw = await file.read()
        # Pasamos un buffer binario al facade; éste usará el adapter para pandas
        buf = io.BytesIO(raw)
        report = validar_archivo(
            fileobj=buf,
            filename=file.filename or "upload",
            fmt=fmt,
            dataset_id=dataset_id,
        )

        # Por conveniencia, asegurar que el dataset_id viaje en la respuesta
        if isinstance(report, dict) and "dataset_id" not in report:
            report["dataset_id"] = dataset_id

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
