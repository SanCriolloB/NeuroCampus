# backend/src/neurocampus/app/routers/datos.py
"""
Router del contexto 'datos'.

Días previos:
- GET  /datos/esquema  → expone el esquema de la plantilla (lee JSON o usa fallback).
- POST /datos/upload   → mock de carga (valida campos mínimos y responde metadatos).

Día 5 (diagnóstico):
- POST /datos/validar → valida CSV/XLSX/Parquet SIN almacenar, usando neurocampus.validadores.run_validations
  y devuelve un dict {ok, sample, message?, missing?, extra?, dataset_id?}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import io
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

# Modelos Pydantic del dominio 'datos' (definidos en schemas/datos.py)
from ..schemas.datos import (
    DatosUploadResponse,
    EsquemaCol,
    EsquemaResponse,
)

# Adaptador de validación Día 5 (firma canónica)
# Estructura del proyecto: neurocampus.app.routers.datos → subir 2 niveles a neurocampus
from ...validadores import run_validations  # <- Día 5: usar adaptador local

router = APIRouter(tags=["datos"])


@router.get("/ping")
def ping() -> dict:
    """Comprobación rápida de vida del router /datos."""
    return {"datos": "pong"}


# ---------------------------------------------------------------------------
# Esquema de plantilla (Día 2)
# ---------------------------------------------------------------------------

# Fallback mínimo por si no existe 'schemas/plantilla_dataset.schema.json'
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
    """
    Intenta deducir la raíz del repositorio desde este archivo:
    .../backend/src/neurocampus/app/routers/datos.py → raíz = parents[5]
    """
    here = Path(__file__).resolve()
    # [0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=repo_root
    return here.parents[5]


@router.get("/esquema", response_model=EsquemaResponse)
def get_esquema(version: Optional[str] = None) -> EsquemaResponse:
    """
    Devuelve el esquema de la plantilla. Prioriza leer 'schemas/plantilla_dataset.schema.json'
    desde la raíz del repositorio. Si no existe o falla, usa un fallback en memoria.
    """
    import json

    repo_root = _repo_root_from_here()
    schema_file = repo_root / "schemas" / "plantilla_dataset.schema.json"

    if schema_file.exists():
        try:
            data = json.loads(schema_file.read_text(encoding="utf-8"))
            props = data.get("properties", {})
            required = set(data.get("required", []))
            columns: List[EsquemaCol] = []

            # Mapear JSON Schema → contrato ligero para la UI
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

                # Rango numérico
                if "minimum" in spec and "maximum" in spec:
                    col["range"] = [spec["minimum"], spec["maximum"]]

                # Longitud máxima para strings
                if "maxLength" in spec:
                    col["max_len"] = int(spec["maxLength"])

                # Dominios cerrados (enum)
                if "enum" in spec and isinstance(spec["enum"], list):
                    col["domain"] = [str(v) for v in spec["enum"]]

                columns.append(EsquemaCol(**col))

            return EsquemaResponse(version=str(data.get("version", "v0.3.0")), columns=columns)

        except Exception:
            # Si el archivo existe pero hay un problema de parseo, caemos al fallback
            pass

    # Fallback seguro
    return EsquemaResponse(
        version=_FALLBACK_SCHEMA["version"],
        columns=[EsquemaCol(**c) for c in _FALLBACK_SCHEMA["columns"]],
    )


# ---------------------------------------------------------------------------
# Mock de carga (Día 2)
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=DatosUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    periodo: str = Form(...),
    overwrite: bool = Form(False),
) -> DatosUploadResponse:
    """
    Mock de carga de dataset. No persiste aún.
    Reglas mínimas:
    - 'periodo' es obligatorio.
    - 'file' debe venir adjunto.
    - Si 'overwrite' es False y el dataset ya existiera, regresaría 409 (futuro).

    Importante:
    - Los campos derivados de PLN (comentario.sent_pos/neg/neu) NO se esperan en el archivo cargado.
      Se calcularán en una etapa posterior del pipeline.
    """
    if not periodo:
        raise HTTPException(status_code=400, detail="periodo es requerido")

    stored_uri = f"localfs://neurocampus/datasets/{periodo}.parquet"
    return DatosUploadResponse(
        dataset_id=periodo,
        rows_ingested=0,  # placeholder hasta implementar persistencia real
        stored_as=stored_uri,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Validación (Día 5 — diagnóstico mínimo)
# ---------------------------------------------------------------------------

@router.post("/validar")
async def validar_datos(
    file: UploadFile = File(..., description="CSV/XLSX/Parquet"),
    dataset_id: str = Form(..., description="Identificador lógico del dataset (p. ej. 'docentes')"),
    fmt: Optional[str] = Form(None, description="Forzar lector: 'csv' | 'xlsx' | 'parquet' (opcional)"),
) -> Dict[str, Any]:
    """
    Valida un archivo SIN almacenarlo, leyendo con pandas y llamando a run_validations(df, dataset_id=...).
    Respuesta (dict):
      { ok: bool, sample: [...], message?: str, missing?: [...], extra?: [...], dataset_id?: str }
    """
    try:
        raw = await file.read()

        # Selección del lector según fmt o extensión
        name = (file.filename or "").lower()
        force = (fmt or "").strip().lower()

        if force == "csv" or (not force and name.endswith(".csv")):
            # CSV: decodificamos a texto y usamos StringIO
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
        elif force == "xlsx" or (not force and name.endswith(".xlsx")):
            df = pd.read_excel(io.BytesIO(raw))
        elif force == "parquet" or (not force and name.endswith(".parquet")):
            df = pd.read_parquet(io.BytesIO(raw))
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado. Use csv/xlsx/parquet o especifique 'fmt'.")

        # Llamar al adaptador canónico (Día 5)
        report = run_validations(df, dataset_id=dataset_id)

        # 'report' ya es un dict con la forma esperada por la UI
        return report

    except HTTPException:
        raise
    except UnicodeDecodeError:
        # Error típico al leer CSV con codificación distinta a UTF-8
        raise HTTPException(status_code=400, detail="No se pudo leer el CSV con UTF-8. Intente especificar 'fmt' o convertir la codificación.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al validar: {e}")
