"""
Router del contexto 'datos'.

Día 2: se añaden endpoints:
- GET  /datos/esquema  → expone el esquema de la plantilla (lee JSON o usa fallback).
- POST /datos/upload   → mock de carga (valida campos mínimos y responde metadatos).

Notas:
- Los sentimientos de comentarios (comentario.sent_pos/neg/neu) NO se suben en el dataset.
  Se calcularán en una etapa de PLN posterior (Día 6), por eso no aparecen como columnas requeridas.

Día 3: se añade endpoint:
- POST /datos/validar → valida un archivo CSV/XLSX/Parquet SIN almacenarlo y retorna
  un reporte con summary e issues (errores/advertencias) para que la UI lo presente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import io  # Día 3: para envolver bytes en BytesIO y mantener interfaz tipo archivo

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

# Modelos Pydantic del dominio 'datos' (definidos en schemas/datos.py)
from ..schemas.datos import (
    DatosUploadResponse,
    EsquemaCol,
    EsquemaResponse,
    DatosValidarResponse,  # Día 3: contrato de respuesta para /datos/validar
)

# Orquestador de validación (lectura + cadena de validadores)
from neurocampus.data.facades.datos_facade import validar_archivo  # Día 3

# Cada router es independiente y luego se registra en main.py
router = APIRouter()


@router.get("/ping")
def ping() -> dict:
    """
    Comprobación rápida de vida del router /datos.
    """
    return {"datos": "pong"}


# ---------------------------------------------------------------------------
# Día 2 — /datos/esquema y /datos/upload
# ---------------------------------------------------------------------------

# Fallback mínimo para que la UI pueda trabajar incluso si no existe el archivo
# schemas/plantilla_dataset.schema.json. Ajusta aquí si cambian los campos base.
_FALLBACK_SCHEMA: Dict[str, Any] = {
    "version": "v0.2.0",
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
def get_esquema(version: str | None = None) -> EsquemaResponse:
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
                # dtype simple: mapea 'number'→number; cualquier otro→string (ajusta si agregas types)
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

                # Dominios cerrados (si existieran enum/const)
                if "enum" in spec and isinstance(spec["enum"], list):
                    col["domain"] = [str(v) for v in spec["enum"]]

                columns.append(EsquemaCol(**col))

            return EsquemaResponse(version=str(data.get("version", "v0.2.0")), columns=columns)

        except Exception:
            # Si el archivo existe pero hay un problema de parseo, caemos al fallback
            pass

    # Fallback seguro
    return EsquemaResponse(
        version=_FALLBACK_SCHEMA["version"],
        columns=[EsquemaCol(**c) for c in _FALLBACK_SCHEMA["columns"]],
    )


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

    # En un próximo día:
    # - Validar cabeceras contra schemas/plantilla_dataset.schema.json
    # - Guardar el archivo como CSV/Parquet
    # - Registrar metadatos en un catálogo
    # - Manejar 'overwrite'

    # Respuesta de ejemplo compatible con el contrato
    stored_uri = f"localfs://neurocampus/datasets/{periodo}.parquet"
    return DatosUploadResponse(
        dataset_id=periodo,
        rows_ingested=0,  # placeholder hasta implementar persistencia real
        stored_as=stored_uri,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Día 3 — /datos/validar
# ---------------------------------------------------------------------------

@router.post("/validar", response_model=DatosValidarResponse)
async def validar_datos(
    file: UploadFile = File(...),
    fmt: str | None = Form(None),  # opcional: "csv" | "xlsx" | "parquet"
) -> DatosValidarResponse:
    """
    Valida un archivo de datos SIN almacenarlo.
    - Acepta CSV/XLSX/Parquet.
    - Si se especifica 'fmt', fuerza el lector; de lo contrario se infiere por extensión.
    - Devuelve KPIs de validación (rows, errors, warnings, engine) y el listado de issues.
    """
    try:
        # FastAPI entrega bytes; los envolvemos en un buffer compatible con .seek()
        raw = await file.read()
        buffer = io.BytesIO(raw)

        report = validar_archivo(buffer, file.filename, fmt)
        # 'report' ya cumple con el contrato DatosValidarResponse (summary + issues).
        return report  # FastAPI hará la validación contra el schema Pydantic
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validación falló: {e}")
