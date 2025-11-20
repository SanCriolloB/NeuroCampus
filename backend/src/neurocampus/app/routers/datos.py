# backend/src/neurocampus/app/routers/datos.py
"""
Router del contexto **datos**.

Responsabilidades principales
-----------------------------
- GET  /datos/ping
    → Ping sencillo para monitoreo del contexto.

- GET  /datos/esquema
    → Expone el esquema esperado de la plantilla de evaluaciones.
      Lee `schemas/plantilla_dataset.schema.json` en la raíz del repo si existe;
      si no, usa un esquema mínimo de fallback.

- POST /datos/validar
    → Valida un archivo (csv/xlsx/parquet) usando el *wrapper* unificado
      (`datos_facade.validar_archivo`) y devuelve un reporte estructurado
      junto con un `sample` de las primeras filas.

- POST /datos/upload
    → Ingesta real del archivo en `datasets/{periodo}.parquet` (o `.csv`
      como fallback), con control de sobrescritura y metadatos mínimos.

- GET  /datos/resumen
    → Devuelve KPIs básicos del dataset (filas, columnas, periodos, docentes,
      asignaturas, etc.) más un resumen por columna para la pestaña **Datos**
      del frontend.

- GET  /datos/sentimientos
    → Devuelve la distribución de sentimientos (global, por docente y por
      asignatura) leída desde los datasets etiquetados por BETO/teacher.
      Esta información alimenta las gráficas de la pestaña **Datos**.

Notas de diseño
---------------
- Este router es deliberadamente delgado: delega la lógica de lectura y
  construcción de respuestas a un módulo de dominio (`neurocampus.data.datos_dashboard`)
  para facilitar su testeo y posterior documentación.
- La estructura de respuestas está tipada mediante los esquemas Pydantic
  definidos en `app/schemas/datos.py`, pensando en la futura documentación
  automática (OpenAPI, Sphinx, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import os

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Query

# Esquemas de respuesta del dominio de datos
from ..schemas.datos import (
    DatosUploadResponse,
    EsquemaCol,
    EsquemaResponse,
    DatasetResumenResponse,
    DatasetSentimientosResponse,
)

# Facade que conecta con el wrapper unificado de validación
from ...data.facades.datos_facade import validar_archivo

# Helpers de dominio para el "dashboard" de la pestaña Datos
from ...data.datos_dashboard import (
    load_processed_dataset,
    build_dataset_resumen,
    load_labeled_dataset,
    build_sentimientos_resumen,
)

router = APIRouter(tags=["datos"])


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _repo_root_from_here() -> Path:
    """
    Detecta la raíz del repo subiendo niveles desde este archivo.

    El layout esperado es:
        <repo_root>/
          backend/
            src/
              neurocampus/
                app/
                  routers/
                    datos.py  ← este archivo

    Desde `routers/` hay que subir 5 niveles para llegar a la raíz.
    """
    here = Path(__file__).resolve()
    # [0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=repo_root
    return here.parents[5]


def _to_bool(x) -> bool:
    """
    Convierte valores provenientes de formularios HTML a bool.

    Acepta:
    - True/False directos.
    - Cadenas tipo "true", "1", "yes", "on", "t", "y" (case-insensitive).
    """
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
    """Pequeño endpoint de salud para el contexto `datos`."""
    return {"datos": "pong"}


# ---------------------------------------------------------------------------
# Esquema de plantilla
# ---------------------------------------------------------------------------

# Esquema de respaldo usado cuando no existe el JSON en /schemas.
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
    Devuelve el esquema esperado de la plantilla de evaluaciones.

    - Si existe el archivo JSON `schemas/plantilla_dataset.schema.json` en la
      raíz del repo, se parsea y se traduce a una lista de `EsquemaCol`.
    - Si hay cualquier error leyendo ese archivo, se utiliza `_FALLBACK_SCHEMA`.

    El parámetro `version` está reservado para futuras extensiones (por ahora
    se ignora y se devuelve siempre la versión detectada en el JSON o la
    versión por defecto del fallback).
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
            # Cualquier error leyendo el JSON → usar fallback sin romper el flujo.
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
    Construye un `sample` con las primeras filas del archivo subido.

    Esta función es independiente del *facade* de validación y solo
    lee mínimamente el archivo para devolver las primeras 5 filas en
    formato JSON (lista de dicts). Esto permite:
    - Mostrar una vista previa en el frontend.
    - Mantener compatibilidad con tests y herramientas ya escritas.

    Si el formato no se reconoce o hay cualquier error, se devuelve
    una lista vacía.
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
        # El sample es auxiliar; si falla no bloqueamos la validación.
        pass
    return []


@router.post("/validar")
async def validar_datos(
    file: UploadFile = File(..., description="CSV/XLSX/Parquet con evaluaciones"),
    dataset_id: str = Form(..., description="Identificador lógico del dataset (p. ej. 'docentes')"),
    fmt: Optional[str] = Form(
        None,
        description="Forzar lector: 'csv' | 'xlsx' | 'parquet' (opcional; normalmente se infiere por extensión)",
    ),
) -> Dict[str, Any]:
    """
    Valida un archivo de datos contra el validador unificado.

    Flujo:
    - Gatea el formato (extensión/`fmt`) para responder 400 si no es csv/xlsx/parquet.
    - Construye un `sample` con las primeras filas (para UI y tests).
    - Invoca el `validar_archivo` del *facade* (wrapper unificado).
    - Enriquecer la respuesta del wrapper con `dataset_id` y `sample`.

    Importante:
    - No persiste el archivo en disco; para eso está `/datos/upload`.
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
            (forced in allowed)
            or lower.endswith(".csv")
            or lower.endswith(".xlsx")
            or lower.endswith(".parquet")
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
            detail="No se pudo leer el CSV con UTF-8. "
            "Intente especificar 'fmt' o convertir la codificación.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al validar: {e}")


# ---------------------------------------------------------------------------
# Ingesta (upload) real a datasets/{periodo}.parquet con control de overwrite
# ---------------------------------------------------------------------------


@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=DatosUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="Archivo CSV/XLSX/Parquet ya validado"),
    periodo: str = Form(..., description="Identificador de periodo (p. ej. '2024-2')"),
    dataset_id: str = Form(
        ..., description="Alias por compatibilidad; actualmente se ignora y se usa 'periodo' como dataset_id"
    ),
    overwrite: bool = Form(False, description="Sobrescribir el dataset si ya existe en disco"),
) -> DatosUploadResponse:
    """
    Ingesta real del dataset.

    - Lee CSV/XLSX/Parquet y lo escribe en `<repo_root>/datasets/{periodo}.parquet` (por defecto).
    - Si el fichero ya existe y `overwrite=False` → 409 Conflict.
    - Si falta motor parquet (pyarrow/fastparquet), cae a CSV (`{periodo}.csv`)
      y lo informa en `stored_as`.

    Este endpoint se invoca normalmente después de una validación exitosa
    con `/datos/validar`.
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
                detail=f"El dataset '{periodo}' ya existe. Activa 'overwrite' para reemplazarlo.",
            )

    # Leer el archivo en memoria y a DataFrame (usando el mismo adapter que el facade)
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Archivo vacío o no leído.")

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
        except (ImportError, ValueError, RuntimeError):
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


# ---------------------------------------------------------------------------
# Resumen de dataset para la pestaña "Datos" (/datos/resumen)
# ---------------------------------------------------------------------------


@router.get(
    "/resumen",
    response_model=DatasetResumenResponse,
    summary="Devuelve KPIs generales del dataset y un resumen de columnas.",
)
def get_dataset_resumen(
    dataset_id: str = Query(
        ...,
        alias="dataset",
        description="Identificador lógico del dataset (por ejemplo, 'evaluaciones_2025')",
    ),
) -> DatasetResumenResponse:
    """
    Devuelve un resumen general del dataset para la UI de la pestaña **Datos**.

    Fuente de datos:
    - Intenta leer el dataset "procesado" usando los helpers de
      `neurocampus.data.datos_dashboard`, típicamente desde:
        - `data/processed/{dataset_id}.parquet`
        - o `datasets/{dataset_id}.parquet`, según configuración.

    Contenido de la respuesta:
    - `n_rows`, `n_cols`
    - lista de `periodos` detectados (si existe la columna `periodo`)
    - número de docentes y asignaturas (si existen columnas compatibles)
    - lista de `columns` con nombre, tipo lógico y muestra de valores
    """
    try:
        df = load_processed_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró dataset procesado para '{dataset_id}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer dataset '{dataset_id}': {e}",
        )

    return build_dataset_resumen(df, dataset_id)


# ---------------------------------------------------------------------------
# Resumen de sentimientos BETO (/datos/sentimientos)
# ---------------------------------------------------------------------------


@router.get(
    "/sentimientos",
    response_model=DatasetSentimientosResponse,
    summary="Devuelve distribución de sentimientos global y por docente/asignatura.",
)
def get_dataset_sentimientos(
    dataset_id: str = Query(
        ...,
        alias="dataset",
        description="Identificador lógico del dataset (mismo que se usó para BETO)",
    ),
) -> DatasetSentimientosResponse:
    """
    Devuelve la distribución de sentimientos sobre los comentarios del dataset.

    Fuente de datos:
    - Lee el dataset etiquetado (`data/labeled/{dataset_id}_beto.parquet`
      o `{dataset_id}_teacher.parquet`) usando los helpers de dominio.

    Contenido de la respuesta:
    - `total_comentarios` considerados (filas con comentario no vacío).
    - `global_counts`: distribución global de sentimientos (neg/neu/pos).
    - `por_docente`: lista de grupos con conteos por sentimiento.
    - `por_asignatura`: idem pero agrupado por asignatura.

    Esta respuesta está pensada para alimentar tres tipos de visualizaciones:
    - Barra/pastel global (Positivo/Neutro/Negativo).
    - Barras apiladas por docente.
    - Barras apiladas por asignatura.
    """
    try:
        df = load_labeled_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró dataset etiquetado (BETO/teacher) para '{dataset_id}'",
        )
    except KeyError as e:
        # Falta la columna de sentimiento esperada
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer dataset etiquetado '{dataset_id}': {e}",
        )

    return build_sentimientos_resumen(df, dataset_id)
