# backend/src/neurocampus/data/facades/datos_facade.py
"""
DatosFacade — punto único para validar un archivo subido (para el endpoint /datos/validar).

✅ Paso 2 (Día 6): Conectar el *wrapper* unificado de validación
    - Se usa `neurocampus.validation.validation_wrapper.run_validations(df, dataset_id)`
    - Mantiene compatibilidad con tu *chain* si éste requiere `schema_path`:
      el facade intentará resolver el schema y lo pasará como **kwarg** opcional.
      (El wrapper reenvía kwargs a tu función real de chain).

Entrada:
    - fileobj  : binario del archivo subido (CSV/XLSX/Parquet)
    - filename : nombre del archivo (para inferir formato por extensión)
    - fmt      : "csv" | "xlsx" | "parquet" (opcional, fuerza el lector)
    - dataset_id : identificador lógico para reglas específicas (opcional)

Salida (dict):
    {
      "ok": bool,  # compat hacia atrás; derivado de issues==0 si no lo provee el validador
      "summary": { "rows": int, "errors": int, "warnings": int, "engine": str, "message"?: str },
      "issues": [ { "type": "error"|"warning", "message": str, "column"?: str, "row"?: int }, ... ]
    }

Uso del schema:
    - Si existe la variable de entorno NC_SCHEMA_PATH, se usa esa ruta.
    - En caso contrario, se intenta descubrir ./schemas/plantilla_dataset.schema.json
      subiendo desde este archivo. Si no existe, simplemente NO se pasa `schema_path`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Optional

from ..adapters.formato_adapter import read_file
from neurocampus.validation.validation_wrapper import run_validations


_THIS = Path(__file__).resolve()


# ----------------------------------------------------------------------------- #
# Resolución opcional del schema (compatibilidad con chains que lo necesiten)
# ----------------------------------------------------------------------------- #
def _resolve_schema_path_optional() -> Optional[Path]:
    """
    Intenta resolver el schema. Si no existe, devuelve None (no rompe flujo).
    """
    # 1) Override por variable de entorno
    env = os.getenv("NC_SCHEMA_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) Búsqueda ascendente: ./schemas/plantilla_dataset.schema.json
    for base in [_THIS.parent] + list(_THIS.parents):
        cand = base / "schemas" / "plantilla_dataset.schema.json"
        if cand.exists():
            return cand

    # 3) Heurística adicional: probar unos niveles más arriba por si varía el layout
    for up in range(3, 8):
        try:
            cand = _THIS.parents[up] / "schemas" / "plantilla_dataset.schema.json"
            if cand.exists():
                return cand
        except IndexError:
            break

    # No encontrado: devolver None (el validador puede no requerir schema)
    return None


# ----------------------------------------------------------------------------- #
# API pública usada por el router /datos/validar
# ----------------------------------------------------------------------------- #
def validar_archivo(
    fileobj: BinaryIO,
    filename: str,
    fmt: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> dict:
    """
    Lee el archivo (CSV/XLSX/Parquet) y ejecuta la validación unificada:
      run_validations(df, dataset_id, **kwargs_opcionales)

    - Si se encuentra un schema, se pasa como kwarg `schema_path` para compatibilidad.
    - El wrapper devuelve { ok, summary, issues } con forma estándar del endpoint.
    """
    # 1) Leer a DataFrame usando el adapter del proyecto
    df = read_file(fileobj, filename, explicit=fmt)

    # 2) Resolver schema opcionalmente (compat con implementaciones previas)
    schema_path = _resolve_schema_path_optional()
    kwargs = {}
    if schema_path:
        kwargs["schema_path"] = str(schema_path)

    # 3) Ejecutar el validador unificado
    report = run_validations(df, dataset_id=dataset_id, **kwargs)

    # 4) Devolver tal cual (contracto del endpoint /datos/validar)
    return report


__all__ = ["validar_archivo"]
