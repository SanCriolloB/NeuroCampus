"""
DatosFacade: punto único para validar un archivo subido.

Fix permanente (resolución robusta de schema):
- Si existe la variable de entorno NC_SCHEMA_PATH, se usa esa ruta.
- En caso contrario, se recorre hacia arriba desde este archivo buscando:
  'schemas/plantilla_dataset.schema.json'.
- Si no se encuentra, se intenta con algunos niveles típicos de proyecto.
- Si nada existe, se lanza FileNotFoundError con un mensaje claro.

Uso:
    export NC_SCHEMA_PATH="/ruta/absoluta/a/schemas/plantilla_dataset.schema.json"
    # o simplemente coloca la carpeta 'schemas/' en la raíz del repo.
"""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, Optional
import os

from ..adapters.formato_adapter import read_file
from ..chain.validadores import validate


_THIS = Path(__file__).resolve()


def _resolve_schema_path() -> Path:
    # 1) Override por variable de entorno
    env = os.getenv("NC_SCHEMA_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) Búsqueda ascendente: en cada directorio ancestro, probar ./schemas/plantilla_dataset.schema.json
    #    Comenzamos en el directorio del archivo y seguimos subiendo.
    for base in [_THIS.parent] + list(_THIS.parents):
        cand = base / "schemas" / "plantilla_dataset.schema.json"
        if cand.exists():
            return cand

    # 3) Heurística adicional: subir varios niveles "típicos" por si el layout cambia
    for up in range(3, 8):
        try:
            cand = _THIS.parents[up] / "schemas" / "plantilla_dataset.schema.json"
            if cand.exists():
                return cand
        except IndexError:
            break

    # 4) Si no se encontró, instruimos cómo resolverlo
    raise FileNotFoundError(
        "No se encontró 'schemas/plantilla_dataset.schema.json'. "
        "Define la variable de entorno NC_SCHEMA_PATH con la ruta absoluta del schema, "
        "o crea la carpeta 'schemas/' en la raíz del repositorio."
    )


SCHEMA_PATH = _resolve_schema_path()


def validar_archivo(fileobj: BinaryIO, filename: str, fmt: Optional[str] = None) -> dict:
    """
    Lee el archivo (CSV/XLSX/Parquet) usando el adapter y ejecuta la cadena de validación
    contra el schema detectado/resuelto.
    """
    df = read_file(fileobj, filename, explicit=fmt)
    report = validate(df, str(SCHEMA_PATH))
    return report
