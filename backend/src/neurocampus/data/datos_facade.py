"""
DatosFacade: punto único para validar un archivo subido.
"""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, Optional

from ..adapters.formato_adapter import read_file
from ..chain.validadores import validate

BASE_DIR = Path(__file__).resolve().parents[3]  # .../backend/src/neurocampus
SCHEMA_PATH = BASE_DIR.parents[2] / "schemas" / "plantilla_dataset.schema.json"
# ^ Ajuste: BASE_DIR -> backend/src/neurocampus; parents[2] = backend; + /schemas a nivel repo

def validar_archivo(fileobj: BinaryIO, filename: str, fmt: Optional[str] = None) -> dict:
    df = read_file(fileobj, filename, explicit=fmt)
    report = validate(df, str(SCHEMA_PATH))
    return report