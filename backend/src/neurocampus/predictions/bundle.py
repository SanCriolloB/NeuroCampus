"""
neurocampus.predictions.bundle
==============================

Contrato y utilidades para el "predictor bundle" persistido en un run.

Objetivo (P2.1)
---------------
Permitir predicción reproducible y desacoplada del proceso de entrenamiento:

- En P0/P1 entrenamos y guardamos metrics/history.
- En P2.1 guardamos también un bundle mínimo en `artifacts/runs/<run_id>/`:
  - `predictor.json`  (metadata/contrato)
  - `model.bin`       (estado serializado del modelo; implementación por estrategia)
  - `preprocess.json` (opcional: mapeos, normalizaciones, etc.)

Este módulo NO implementa inferencia final aún; define contratos y
helpers de lectura/escritura que el service usará.

Compatibilidad
--------------
- No modifica P0/P1.
- Si un run no tiene bundle, P2 deberá responder 404/422 al intentar predecir.

Notas de serialización
----------------------
Por ahora se define un formato genérico. Las estrategias pueden:
- usar `joblib`/`pickle` para `model.bin`,
- o guardar pesos en otro formato, pero deben declarar `format` en predictor.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


PREDICTOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictorBundlePaths:
    """Paths canónicos del bundle dentro de un run."""
    run_dir: Path
    predictor_json: Path
    model_bin: Path
    preprocess_json: Path


def bundle_paths(run_dir: Path) -> PredictorBundlePaths:
    """Retorna los paths estándar del predictor bundle para un `run_dir`."""
    rd = Path(run_dir).expanduser().resolve()
    return PredictorBundlePaths(
        run_dir=rd,
        predictor_json=rd / "predictor.json",
        model_bin=rd / "model.bin",
        preprocess_json=rd / "preprocess.json",
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Escribe JSON UTF-8 con indent estable (para diffs y auditoría)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """Lee JSON UTF-8 y retorna dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_predictor_manifest(
    *,
    run_id: str,
    dataset_id: str,
    model_name: str,
    task_type: str,
    input_level: str,
    target_col: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye el `predictor.json` (manifest del bundle).

    Campos mínimos:
    - schema_version: versión del contrato del manifest
    - run_id / dataset_id / model_name
    - task_type: classification|regression (o equivalente)
    - input_level: row|pair
    - target_col: columna objetivo usada al entrenar (si aplica)
    - format: formato de model.bin (default "pickle")

    Args:
        run_id: id del run.
        dataset_id: dataset_id.
        model_name: nombre del modelo.
        task_type: tipo de tarea.
        input_level: nivel de entrada.
        target_col: columna target.
        extra: dict opcional para extender sin romper contrato.

    Returns:
        Dict JSON-serializable.
    """
    payload: Dict[str, Any] = {
        "schema_version": PREDICTOR_SCHEMA_VERSION,
        "run_id": str(run_id),
        "dataset_id": str(dataset_id),
        "model_name": str(model_name),
        "task_type": str(task_type),
        "input_level": str(input_level),
        "target_col": str(target_col) if target_col else None,
        "format": "pickle",
    }
    if extra:
        payload["extra"] = extra
    return payload
