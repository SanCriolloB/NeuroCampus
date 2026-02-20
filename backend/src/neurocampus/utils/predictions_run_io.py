"""neurocampus.utils.predictions_run_io
======================================

IO para predicciones persistidas (batch e individual) de la pestaña Predicciones.

Layout de artifacts
-------------------
Este módulo define y centraliza el layout para artefactos de predicción:

    artifacts/predictions/<dataset_id>/score_docente/<pred_run_id>/
        predicciones.parquet
        meta.json

Convención de ``pred_run_id``
------------------------------
Formato ``pred_YYYYMMDD_HHMMSS`` para ordenamiento lexicográfico
cronológico directo sin dependencias externas.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neurocampus.utils.paths import artifacts_dir


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Sub-ruta fija para la familia score_docente dentro de predictions/.
_FAMILY_SEGMENT: str = "score_docente"


# ---------------------------------------------------------------------------
# Funciones de escritura
# ---------------------------------------------------------------------------

def create_pred_run_dir(dataset_id: str) -> Tuple[str, Path]:
    """Crea el directorio para un nuevo run de predicción.

    El directorio se crea en::

        artifacts/predictions/<dataset_id>/score_docente/<pred_run_id>/

    Args:
        dataset_id: Identificador del dataset (ej. ``"2024-2"``).

    Returns:
        Tupla ``(pred_run_id, path)`` donde ``pred_run_id`` es el nombre
        del directorio creado y ``path`` es el :class:`~pathlib.Path` absoluto.
    """
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_run_id = f"pred_{ts}"
    out_dir = (
        artifacts_dir()
        / "predictions"
        / str(dataset_id)
        / _FAMILY_SEGMENT
        / pred_run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return pred_run_id, out_dir


def write_pred_meta(out_dir: Path, meta: Dict[str, Any]) -> None:
    """Persiste ``meta.json`` en el directorio del run de predicción.

    Args:
        out_dir: Directorio del run (retornado por :func:`create_pred_run_dir`).
        meta: Diccionario con la configuración del run (dataset_id,
            champion_run_id, n_pairs, thresholds, timestamp, etc.).
    """
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Funciones de lectura
# ---------------------------------------------------------------------------

def list_pred_runs(dataset_id: str) -> List[Dict[str, Any]]:
    """Lista los runs de predicción de un dataset, más recientes primero.

    Args:
        dataset_id: Identificador del dataset.

    Returns:
        Lista de dicts con el contenido de cada ``meta.json`` disponible.
        Vacía si no hay runs o si el directorio no existe.
    """
    base = (
        artifacts_dir()
        / "predictions"
        / str(dataset_id)
        / _FAMILY_SEGMENT
    )
    if not base.exists():
        return []

    result: List[Dict[str, Any]] = []
    for p in sorted(base.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        if meta_path.exists():
            try:
                result.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                # Directorio corrupto; ignorar y seguir.
                pass
    return result


def resolve_pred_artifact(pred_run_id: str, dataset_id: str) -> Path:
    """Resuelve la ruta absoluta del ``predicciones.parquet`` de un run.

    Args:
        pred_run_id: Identificador del run (ej. ``"pred_20250315_102233"``).
        dataset_id: Dataset al que pertenece el run.

    Returns:
        Ruta absoluta al archivo ``predicciones.parquet``.

    Raises:
        FileNotFoundError: Si el parquet no existe para ese ``dataset_id`` y ``pred_run_id``.
    """
    p = (
        artifacts_dir()
        / "predictions"
        / str(dataset_id)
        / _FAMILY_SEGMENT
        / str(pred_run_id)
        / "predicciones.parquet"
    )
    if not p.exists():
        raise FileNotFoundError(
            f"No existe predicciones.parquet para dataset_id={dataset_id} pred_run_id={pred_run_id}"
        )
    return p
