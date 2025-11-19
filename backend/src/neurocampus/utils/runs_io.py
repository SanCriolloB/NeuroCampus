# backend/src/neurocampus/utils/runs_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import os
import datetime as dt

BASE_DIR = Path(__file__).resolve().parents[4]  # .../backend/src/neurocampus/utils.py → raíz
ARTIFACTS_DIR = Path(os.getenv("NC_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
RUNS_DIR = ARTIFACTS_DIR / "runs"
CHAMPIONS_DIR = ARTIFACTS_DIR / "champions"


def list_runs(model_name: str | None = None) -> list[dict[str, Any]]:
    """
    Lista los runs registrados en artifacts/runs.

    Cada subdirectorio representa un run con al menos:
      - metrics.json
      - config.json / config.yaml (opcional)
    """
    if not RUNS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            continue

        # Filtro opcional por nombre de modelo
        run_model = metrics.get("model_name") or metrics.get("model") or "rbm"
        if model_name and run_model != model_name:
            continue

        created_at = dt.datetime.utcfromtimestamp(run_dir.stat().st_mtime).isoformat() + "Z"

        # Elegimos algunas métricas "top" si existen
        summary = {
            "run_id": run_dir.name,
            "model_name": run_model,
            "created_at": created_at,
            "metrics": {
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "loss": metrics.get("loss"),
            },
        }
        runs.append(summary)

    return runs


def load_run_details(run_id: str) -> dict[str, Any] | None:
    """
    Carga detalles completos de un run concreto:
     - metrics.json (incluyendo histórico por época si se guardó)
     - config.json/config.yaml si existe
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return None

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Aquí podrías añadir lectura de config si te interesa
    return {
        "run_id": run_id,
        "metrics": metrics,
    }


def load_current_champion(model_name: str | None = None) -> dict[str, Any] | None:
    """
    Devuelve info del modelo campeón actual.

    Asume estructura tipo:
      artifacts/champions/rbm/metrics.json
      artifacts/champions/rbm/model.bin
    """
    if not CHAMPIONS_DIR.exists():
        return None

    # Por ahora asumimos un solo subdirectorio por modelo, ej: champions/rbm
    if model_name:
        candidate = CHAMPIONS_DIR / model_name
        metrics_path = candidate / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            return {
                "model_name": model_name,
                "metrics": metrics,
                "path": str(candidate),
            }
        return None

    # Sin nombre, devolvemos el primero que encontremos
    for subdir in CHAMPIONS_DIR.glob("*"):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        return {
            "model_name": subdir.name,
            "metrics": metrics,
            "path": str(subdir),
        }

    return None
