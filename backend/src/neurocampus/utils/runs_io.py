from __future__ import annotations

"""
neurocampus.utils.runs_io
========================

Utilidades para persistencia y lectura de *artifacts* de entrenamiento en NeuroCampus.

Este módulo encapsula el IO de:

- **Runs**: cada entrenamiento produce un run en ``artifacts/runs/<run_id>/``.
- **Champions**: el mejor run por dataset (y/o por familia de modelos) se guarda en
  ``artifacts/champions/<dataset_id>/...``.

Layout de artifacts
------------------------------
Runs
~~~~
Se persiste al menos:

- ``artifacts/runs/<run_id>/metrics.json``  (incluye métricas + metadata + history)
- ``artifacts/runs/<run_id>/history.json``  (opcional pero recomendado para UI)
- ``artifacts/runs/<run_id>/job_meta.json`` (trazabilidad: dataset/model/job/timestamps)
- ``artifacts/runs/<run_id>/config.snapshot.yaml`` (request snapshot reproducible)

Y, si la estrategia lo guarda, también pesos:

- ``rbm.pt``, ``head.pt`` u otros archivos del modelo.

Champions
~~~~~~~~~
Se mantiene:

- ``artifacts/champions/<dataset_id>/champion.json`` (puntero al campeón actual)
- ``artifacts/champions/<dataset_id>/<model_name>/`` (snapshot del campeón por modelo)
  con copia de:
  - ``metrics.json``, ``history.json``, ``config.snapshot.yaml``, ``job_meta.json``
  - y pesos del modelo (si existen)

Criterio de champion
--------------------
Se compara con :func:`_champion_score` usando prioridad:

1) ``val_f1_macro`` o ``f1_macro`` (más alto mejor)
2) ``val_accuracy`` o ``accuracy`` (más alto mejor)
3) ``loss`` (más bajo mejor -> score negativo)

Compatibilidad
--------------
- Se conserva la estructura de ``metrics.json`` "flattened" para que :func:`list_runs`
  pueda listar rápido sin parseos complejos.
- Se soporta directorio de champions con nombre raw o slug (defensivo).
"""

from pathlib import Path
from typing import Any, Optional

import re
import yaml
import json
import os
import datetime as dt
import shutil


# ---------------------------------------------------------------------------
# Paths base
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[4]  # .../backend/src/neurocampus/utils.py → raíz
ARTIFACTS_DIR = Path(os.getenv("NC_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
RUNS_DIR = ARTIFACTS_DIR / "runs"
CHAMPIONS_DIR = ARTIFACTS_DIR / "champions"

_DATASET_STEM_RE = re.compile(r"^(?P<base>.+?)(_beto.*)?$", re.IGNORECASE)


def ensure_artifacts_dirs() -> None:
    """
    Garantiza que existan los directorios base de artifacts.

    Crea:
      - ``artifacts/``
      - ``artifacts/runs/``
      - ``artifacts/champions/``
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    """
    Normaliza texto a un formato seguro para nombres de carpeta/archivo.

    - Lowercase
    - Reemplaza caracteres fuera de ``[a-z0-9._-]`` por ``_``

    :param text: texto arbitrario.
    :return: slug no vacío.
    """
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_") or "x"


# ---------------------------------------------------------------------------
# Runs: id, persistencia, lectura
# ---------------------------------------------------------------------------

def build_run_id(dataset_id: str, model_name: str, job_id: str) -> str:
    """
    Construye un ``run_id`` único y legible.

    Formato::

        <dataset>__<model>__<timestampUTC>__<job8>

    Ejemplo::

        2025-1__rbm_general__20260123T205900Z__a1b2c3d4

    :param dataset_id: id del dataset (ej. "2025-1").
    :param model_name: nombre lógico del modelo (ej. "rbm_general").
    :param job_id: id del job async (UUID).
    """
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    job8 = _slug(job_id)[:8]
    return f"{_slug(dataset_id)}__{_slug(model_name)}__{ts}__{job8}"


def _write_json(path: Path, payload: Any) -> None:
    """Escribe JSON con indent y utf-8 (helper)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_yaml(path: Path, payload: Any) -> None:
    """Escribe YAML (helper)."""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def save_run(
    *,
    run_id: str,
    job_id: str,
    dataset_id: str,
    model_name: str,
    data_ref: str | None,
    params: dict[str, Any] | None,
    final_metrics: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> Path:
    """
    Persiste un run en ``artifacts/runs/<run_id>/``.

    Archivos generados (Commit 3):
    - ``metrics.json``: flattened para listados + metadata + history
    - ``history.json``: lista de puntos por época (para UI)
    - ``job_meta.json``: metadata de ejecución
    - ``config.snapshot.yaml``: snapshot reproducible de request/params

    .. note::
       La estrategia (RBMGeneral/RBMRestringida) puede escribir pesos en el mismo
       directorio (por ejemplo, ``rbm.pt``, ``head.pt``). Este helper no los crea,
       pero sí deja el directorio listo.

    :return: Path al directorio del run.
    """
    ensure_artifacts_dirs()

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    created_at = dt.datetime.utcnow().isoformat() + "Z"

    fm = dict(final_metrics or {})
    hist = list(history or [])

    # loss: preferimos último loss del history si existe
    last_loss = None
    try:
        if hist:
            last_loss = hist[-1].get("loss")
    except Exception:
        last_loss = None

    # time_sec: si hay time_epoch_ms en history, lo sumamos
    time_sec = None
    try:
        ms = 0.0
        for p in hist:
            v = p.get("time_epoch_ms")
            if isinstance(v, (int, float)):
                ms += float(v)
        if ms > 0:
            time_sec = ms / 1000.0
    except Exception:
        time_sec = None

    # (A) job_meta.json
    job_meta = {
        "run_id": run_id,
        "job_id": job_id,
        "dataset_id": dataset_id,
        "model_name": model_name,
        "created_at": created_at,
        "data_ref": data_ref,
    }
    _write_json(run_dir / "job_meta.json", job_meta)

    # (B) config.snapshot.yaml (params snapshot)
    _write_yaml(run_dir / "config.snapshot.yaml", params or {})

    # (C) history.json
    _write_json(run_dir / "history.json", hist)

    # (D) metrics.json (flattened + metadata + history)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "job_id": job_id,
        "dataset_id": dataset_id,
        "model_name": model_name,
        "created_at": created_at,
        "data_ref": data_ref,
        "params": params or {},
        "history": hist,
    }

    # Flatten de métricas finales al root del JSON
    for k, v in fm.items():
        payload[k] = v

    if "loss" not in payload and isinstance(last_loss, (int, float)):
        payload["loss"] = float(last_loss)

    if time_sec is not None and "time_sec" not in payload:
        payload["time_sec"] = float(time_sec)

    _write_json(run_dir / "metrics.json", payload)

    return run_dir


def _dataset_id_from_any(value: Any) -> str | None:
    """
    Convierte un valor cualquiera a dataset_id si es posible.

    Se usa para inferir dataset_id desde metrics/config.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _infer_dataset_id_from_path(path_str: str) -> str | None:
    """
    Intenta inferir dataset_id desde una ruta o nombre de archivo.

    Ejemplos:
      - ``.../2025-1.parquet``      -> ``2025-1``
      - ``.../2025-1_beto.parquet`` -> ``2025-1`` (strip de sufijo BETO)
    """
    try:
        name = Path(str(path_str)).name
        stem = Path(name).stem
        m = _DATASET_STEM_RE.match(stem)
        return (m.group("base") if m else stem) or None
    except Exception:
        return None


def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    """
    Carga un YAML si existe; retorna None si no existe o si no parsea.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _infer_dataset_id(run_dir: Path, metrics: dict[str, Any]) -> str | None:
    """
    Intenta determinar el dataset_id asociado a un run.

    Prioridad:
      1) metrics.json: dataset_id/dataset/periodo
      2) config.snapshot.yaml / config.yaml:
         - dataset.id/dataset_id/periodo
         - dataset.path (inferido)
         - dataset_path (inferido)
    """
    for k in ("dataset_id", "dataset", "periodo"):
        v = _dataset_id_from_any(metrics.get(k))
        if v:
            return v

    cfg = _load_yaml_if_exists(run_dir / "config.snapshot.yaml") or _load_yaml_if_exists(run_dir / "config.yaml")
    if cfg:
        ds = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else None
        if isinstance(ds, dict):
            v = _dataset_id_from_any(ds.get("id") or ds.get("dataset_id") or ds.get("periodo"))
            if v:
                return v
            p = ds.get("path")
            v2 = _infer_dataset_id_from_path(p) if p else None
            if v2:
                return v2

        v3 = _infer_dataset_id_from_path(cfg.get("dataset_path")) if cfg.get("dataset_path") else None
        if v3:
            return v3

    return None


def list_runs(
    model_name: str | None = None,
    dataset_id: str | None = None,
    periodo: str | None = None,
) -> list[dict[str, Any]]:
    """
    Lista runs registrados en ``artifacts/runs``.

    Retorna un resumen amigable para la UI (tabla de runs), con métricas clave.

    :param model_name: filtra por modelo (ej. 'rbm_general').
    :param dataset_id: filtra por dataset.
    :param periodo: alias de dataset_id (compatibilidad).
    """
    if not RUNS_DIR.exists():
        return []

    ds_filter = dataset_id or periodo
    runs: list[dict[str, Any]] = []

    for run_dir in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f) or {}
        except Exception:
            continue

        run_model = metrics.get("model_name") or metrics.get("model") or "rbm"
        if model_name and run_model != model_name:
            continue

        run_dataset_id = _infer_dataset_id(run_dir, metrics)
        if ds_filter and run_dataset_id != ds_filter:
            continue

        created_at = dt.datetime.utcfromtimestamp(run_dir.stat().st_mtime).isoformat() + "Z"

        summary = {
            "run_id": run_dir.name,
            "model_name": run_model,
            "dataset_id": run_dataset_id,
            "created_at": created_at,
            "metrics": {
                # val_* preferido si existe (commit 4),
                # pero no rompe si aún no se produce
                "val_f1_macro": metrics.get("val_f1_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "val_accuracy": metrics.get("val_accuracy"),
                "accuracy": metrics.get("accuracy"),
                "loss": metrics.get("loss"),
                "time_sec": metrics.get("time_sec"),
            },
        }
        runs.append(summary)

    return runs


def load_run_details(run_id: str) -> dict[str, Any] | None:
    """
    Carga detalles completos de un run desde ``artifacts/runs/<run_id>/``.

    Lee:
      - metrics.json (obligatorio)
      - config.snapshot.yaml o config.yaml (opcional)

    Retorna dict compatible con RunDetails:
    ``{run_id, dataset_id, metrics, config, artifact_path}``.
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return None

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f) or {}

    cfg = _load_yaml_if_exists(run_dir / "config.snapshot.yaml") or _load_yaml_if_exists(run_dir / "config.yaml")
    ds = _infer_dataset_id(run_dir, metrics)

    return {
        "run_id": run_id,
        "dataset_id": ds,
        "metrics": metrics,
        "config": cfg,
        "artifact_path": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Champion: score, snapshot, promote
# ---------------------------------------------------------------------------

def _champion_score(metrics: dict[str, Any]) -> tuple[int, float]:
    """
    Calcula un score comparable para decidir campeón.

    Prioridad:
      1) val_f1_macro (más alto es mejor)
      2) f1_macro (más alto es mejor)
      3) val_accuracy (más alto es mejor)
      4) accuracy (más alto es mejor)
      5) loss (más bajo es mejor) -> score negativo

    Retorna:
      (tier, score) para comparar (tuple comparable).
    """
    if isinstance(metrics.get("val_f1_macro"), (int, float)):
        return (5, float(metrics["val_f1_macro"]))
    if isinstance(metrics.get("f1_macro"), (int, float)):
        return (4, float(metrics["f1_macro"]))
    if isinstance(metrics.get("val_accuracy"), (int, float)):
        return (3, float(metrics["val_accuracy"]))
    if isinstance(metrics.get("accuracy"), (int, float)):
        return (2, float(metrics["accuracy"]))
    if isinstance(metrics.get("loss"), (int, float)):
        return (1, -float(metrics["loss"]))
    return (0, float("-inf"))


def _dataset_dir_candidates(dataset_id: str) -> list[Path]:
    """
    Retorna posibles directorios para un dataset en champions.

    Se mantiene compatibilidad si en algún momento se creó sin slug.
    """
    raw = CHAMPIONS_DIR / str(dataset_id)
    slug = CHAMPIONS_DIR / _slug(dataset_id)
    # mantener orden: primero raw, luego slug
    return [raw, slug] if raw != slug else [raw]


def load_dataset_champion(dataset_id: str) -> dict[str, Any] | None:
    """
    Devuelve el champion de un dataset.

    Orden de búsqueda:
      1) ``artifacts/champions/<dataset_id>/champion.json``
      2) Fallback: escoger el mejor entre:
         ``artifacts/champions/<dataset_id>/*/metrics.json``

    Retorna dict compatible con ChampionInfo:
    ``{model_name, dataset_id, metrics, path}``.
    """
    for ds_dir in _dataset_dir_candidates(dataset_id):
        champ_json = ds_dir / "champion.json"

        # 1) Preferir champion.json
        if champ_json.exists():
            try:
                champ = json.loads(champ_json.read_text(encoding="utf-8")) or {}
                mname = champ.get("model_name")
                metrics = champ.get("metrics") if isinstance(champ.get("metrics"), dict) else {}
                if mname:
                    model_dir = ds_dir / str(mname)
                    return {
                        "model_name": str(mname),
                        "dataset_id": str(dataset_id),
                        "metrics": metrics,
                        "path": str(model_dir if model_dir.exists() else ds_dir),
                    }
            except Exception:
                pass

        # 2) Fallback por score
        if not ds_dir.exists():
            continue

        best: dict[str, Any] | None = None
        best_score = (0, float("-inf"))

        for model_dir in ds_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            mp = model_dir / "metrics.json"
            if not mp.exists():
                continue
            try:
                metrics = json.loads(mp.read_text(encoding="utf-8")) or {}
            except Exception:
                continue

            score = _champion_score(metrics)
            if score > best_score:
                best_score = score
                best = {
                    "model_name": str(metrics.get("model_name") or model_dir.name),
                    "dataset_id": str(dataset_id),
                    "metrics": metrics,
                    "path": str(model_dir),
                }

        if best:
            return best

    return None


def _copy_run_artifacts_to_dir(run_dir: Path, target_dir: Path) -> None:
    """
    Copia archivos relevantes del run a un directorio destino (champion snapshot).

    Copia:
    - archivos estándar (metrics/history/config/job_meta)
    - pesos "conocidos" si existen (rbm.pt/head.pt/model.pt/weights.pt/etc.)

    .. note::
       Si tu estrategia guarda pesos con otros nombres, agrégalos a la lista.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Archivos estándar
    for fname in ("metrics.json", "history.json", "config.snapshot.yaml", "job_meta.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)

    # Pesos / artefactos de modelo comunes
    for fname in ("rbm.pt", "head.pt", "model.pt", "weights.pt", "vectorizer.json", "encoder.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)


def maybe_update_champion(
    *,
    dataset_id: str,
    model_name: str,
    metrics: dict[str, Any],
    source_run_id: str | None = None,
) -> dict[str, Any]:
    """
    Registra/actualiza el champion del dataset si el nuevo run es mejor.

    Siempre escribe snapshot del modelo en::

      artifacts/champions/<dataset_id>/<model_name>/metrics.json

    Y si el nuevo run supera al champion actual, actualiza::

      artifacts/champions/<dataset_id>/champion.json

    Si ``source_run_id`` se provee y existe el run_dir, también copia
    artefactos del run al directorio del champion del modelo.

    :return: dict con {dataset_id, model_name, promoted, path}.
    """
    ensure_artifacts_dirs()

    ds_dir = CHAMPIONS_DIR / _slug(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)

    model_dir = ds_dir / _slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Guardar snapshot de métricas (siempre)
    _write_json(model_dir / "metrics.json", metrics)

    # Si hay run, copiar artefactos al snapshot del modelo (si existen)
    if source_run_id:
        run_dir = RUNS_DIR / source_run_id
        if run_dir.exists():
            _copy_run_artifacts_to_dir(run_dir, model_dir)

    # Comparar contra champion actual
    current = load_dataset_champion(dataset_id)
    new_score = _champion_score(metrics)

    cur_score = (0, float("-inf"))
    if current and isinstance(current.get("metrics"), dict):
        cur_score = _champion_score(current["metrics"])

    promoted = new_score > cur_score

    if promoted:
        payload = {
            "model_name": model_name,
            "dataset_id": dataset_id,
            "metrics": metrics,
            "source_run_id": source_run_id,
            "updated_at": dt.datetime.utcnow().isoformat() + "Z",
        }
        _write_json(ds_dir / "champion.json", payload)

    return {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "promoted": promoted,
        "path": str(ds_dir),
    }


def promote_run_to_champion(
    *,
    dataset_id: str,
    run_id: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Promueve un run existente a champion **manual**, copiando artefactos y actualizando
    ``champion.json`` sin evaluar score.

    Es útil para el endpoint opcional ``POST /modelos/champion/promote``.

    :param dataset_id: dataset target.
    :param run_id: run a promover.
    :param model_name: si no se provee, se infiere desde metrics/job_meta o desde el nombre del run.
    :raises FileNotFoundError: si el run no existe.
    :return: dict compatible con ChampionInfo.
    """
    ensure_artifacts_dirs()

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run {run_id} no existe en {RUNS_DIR}")

    # 1) leer metrics
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) or {}
        except Exception:
            metrics = {}

    # 2) inferir model_name si no viene
    if not model_name:
        # prioridad: metrics.json -> model_name
        m = metrics.get("model_name") or metrics.get("model")
        if isinstance(m, str) and m.strip():
            model_name = m.strip()

    if not model_name:
        # fallback: parsear run_id: <dataset>__<model>__...
        parts = run_id.split("__")
        model_name = parts[1] if len(parts) >= 2 else "rbm_general"

    # 3) copiar artefactos run -> champions/<dataset>/<model>
    ds_dir = CHAMPIONS_DIR / _slug(dataset_id)
    model_dir = ds_dir / _slug(model_name)
    _copy_run_artifacts_to_dir(run_dir, model_dir)

    # 4) escribir champion.json apuntando a este run
    payload = {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "source_run_id": run_id,
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    ds_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ds_dir / "champion.json", payload)

    return {
        "model_name": str(model_name),
        "dataset_id": str(dataset_id),
        "metrics": metrics or {},
        "path": str(model_dir),
    }


def load_current_champion(
    model_name: str | None = None,
    dataset_id: str | None = None,
    periodo: str | None = None,
) -> dict[str, Any] | None:
    """
    Devuelve info del campeón actual.

    Soporta 2 estructuras:

    (1) NUEVA (por dataset):
        ``artifacts/champions/<dataset_id>/<model_name>/metrics.json``

    (2) LEGACY (por modelo):
        ``artifacts/champions/<model_name>/metrics.json``

    Retorna dict compatible con ChampionInfo o None.
    """
    if not CHAMPIONS_DIR.exists():
        return None

    ds_filter = dataset_id or periodo

    # -------------------------
    # Nuevo patrón por dataset
    # -------------------------
    if ds_filter:
        # si no especifican model_name, preferir champion.json
        champ = load_dataset_champion(ds_filter)
        if champ and (not model_name or champ.get("model_name") == model_name):
            return champ

        # si especifican model_name, buscar directo
        if model_name:
            base = CHAMPIONS_DIR / _slug(ds_filter) / _slug(model_name)
            mp = base / "metrics.json"
            if mp.exists():
                metrics = json.loads(mp.read_text(encoding="utf-8")) or {}
                return {"model_name": model_name, "dataset_id": ds_filter, "metrics": metrics, "path": str(base)}
            return None

    # -------------------------
    # Patrón legacy por modelo
    # -------------------------
    if model_name:
        candidate = CHAMPIONS_DIR / _slug(model_name)
        mp = candidate / "metrics.json"
        if mp.exists():
            metrics = json.loads(mp.read_text(encoding="utf-8")) or {}
            ds = _dataset_id_from_any(metrics.get("dataset_id") or metrics.get("dataset") or metrics.get("periodo"))
            return {"model_name": model_name, "dataset_id": ds, "metrics": metrics, "path": str(candidate)}
        return None

    for subdir in CHAMPIONS_DIR.glob("*"):
        if not subdir.is_dir():
            continue
        mp = subdir / "metrics.json"
        if not mp.exists():
            continue
        metrics = json.loads(mp.read_text(encoding="utf-8")) or {}
        ds = _dataset_id_from_any(metrics.get("dataset_id") or metrics.get("dataset") or metrics.get("periodo"))
        return {"model_name": subdir.name, "dataset_id": ds, "metrics": metrics, "path": str(subdir)}

    return None
