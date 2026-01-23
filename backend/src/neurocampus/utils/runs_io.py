# backend/src/neurocampus/utils/runs_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import re
import yaml
from typing import Optional

import json
import os
import datetime as dt

BASE_DIR = Path(__file__).resolve().parents[4]  # .../backend/src/neurocampus/utils.py → raíz
ARTIFACTS_DIR = Path(os.getenv("NC_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
RUNS_DIR = ARTIFACTS_DIR / "runs"
CHAMPIONS_DIR = ARTIFACTS_DIR / "champions"
_DATASET_STEM_RE = re.compile(r"^(?P<base>.+?)(_beto.*)?$", re.IGNORECASE)

def ensure_artifacts_dirs() -> None:
    """
    Garantiza que existan los directorios base de artifacts.

    Crea:
      - artifacts/
      - artifacts/runs/
      - artifacts/champions/
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    """
    Normaliza texto a un formato seguro para nombres de carpeta/archivo.
    """
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_") or "x"


def build_run_id(dataset_id: str, model_name: str, job_id: str) -> str:
    """
    Construye un run_id único y legible.

    Formato:
      <dataset>__<model>__<timestampUTC>__<job8>

    Ej:
      2025-1__rbm_general__20260123T205900Z__a1b2c3d4
    """
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    job8 = _slug(job_id)[:8]
    return f"{_slug(dataset_id)}__{_slug(model_name)}__{ts}__{job8}"


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
    Persiste un run en artifacts/runs/<run_id>/metrics.json.

    El archivo metrics.json se guarda *flattened* para que list_runs()
    pueda leer directamente claves como:
      - accuracy, f1_macro, loss, time_sec, etc.

    Además incluye metadata:
      - run_id, job_id, dataset_id, model_name, created_at, params, data_ref, history
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

    # Flatten de métricas finales (numéricas) al root del JSON
    for k, v in fm.items():
        payload[k] = v

    if "loss" not in payload and isinstance(last_loss, (int, float)):
        payload["loss"] = float(last_loss)

    if time_sec is not None and "time_sec" not in payload:
        payload["time_sec"] = float(time_sec)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return run_dir


def _champion_score(metrics: dict[str, Any]) -> tuple[int, float]:
    """
    Puntaje para decidir campeón.

    Prioridad:
      1) f1_macro (más alto es mejor)
      2) accuracy (más alto es mejor)
      3) loss (más bajo es mejor) -> se convierte a score negativo

    Retorna:
      (tier, score)
    donde tier más alto indica mejor fuente.
    """
    if isinstance(metrics.get("f1_macro"), (int, float)):
        return (3, float(metrics["f1_macro"]))
    if isinstance(metrics.get("accuracy"), (int, float)):
        return (2, float(metrics["accuracy"]))
    if isinstance(metrics.get("loss"), (int, float)):
        return (1, -float(metrics["loss"]))
    return (0, float("-inf"))

def _champion_score(metrics: dict[str, Any]) -> tuple[int, float]:
    """
    Calcula un score comparable para decidir campeón.

    Prioridad:
      1) f1_macro (más alto es mejor)
      2) accuracy (más alto es mejor)
      3) loss (más bajo es mejor) -> se compara como score negativo

    Retorna:
      (tier, score) para comparar (tuple comparable).
    """
    if isinstance(metrics.get("f1_macro"), (int, float)):
        return (3, float(metrics["f1_macro"]))
    if isinstance(metrics.get("accuracy"), (int, float)):
        return (2, float(metrics["accuracy"]))
    if isinstance(metrics.get("loss"), (int, float)):
        return (1, -float(metrics["loss"]))
    return (0, float("-inf"))

def load_dataset_champion(dataset_id: str) -> dict[str, Any] | None:
    """
    Devuelve el champion de un dataset.

    Orden de búsqueda:
      1) artifacts/champions/<dataset_id>/champion.json
         - Este archivo debe contener al menos: { "model_name": ..., "metrics": {...} }
      2) Fallback: elegir el mejor entre:
         artifacts/champions/<dataset_id>/*/metrics.json
         comparando con _champion_score()

    Retorna un dict compatible con ChampionInfo:
      {
        "model_name": str,
        "dataset_id": str,
        "metrics": dict,
        "path": str
      }
    """
    ds_dir = CHAMPIONS_DIR / str(dataset_id)
    champ_json = ds_dir / "champion.json"

    # 1) Preferir champion.json
    if champ_json.exists():
        try:
            with champ_json.open("r", encoding="utf-8") as f:
                champ = json.load(f) or {}
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
            # Si champion.json está corrupto, seguimos a fallback
            pass

    # 2) Fallback: escoge el mejor subdir por score
    if not ds_dir.exists():
        return None

    best: dict[str, Any] | None = None
    best_score = (0, float("-inf"))

    for model_dir in ds_dir.glob("*"):
        if not model_dir.is_dir():
            continue
        mp = model_dir / "metrics.json"
        if not mp.exists():
            continue
        try:
            with mp.open("r", encoding="utf-8") as f:
                metrics = json.load(f) or {}
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

    return best


def maybe_update_champion(
    *,
    dataset_id: str,
    model_name: str,
    metrics: dict[str, Any],
    source_run_id: str | None = None,
) -> dict[str, Any]:
    """
    Registra/actualiza el champion del dataset si el nuevo run es mejor.

    Escribe siempre el snapshot del modelo en:
      artifacts/champions/<dataset_id>/<model_name>/metrics.json

    Y si el nuevo run supera al champion actual, actualiza:
      artifacts/champions/<dataset_id>/champion.json
    """
    ensure_artifacts_dirs()

    ds_dir = CHAMPIONS_DIR / _slug(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)

    model_dir = ds_dir / _slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Guardar snapshot del modelo
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

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
        with (ds_dir / "champion.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "promoted": promoted,
        "path": str(ds_dir),
    }

def _dataset_id_from_any(value: Any) -> str | None:
    """
    Convierte un valor cualquiera a dataset_id si es posible.

    - Retorna None si el valor es None o string vacío.
    - Retorna el string limpio en caso contrario.

    Se usa para leer dataset_id desde:
      - metrics.json (dataset_id / dataset / periodo)
      - config.yaml (dataset.id / dataset.dataset_id / dataset.periodo)
    """
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _infer_dataset_id_from_path(path_str: str) -> str | None:
    """
    Intenta inferir dataset_id desde una ruta o nombre de archivo.

    Ejemplos:
      - '.../2025-1.parquet'      -> '2025-1'
      - '.../2025-1_beto.parquet' -> '2025-1' (strip de sufijo BETO)

    Retorna None si no puede inferirse.
    """
    try:
        name = Path(str(path_str)).name  # soporta rutas con carpetas
        stem = Path(name).stem           # sin extensión
        m = _DATASET_STEM_RE.match(stem)
        return (m.group("base") if m else stem) or None
    except Exception:
        return None


def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    """
    Carga un YAML si existe.

    - Retorna dict si puede leerse.
    - Retorna None si no existe o si ocurre error de parseo.

    Se usa para leer 'config.snapshot.yaml' o 'config.yaml' dentro del run.
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
      1) metrics.json:
         - dataset_id / dataset / periodo
      2) config.snapshot.yaml o config.yaml (si existe):
         - dataset.id / dataset.dataset_id / dataset.periodo
         - dataset.path (infiriendo por nombre)
         - dataset_path (fallback)

    Retorna None si no encuentra ninguna pista.
    """
    # 1) Preferimos explícito en metrics.json
    for k in ("dataset_id", "dataset", "periodo"):
        v = _dataset_id_from_any(metrics.get(k))
        if v:
            return v

    # 2) Si hay snapshot de config, intentar dataset.path o dataset_id
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

        # fallback por si config guarda "dataset_path" directo
        v3 = _infer_dataset_id_from_path(cfg.get("dataset_path")) if cfg.get("dataset_path") else None
        if v3:
            return v3

    return None

def list_runs(
    model_name: str | None = None,
    dataset_id: str | None = None,
    periodo: str | None = None,  # alias útil para el frontend
) -> list[dict[str, Any]]:
    """
    Lista los runs registrados en artifacts/runs.

    Cada subdirectorio representa un run con al menos:
      - metrics.json
      - config.snapshot.yaml / config.yaml (opcional)

    Parámetros:
      - model_name: filtra por nombre de modelo (ej: 'rbm', 'rbm_general').
      - dataset_id: filtra por dataset asociado al run (si fue registrado o inferible).
      - periodo: alias de dataset_id (compatibilidad con UI/planes).

    Notas:
      - Si dataset_id/periodo no se puede inferir (no hay metadata),
        el run aparecerá con dataset_id = None y no matcheará filtros.
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
                metrics = json.load(f)
        except Exception:
            continue

        # Filtro opcional por nombre de modelo
        run_model = metrics.get("model_name") or metrics.get("model") or "rbm"
        if model_name and run_model != model_name:
            continue

        run_dataset_id = _infer_dataset_id(run_dir, metrics)
        if ds_filter and run_dataset_id != ds_filter:
            continue

        created_at = dt.datetime.utcfromtimestamp(run_dir.stat().st_mtime).isoformat() + "Z"

        # Elegimos algunas métricas "top" si existen
        summary = {
            "run_id": run_dir.name,
            "model_name": run_model,
            "dataset_id": run_dataset_id,
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
    Carga detalles completos de un run concreto desde artifacts/runs/<run_id>/.

    Lee:
      - metrics.json (obligatorio)
      - config.snapshot.yaml o config.yaml (opcional)

    Retorna:
      {
        "run_id": str,
        "dataset_id": str | None,
        "metrics": dict,
        "config": dict | None,
        "artifact_path": str
      }

    Retorna None si el run no existe o si no tiene metrics.json.
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return None

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    cfg = _load_yaml_if_exists(run_dir / "config.snapshot.yaml") or _load_yaml_if_exists(run_dir / "config.yaml")
    ds = _infer_dataset_id(run_dir, metrics)

    return {
        "run_id": run_id,
        "dataset_id": ds,
        "metrics": metrics,
        "config": cfg,
        "artifact_path": str(run_dir),
    }


def load_current_champion(
    model_name: str | None = None,
    dataset_id: str | None = None,
    periodo: str | None = None,
) -> dict[str, Any] | None:
    """
    Devuelve info del modelo campeón actual.

    Soporta 2 estructuras:

    (1) NUEVA (por dataset):
        artifacts/champions/<dataset_id>/<model_name>/metrics.json

        - Si dataset_id está presente, se busca primero en esta estructura.
        - Si además model_name está presente, se exige esa subcarpeta.

    (2) LEGACY (por modelo):
        artifacts/champions/<model_name>/metrics.json

        - Si dataset_id no está presente, o si no existe estructura nueva,
          cae a la estructura legacy.

    Retorna:
      {
        "model_name": str,
        "dataset_id": str | None,
        "metrics": dict,
        "path": str
      }

    Retorna None si no existe champion.
    """
    if not CHAMPIONS_DIR.exists():
        return None

    ds_filter = dataset_id or periodo

    # -------------------------
    # Nuevo patrón por dataset
    # -------------------------
    if ds_filter:
        base = CHAMPIONS_DIR / ds_filter

        if model_name:
            candidate = base / model_name
            mp = candidate / "metrics.json"
            if mp.exists():
                with mp.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                return {"model_name": model_name, "dataset_id": ds_filter, "metrics": metrics, "path": str(candidate)}
            return None

        # Sin model_name: devolver el primero válido bajo champions/<dataset_id>/*
        if base.exists():
            for subdir in base.glob("*"):
                if not subdir.is_dir():
                    continue
                mp = subdir / "metrics.json"
                if not mp.exists():
                    continue
                with mp.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                return {"model_name": subdir.name, "dataset_id": ds_filter, "metrics": metrics, "path": str(subdir)}
            return None

    # -------------------------
    # Patrón legacy por modelo
    # -------------------------
    if model_name:
        candidate = CHAMPIONS_DIR / model_name
        mp = candidate / "metrics.json"
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            ds = _dataset_id_from_any(metrics.get("dataset_id") or metrics.get("dataset") or metrics.get("periodo"))
            return {"model_name": model_name, "dataset_id": ds, "metrics": metrics, "path": str(candidate)}
        return None

    # Sin nombre: devolver el primero válido bajo champions/*
    for subdir in CHAMPIONS_DIR.glob("*"):
        if not subdir.is_dir():
            continue
        mp = subdir / "metrics.json"
        if not mp.exists():
            continue
        with mp.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        ds = _dataset_id_from_any(metrics.get("dataset_id") or metrics.get("dataset") or metrics.get("periodo"))
        return {"model_name": subdir.name, "dataset_id": ds, "metrics": metrics, "path": str(subdir)}

    return None
