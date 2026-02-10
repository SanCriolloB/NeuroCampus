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
from typing import Any, Optional, Dict, List

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
RUNS_DIR = BASE_DIR / "artifacts" / "runs"
_DATASET_STEM_RE = re.compile(r"^(?P<base>.+?)(_beto.*)?$", re.IGNORECASE)
LEGACY_CHAMPIONS_DIR = ARTIFACTS_DIR / "champions"
LEGACY_CHAMPIONS_DIR_MODELS = LEGACY_CHAMPIONS_DIR / "models"

# ---------------------------------------------------------------------------
# Helpers base (faltantes en el archivo actual)
# ---------------------------------------------------------------------------

def _try_load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def slug(text: str) -> str:
    """
    Slug estable para nombres de carpeta/archivo.
    (Se usa en champions/<family>/<dataset_id> y runs)
    """
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_") or "x"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _abs_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (BASE_DIR / pp)

def _norm_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _deep_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_req_field(metrics: dict[str, Any], field: str) -> Any:
    """
    Intenta leer un campo desde:
      - top-level metrics[field]
      - metrics['params']['req'][field]
    """
    if not isinstance(metrics, dict):
        return None
    if field in metrics:
        return metrics.get(field)
    return _deep_get(metrics, "params", "req", field)


# ---------------------------------------------------------------------------
# Champions family-aware + metadata estable en runs
# ---------------------------------------------------------------------------

def _family_slug(family: Optional[str]) -> Optional[str]:
    fam = (family or "").strip()
    return _slug(fam) if fam else None


def _champions_ds_dir(dataset_id: str, family: Optional[str] = None) -> Path:
    """Directorio de champions para un dataset.

    Si `family` es provista, usamos la estructura:
        artifacts/champions/<family>/<dataset_id>/

    Caso legacy (sin family):
        artifacts/champions/<dataset_id>/
    """
    ds_slug = _slug(dataset_id)
    if family:
        fam_slug = _slug(str(family))
        return CHAMPIONS_DIR / fam_slug / ds_slug
    return CHAMPIONS_DIR / ds_slug



def _ensure_champions_ds_dir(dataset_id: str, family: str | None = None) -> Path:
    return ensure_dir(_champions_ds_dir(dataset_id, family=family))


def _champions_family_root(family: str) -> Path:
    """champions/<family> (sin crear)."""
    fam = _family_slug(family)
    if not fam:
        raise ValueError("family vacío para champions/<family>")
    return CHAMPIONS_DIR / fam


def _ensure_champions_family_root(family: str) -> Path:
    return ensure_dir(_champions_family_root(family))


def _try_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _extract_req_from_params(params: Any) -> dict[str, Any]:
    """params puede venir como {'req': {...}} o directamente {...}."""
    if isinstance(params, dict):
        if isinstance(params.get("req"), dict):
            return params["req"]
        return params
    return {}


def _extract_run_context_from_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    """Saca contexto estable desde params.req (o fallback a top-level)."""
    params = metrics_payload.get("params") or {}
    req = _extract_req_from_params(params)

    def pick(key: str) -> Any:
        v = req.get(key)
        return v if v is not None else metrics_payload.get(key)

    ctx = {
        "family": pick("family"),
        "task_type": pick("task_type"),
        "input_level": pick("input_level"),
        "target_col": pick("target_col"),
        "data_plan": pick("data_plan"),
        "data_source": pick("data_source"),
        "target_mode": pick("target_mode"),
        "split_mode": pick("split_mode"),
        "val_ratio": pick("val_ratio"),
        # incremental / warm-start (si aplica)
        "window_k": req.get("window_k"),
        "replay_size": req.get("replay_size"),
        "replay_strategy": req.get("replay_strategy"),
        "warm_start_from": req.get("warm_start_from"),
        "warm_start_run_id": req.get("warm_start_run_id"),
    }
    return ctx


def _data_meta_from_data_ref(data_ref: str | None) -> dict[str, Any] | None:
    """
    Metadata mínima para auditoría/compatibilidad.
    - feature_pack -> lee meta.json
    - pair_matrix  -> lee pair_meta.json
    """
    if not data_ref:
        return None

    p = _abs_path(data_ref)
    if not p.exists():
        return None

    meta: dict[str, Any] = {"data_ref_basename": p.name}

    if p.name == "train_matrix.parquet":
        m = _try_read_json(p.parent / "meta.json")
        if m:
            meta.update(
                {
                    "input_uri": m.get("input_uri"),
                    "created_at": m.get("created_at"),
                    "tfidf_dims": m.get("tfidf_dims"),
                    "blocks": m.get("blocks"),
                    "has_text": m.get("has_text"),
                    "has_accept": m.get("has_accept"),
                    "n_columns": (len(m["columns"]) if isinstance(m.get("columns"), list) else None),
                }
            )

    if p.name == "pair_matrix.parquet":
        pm = _try_read_json(p.parent / "pair_meta.json")
        if pm:
            meta.update(
                {
                    "created_at": pm.get("created_at"),
                    "tfidf_dims": pm.get("tfidf_dims"),
                    "blocks": pm.get("blocks"),
                    "target_col_pair": pm.get("target_col"),
                    "n_pairs": pm.get("n_pairs"),
                    "n_par_stats": pm.get("n_par_stats") if isinstance(pm.get("n_par_stats"), dict) else None,
                }
            )

    meta = {k: v for k, v in meta.items() if v is not None}
    return meta or None


def _enrich_run_metrics_payload(payload: dict[str, Any]) -> None:
    """
    Agrega al top-level: family/task_type/input_level/target_col/data_plan (+ extras)
    y data_meta (tfidf_dims, blocks, etc.). No rompe si no hay nada.
    """
    ctx = _extract_run_context_from_metrics(payload)
    for k, v in ctx.items():
        if v is not None:
            payload[k] = v

    dm = _data_meta_from_data_ref(payload.get("data_ref"))
    if dm is not None:
        payload["data_meta"] = dm


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

def _ensure_source_run_id(
    champ: Dict[str, Any],
    ds_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Asegura `source_run_id` en el dict champion de forma robusta.

    - Si falta `source_run_id`, lo deriva de:
      (champ.metrics.run_id) o (champ.run_id)
    - Si `ds_dir` está disponible, intenta persistir el campo en champion.json,
      pero NUNCA debe tumbar el endpoint si falla la escritura.
    """
    if not isinstance(champ, dict):
        return champ

    # 1) Normalizar `source_run_id`
    if not champ.get("source_run_id"):
        fallback = None
        metrics = champ.get("metrics") or {}
        if isinstance(metrics, dict):
            fallback = metrics.get("run_id")
        fallback = fallback or champ.get("run_id")

        if fallback:
            champ["source_run_id"] = fallback

    # 2) Persistir (best-effort) si hay ds_dir
    if ds_dir is not None and champ.get("source_run_id"):
        try:
            champion_path = ds_dir / "champion.json"
            if champion_path.exists():
                payload = _try_read_json(champion_path) or {}
                if isinstance(payload, dict) and not payload.get("source_run_id"):
                    payload["source_run_id"] = champ["source_run_id"]
                    _write_json(champion_path, payload)
        except Exception:
            # Nunca debe tumbar el servicio por intentar "sanear" metadata
            pass

    return champ





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
    Persiste un run en artifacts/runs/<run_id>/ con:
      - metrics.json (flatten + history + params + contexto estable + data_meta)
      - history.json
      - job_meta.json
      - config.snapshot.yaml
    """
    ensure_artifacts_dirs()

    run_dir = ensure_dir(RUNS_DIR / run_id)
    created_at = now_utc_iso()

    fm = dict(final_metrics or {})
    hist = list(history or [])

    # loss preferido: último loss del history
    last_loss = None
    if hist:
        try:
            last_loss = hist[-1].get("loss")
        except Exception:
            last_loss = None

    # time_sec: suma time_epoch_ms si existe
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

    # (B) config.snapshot.yaml (snapshot reproducible)
    _write_yaml(run_dir / "config.snapshot.yaml", params or {})

    # (C) history.json (para UI)
    _write_json(run_dir / "history.json", hist)

    # (D) metrics.json (payload principal)
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

    # flatten final_metrics a top-level
    for k, v in fm.items():
        payload[k] = v

    if "loss" not in payload and isinstance(last_loss, (int, float)):
        payload["loss"] = float(last_loss)

    if time_sec is not None and "time_sec" not in payload:
        payload["time_sec"] = float(time_sec)

    # contexto estable + data_meta (NO debe tumbar run si falla)
    try:
        _enrich_run_metrics_payload(payload)
    except Exception:
        pass

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
    base_dir: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    family: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Lista runs en artifacts/runs (más recientes primero).

    Retorna summaries listos para UI/API. Además de métricas, incluye *contexto*
    (family, task_type, input_level, target_col, data_plan) inferido desde metrics.json
    y/o params.req.
    """
    base = Path(base_dir) if base_dir is not None else BASE_DIR
    runs_dir = base / "artifacts" / "runs"
    if not runs_dir.exists():
        return []

    fam_norm = _slug(str(family)) if family else None

    # Orden: más reciente primero
    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    out: list[dict] = []
    for run_dir in run_dirs:
        metrics_p = run_dir / "metrics.json"
        if not metrics_p.exists():
            continue

        try:
            metrics = json.loads(metrics_p.read_text(encoding="utf-8"))
        except Exception:
            continue

        ds = metrics.get("dataset_id") or _infer_dataset_id(metrics, run_dir.name)
        if dataset_id and ds != dataset_id:
            continue

        ctx = _extract_run_context_from_metrics(metrics)
        if fam_norm and _slug(str(ctx.get("family") or "")) != fam_norm:
            continue

        # Resumen base (top-level)
        summary: dict[str, Any] = {
            "run_id": metrics.get("run_id", run_dir.name),
            "dataset_id": ds,
            "model_name": metrics.get("model_name"),
            "created_at": metrics.get("created_at"),
            "artifact_path": str(run_dir),
            "family": ctx.get("family"),
            "task_type": ctx.get("task_type"),
            "input_level": ctx.get("input_level"),
            "target_col": ctx.get("target_col"),
            "data_plan": ctx.get("data_plan"),
            # Métricas UI-friendly
            "metrics": {},
        }

        # Métricas comunes (si existen)
        keep = [
            "epoch",
            "loss",
            "loss_final",
            "recon_error",
            "recon_error_final",
            "rbm_grad_norm",
            "cls_loss",
            "accuracy",
            "f1_macro",
            "val_accuracy",
            "val_f1_macro",
            "train_mae",
            "train_rmse",
            "train_r2",
            "val_mae",
            "val_rmse",
            "val_r2",
            "n_train",
            "n_val",
            "pred_min",
            "pred_max",
        ]
        for k in keep:
            if k in metrics and metrics[k] is not None:
                summary["metrics"][k] = metrics[k]

        out.append(summary)
        if len(out) >= int(limit):
            break

    return out



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


def _dataset_dir_candidates(dataset_id: str, family: str | None = None) -> list[Path]:
    """
    Candidatos de directorio para champions, soportando:
      - layout nuevo: artifacts/champions/<family>/<dataset_id>/
      - layout legacy: artifacts/champions/<dataset_id>/
    Incluye también variantes con _slug(...) por compatibilidad.
    """
    ds_raw = (dataset_id or "").strip()
    ds_slug = _slug(ds_raw) if ds_raw else ds_raw

    fam_raw = (family or "").strip()
    fam_slug = _slug(fam_raw) if fam_raw else fam_raw

    candidates: list[Path] = []

    # Nuevo layout por family (preferido)
    if fam_raw:
        for fam in [fam_raw, fam_slug]:
            if not fam:
                continue
            for ds in [ds_raw, ds_slug]:
                if not ds:
                    continue
                candidates.append(CHAMPIONS_DIR / fam / ds)

    # Legacy layout (fallback)
    for ds in [ds_raw, ds_slug]:
        if not ds:
            continue
        candidates.append(CHAMPIONS_DIR / ds)

    # unique preservando orden
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)

    return out



def load_dataset_champion(
    dataset_id: str,
    *,
    family: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Carga el champion.json para un dataset (y opcionalmente por family)
    soportando el layout nuevo:
        artifacts/champions/<family>/<dataset_id>/champion.json
    y manteniendo compatibilidad con layout legacy.

    Garantiza campos derivados necesarios por schemas/routers:
    - source_run_id (derivable desde metrics.run_id si falta)
    - path (derivable desde el layout de champions aunque el json sea antiguo)
    """
    ds_dir = _champions_ds_dir(str(dataset_id), family=family)
    champion_path = ds_dir / "champion.json"

    champ = _try_read_json(champion_path)
    if not champ:
        return None

    # Asegura source_run_id (si el champion.json es antiguo puede no traerlo)
    champ = _ensure_source_run_id(champ, ds_dir=ds_dir)

        # Hidrata metrics si champion.json es "liviano" (layout nuevo):
    # Preferimos artifacts/runs/<source_run_id>/metrics.json
    metrics = champ.get("metrics")
    if not isinstance(metrics, dict) or not metrics.get("run_id"):
        src = champ.get("source_run_id")
        if src:
            run_metrics_path = RUNS_DIR / str(src) / "metrics.json"
            loaded = _try_read_json(run_metrics_path)
            if isinstance(loaded, dict) and loaded.get("run_id"):
                champ["metrics"] = loaded


    # Asegura path (para ChampionInfo; si el champion.json es antiguo puede no traerlo)
    if not champ.get("path"):
        model_name = champ.get("model_name")

        model_dir: Optional[Path] = None
        if model_name:
            cand = ds_dir / str(model_name)
            if cand.exists() and cand.is_dir():
                model_dir = cand

        # Fallback: primera carpeta dentro de ds_dir (excluye champion.json)
        if model_dir is None:
            try:
                for child in ds_dir.iterdir():
                    if child.is_dir():
                        model_dir = child
                        break
            except Exception:
                model_dir = None

        # Último fallback: ds_dir
        if model_dir is None:
            model_dir = ds_dir

        champ["path"] = str(model_dir.resolve())
        
        # Fallback adicional: si aún no hay metrics, intenta leerlos del directorio del champion
        metrics = champ.get("metrics")
        if not isinstance(metrics, dict) or not metrics.get("run_id"):
            mp = model_dir / "metrics.json"
            loaded = _try_read_json(mp)
            if isinstance(loaded, dict) and loaded.get("run_id"):
                champ["metrics"] = loaded


    return champ


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
    # Nota: meta.json es CRÍTICO para inferencia (feat_cols_, task_type, params de regresión, etc.)
    for fname in ("rbm.pt", "head.pt", "model.pt", "weights.pt", "vectorizer.json", "encoder.json", "meta.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)


def _extract_champion_context(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae campos clave (family/task_type/input_level/target_col/data_plan) de manera robusta
    desde distintas variantes de estructura en metrics.json.

    Soporta:
      - metrics.<field>
      - metrics.params.<field>
      - metrics.params.req.<field>
      - metrics.params.req.hparams.<field>
      - metrics.params.hparams.<field>
    """
    def _as_dict(x: Any) -> dict[str, Any]:
        return x if isinstance(x, dict) else {}

    m = _as_dict(metrics)
    params = _as_dict(m.get("params"))
    req = _as_dict(params.get("req"))
    hparams = _as_dict(req.get("hparams")) or _as_dict(params.get("hparams"))

    def pick(*candidates: Any) -> Any:
        for v in candidates:
            if v is None:
                continue
            # aceptar strings no vacíos
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
        return None

    family = pick(
        req.get("family"),
        params.get("family"),
        hparams.get("family"),
        m.get("family"),
    )

    task_type = pick(
        req.get("task_type"),
        params.get("task_type"),
        hparams.get("task_type"),
        m.get("task_type"),
    )

    input_level = pick(
        req.get("input_level"),
        params.get("input_level"),
        hparams.get("input_level"),
        m.get("input_level"),
    )

    target_col = pick(
        req.get("target_col"),
        params.get("target_col"),
        hparams.get("target_col"),
        m.get("target_col"),
    )

    data_plan = pick(
        req.get("data_plan"),
        params.get("data_plan"),
        hparams.get("data_plan"),
        m.get("data_plan"),
    )

    return {
        "family": family,
        "task_type": task_type,
        "input_level": input_level,
        "target_col": target_col,
        "data_plan": data_plan,
    }


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
        ctx = _extract_champion_context(metrics or {})
        payload = {
            "model_name": model_name,
            "dataset_id": dataset_id,
            "family": ctx.get("family"),
            "task_type": ctx.get("task_type"),
            "input_level": ctx.get("input_level"),
            "target_col": ctx.get("target_col"),
            "data_plan": ctx.get("data_plan"),
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
    dataset_id: str,
    run_id: str,
    model_name: Optional[str] = None,
    *,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = RUNS_DIR / str(run_id)
    metrics_path = run_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe metrics.json en run: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Inferir family si no viene explícita
    req = (metrics.get("params") or {}).get("req") or {}
    inferred_family = family or metrics.get("family") or req.get("family")
    if not inferred_family:
        raise ValueError("family es requerida para promover champion (no se pudo inferir desde metrics.json).")

    fam_slug = _slug(str(inferred_family))
    ds = str(dataset_id)

    # Modelo
    inferred_model = model_name or metrics.get("model_name") or req.get("modelo") or "model"
    inferred_model = str(inferred_model)

    # === NUEVO LAYOUT (principal) ===
    ds_dir = CHAMPIONS_DIR / fam_slug / ds
    dst_dir = ds_dir / inferred_model

    ds_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copiar snapshot run -> champion dir
    # (usa tu copytree/archivos existentes; este patrón es seguro en Win)
    shutil.copytree(run_dir, dst_dir, dirs_exist_ok=True)

    # Construir champion.json robusto
    champion = {
        "family": str(inferred_family),
        "dataset_id": ds,
        "model_name": inferred_model,
        "task_type": metrics.get("task_type") or req.get("task_type"),
        "input_level": metrics.get("input_level") or req.get("input_level"),
        "target_col": metrics.get("target_col") or req.get("target_col"),
        "data_plan": metrics.get("data_plan") or req.get("data_plan"),
        "source_run_id": str(run_id),
        "created_at": metrics.get("created_at") or dt.datetime.utcnow().isoformat() + "Z",
        "path": str(dst_dir),
    }

    champ_path = ds_dir / "champion.json"
    with open(champ_path, "w", encoding="utf-8") as f:
        json.dump(champion, f, ensure_ascii=False, indent=2)

    # === (Opcional) LEGACY MIRROR ===
    # Si aún tienes piezas antiguas leyendo artifacts/champions/<dataset_id>/champion.json,
    # espejea también (no rompe el layout nuevo).
    legacy_ds_dir = CHAMPIONS_DIR / ds
    legacy_ds_dir.mkdir(parents=True, exist_ok=True)
    with open(legacy_ds_dir / "champion.json", "w", encoding="utf-8") as f:
        json.dump(champion, f, ensure_ascii=False, indent=2)

    # También espejea snapshot (si lo necesitas)
    legacy_model_dir = legacy_ds_dir / inferred_model
    legacy_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, legacy_model_dir, dirs_exist_ok=True)

    return champion



def load_current_champion(
    *,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper de conveniencia usado por routers:
    - Soporta dataset_id o periodo (alias)
    - Soporta family (layout nuevo: artifacts/champions/<family>/<dataset_id>/...)
      con fallback a layout legacy si tu loader lo implementa.
    - Filtro opcional por model_name
    """
    ds = (dataset_id or periodo)
    if not ds:
        return None

    champ = load_dataset_champion(dataset_id=str(ds), family=family)
    if not champ:
        return None

    if model_name:
        req_model = str(model_name).strip()
        champ_model = champ.get("model_name") or champ.get("model")
        if champ_model and str(champ_model).strip() != req_model:
            return None
    
    # Robustez: si falta source_run_id pero hay metrics.run_id, derivarlo
    _ensure_source_run_id(champ)
            
    return champ



