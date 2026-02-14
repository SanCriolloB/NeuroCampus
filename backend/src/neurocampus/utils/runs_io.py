from __future__ import annotations

"""neurocampus.utils.runs_io

P0: IO estable para artifacts de Modelos.

Este módulo es la *fuente de verdad* para:

- Runs: ``artifacts/runs/<run_id>/`` (metrics/history/config/job_meta)
- Champions: ``artifacts/champions/<family>/<dataset_id>/champion.json`` (+ snapshot)

Problemas corregidos en esta reescritura:
- Respetar ``NC_ARTIFACTS_DIR`` (antes se sobreescribía RUNS_DIR y se ignoraba el env var).
- Listado de runs consistente incluso cuando el backend se ejecuta desde ``backend/``.
- Validación robusta de ``run_id`` (evita 'null' y errores leyendo paths).
- Promote champion usa el mismo layout/slugging que load_dataset_champion.
- Elimina duplicidad de helpers y referencias a funciones inexistentes.

Compatibilidad:
- Se mantienen los nombres públicos usados por routers: ``build_run_id``, ``save_run``, ``list_runs``,
  ``load_run_details``, ``load_run_metrics``, ``maybe_update_champion``, ``promote_run_to_champion``,
  ``load_dataset_champion``, ``load_current_champion``, ``champion_score``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import json
import os
import re
import shutil

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Encuentra una raíz razonable del repo.

    1) Si existe NC_PROJECT_ROOT, úsalo.
    2) Busca hacia arriba un directorio con Makefile o backend/.
    3) Fallback al cwd.

    Nota: en el router ``modelos.py`` ya se fija ``NC_ARTIFACTS_DIR`` antes de importar este módulo.
    """
    env_root = os.getenv("NC_PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "Makefile").exists() or (p / "backend").is_dir():
            return p

    return Path.cwd().resolve()


BASE_DIR: Path = _find_project_root()

# La fuente de verdad para artifacts debe ser NC_ARTIFACTS_DIR si existe.
_ART_ENV = os.getenv("NC_ARTIFACTS_DIR")
ARTIFACTS_DIR: Path = Path(_ART_ENV).expanduser().resolve() if _ART_ENV else (BASE_DIR / "artifacts").resolve()
RUNS_DIR: Path = (ARTIFACTS_DIR / "runs").resolve()
CHAMPIONS_DIR: Path = (ARTIFACTS_DIR / "champions").resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INVALID_RUN_IDS = {"", "null", "none", "nil"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_artifacts_dirs() -> None:
    ensure_dir(RUNS_DIR)
    ensure_dir(CHAMPIONS_DIR)


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slug(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_") or "x"


# Alias público (compat)
slug = _slug


def _norm_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = _read_json(path)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_yaml(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _deep_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_req(metrics: Dict[str, Any]) -> Dict[str, Any]:
    params = metrics.get("params")
    if isinstance(params, dict):
        req = params.get("req")
        return req if isinstance(req, dict) else {}
    return {}


def _extract_ctx(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Contexto estable para UI/API, leído desde top-level o params.req."""
    req = _extract_req(metrics)

    def pick(key: str) -> Any:
        return req.get(key) if req.get(key) is not None else metrics.get(key)

    return {
        "family": pick("family"),
        "task_type": pick("task_type"),
        "input_level": pick("input_level"),
        "target_col": pick("target_col"),
        "data_plan": pick("data_plan"),
        "data_source": pick("data_source"),
        "target_mode": pick("target_mode"),
        "split_mode": pick("split_mode"),
        "val_ratio": pick("val_ratio"),
        "window_k": req.get("window_k"),
        "replay_size": req.get("replay_size"),
        "replay_strategy": req.get("replay_strategy"),
        "warm_start_from": req.get("warm_start_from"),
        "warm_start_run_id": req.get("warm_start_run_id"),
    }


_DATASET_STEM_RE = re.compile(r"^(?P<base>.+?)(_beto.*)?$", re.IGNORECASE)


def _infer_dataset_id_from_path(path_str: str) -> Optional[str]:
    try:
        name = Path(str(path_str)).name
        stem = Path(name).stem
        m = _DATASET_STEM_RE.match(stem)
        return (m.group("base") if m else stem) or None
    except Exception:
        return None


def _load_yaml_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return None


def _infer_dataset_id(run_dir: Path, metrics: Dict[str, Any]) -> Optional[str]:
    for k in ("dataset_id", "dataset", "periodo"):
        v = _norm_str(metrics.get(k))
        if v:
            return v

    cfg = _load_yaml_if_exists(run_dir / "config.snapshot.yaml") or _load_yaml_if_exists(run_dir / "config.yaml")
    if cfg:
        ds = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else None
        if isinstance(ds, dict):
            v = _norm_str(ds.get("id") or ds.get("dataset_id") or ds.get("periodo"))
            if v:
                return v
            p = ds.get("path")
            if p:
                v2 = _infer_dataset_id_from_path(str(p))
                if v2:
                    return v2
        p2 = cfg.get("dataset_path")
        if p2:
            v3 = _infer_dataset_id_from_path(str(p2))
            if v3:
                return v3

    return None


def _data_meta_from_data_ref(data_ref: Optional[str]) -> Optional[Dict[str, Any]]:
    """Metadata mínima para auditoría.

    Si el data_ref apunta a un feature-pack, intenta leer meta.json/pair_meta.json.
    """
    if not data_ref:
        return None

    p = Path(data_ref)
    if not p.is_absolute():
        # data_ref suele ser relativo al repo root; si está relativo, lo resolvemos contra BASE_DIR.
        p = (BASE_DIR / p).resolve()

    if not p.exists():
        return None

    meta: Dict[str, Any] = {"data_ref_basename": p.name}

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
                    "n_columns": len(m["columns"]) if isinstance(m.get("columns"), list) else None,
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
                }
            )

    meta = {k: v for k, v in meta.items() if v is not None}
    return meta or None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def build_run_id(dataset_id: str, model_name: str, job_id: str) -> str:
    """Construye un run_id único y legible."""
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    job8 = _slug(job_id)[:8]
    return f"{_slug(dataset_id)}__{_slug(model_name)}__{ts}__{job8}"


def load_run_metrics(run_id: str) -> Dict[str, Any]:
    p = (RUNS_DIR / str(run_id) / "metrics.json").resolve()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_run(
    *,
    run_id: str,
    job_id: str,
    dataset_id: str,
    model_name: str,
    data_ref: str | None,
    params: Dict[str, Any] | None,
    final_metrics: Dict[str, Any] | None,
    history: List[Dict[str, Any]] | None,
) -> Path:
    """Persiste un run en ``artifacts/runs/<run_id>/``.

    Escribe:
    - metrics.json (flatten + params + history + contexto + data_meta)
    - history.json
    - job_meta.json
    - config.snapshot.yaml
    """
    ensure_artifacts_dirs()

    run_dir = ensure_dir(RUNS_DIR / str(run_id))
    created_at = now_utc_iso()

    fm = dict(final_metrics or {})
    hist = list(history or [])

    # (A) job_meta
    _write_json(
        run_dir / "job_meta.json",
        {
            "run_id": str(run_id),
            "job_id": str(job_id),
            "dataset_id": str(dataset_id),
            "model_name": str(model_name),
            "created_at": created_at,
            "data_ref": data_ref,
        },
    )

    # (B) snapshot config
    _write_yaml(run_dir / "config.snapshot.yaml", params or {})

    # (C) history
    _write_json(run_dir / "history.json", hist)

    # (D) metrics principal
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "job_id": str(job_id),
        "dataset_id": str(dataset_id),
        "model_name": str(model_name),
        "created_at": created_at,
        "data_ref": data_ref,
        "params": params or {},
        "history": hist,
    }

    # flatten métricas
    for k, v in fm.items():
        payload[k] = v

    # contexto estable
    try:
        ctx = _extract_ctx(payload)
        for k, v in ctx.items():
            if v is not None:
                payload[k] = v
    except Exception:
        pass

    # data_meta best-effort
    try:
        dm = _data_meta_from_data_ref(data_ref)
        if dm is not None:
            payload["data_meta"] = dm
    except Exception:
        pass

    _write_json(run_dir / "metrics.json", payload)
    return run_dir


def list_runs(
    base_dir: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
    family: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Lista runs (más recientes primero)."""

    ds_filter = dataset_id or periodo

    runs_dir = (Path(base_dir) / "artifacts" / "runs").resolve() if base_dir is not None else RUNS_DIR
    if not runs_dir.exists():
        return []

    fam_norm = _slug(family) if family else None
    model_norm = _slug(model_name) if model_name else None

    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for rd in run_dirs:
        mp = rd / "metrics.json"
        if not mp.exists():
            continue

        metrics = _try_read_json(mp) or {}
        if not metrics:
            continue

        ds = _norm_str(metrics.get("dataset_id")) or _infer_dataset_id(rd, metrics)
        if ds_filter and ds != ds_filter:
            continue

        ctx = _extract_ctx(metrics)

        if fam_norm:
            rf = ctx.get("family") or metrics.get("family")
            if _slug(rf) != fam_norm:
                continue

        if model_norm:
            rm = metrics.get("model_name") or metrics.get("model")
            if _slug(rm) != model_norm:
                continue

        summary: Dict[str, Any] = {
            "run_id": metrics.get("run_id") or rd.name,
            "dataset_id": ds,
            "model_name": metrics.get("model_name"),
            "created_at": metrics.get("created_at"),
            "artifact_path": str(rd),
            "family": ctx.get("family"),
            "task_type": ctx.get("task_type"),
            "input_level": ctx.get("input_level"),
            "target_col": ctx.get("target_col"),
            "data_plan": ctx.get("data_plan"),
            "metrics": {},
        }

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
        ]
        for k in keep:
            v = metrics.get(k)
            if v is not None:
                summary["metrics"][k] = v

        out.append(summary)
        if len(out) >= int(limit):
            break

    return out


def load_run_details(run_id: str) -> Optional[Dict[str, Any]]:
    rd = (RUNS_DIR / str(run_id)).resolve()
    if not rd.exists():
        return None

    mp = rd / "metrics.json"
    if not mp.exists():
        return None

    metrics = _try_read_json(mp) or {}
    cfg = _load_yaml_if_exists(rd / "config.snapshot.yaml") or _load_yaml_if_exists(rd / "config.yaml")
    ds = _norm_str(metrics.get("dataset_id")) or _infer_dataset_id(rd, metrics)

    return {
        "run_id": str(run_id),
        "dataset_id": ds,
        "metrics": metrics,
        "config": cfg,
        "artifact_path": str(rd),
    }


# ---------------------------------------------------------------------------
# Champions
# ---------------------------------------------------------------------------

def _champion_score(metrics: Dict[str, Any]) -> Tuple[int, float]:
    """Retorna (tier, score). Mayor es mejor."""
    task_type = str(metrics.get("task_type") or "").lower().strip()

    is_regression = (
        task_type == "regression"
        or any(
            k in metrics
            for k in ("val_rmse", "val_mae", "val_r2", "train_rmse", "train_mae", "train_r2")
        )
    )

    if is_regression:
        if isinstance(metrics.get("val_rmse"), (int, float)):
            return (60, -float(metrics["val_rmse"]))
        if isinstance(metrics.get("val_mae"), (int, float)):
            return (50, -float(metrics["val_mae"]))
        if isinstance(metrics.get("val_r2"), (int, float)):
            return (40, float(metrics["val_r2"]))
        if isinstance(metrics.get("train_rmse"), (int, float)):
            return (30, -float(metrics["train_rmse"]))
        if isinstance(metrics.get("train_mae"), (int, float)):
            return (20, -float(metrics["train_mae"]))
        if isinstance(metrics.get("train_r2"), (int, float)):
            return (10, float(metrics["train_r2"]))
        if isinstance(metrics.get("loss"), (int, float)):
            return (0, -float(metrics["loss"]))
        return (-1, float("-inf"))

    # classification
    if isinstance(metrics.get("val_f1_macro"), (int, float)):
        return (4, float(metrics["val_f1_macro"]))
    if isinstance(metrics.get("f1_macro"), (int, float)):
        return (3, float(metrics["f1_macro"]))
    if isinstance(metrics.get("val_accuracy"), (int, float)):
        return (2, float(metrics["val_accuracy"]))
    if isinstance(metrics.get("accuracy"), (int, float)):
        return (1, float(metrics["accuracy"]))
    if isinstance(metrics.get("loss"), (int, float)):
        return (0, -float(metrics["loss"]))
    return (-1, float("-inf"))


def champion_score(metrics: Dict[str, Any]) -> Tuple[int, float]:
    return _champion_score(metrics)


def _champions_ds_dir(dataset_id: str, family: Optional[str] = None) -> Path:
    ds_slug = _slug(dataset_id)
    if family:
        fam_slug = _slug(family)
        return (CHAMPIONS_DIR / fam_slug / ds_slug).resolve()
    return (CHAMPIONS_DIR / ds_slug).resolve()


def _ensure_champions_ds_dir(dataset_id: str, family: Optional[str] = None) -> Path:
    return ensure_dir(_champions_ds_dir(dataset_id, family=family))


def _copy_run_artifacts_to_dir(run_dir: Path, target_dir: Path) -> None:
    """Copia artifacts relevantes del run al directorio del champion."""
    ensure_dir(target_dir)

    for fname in ("metrics.json", "history.json", "config.snapshot.yaml", "job_meta.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)

    # pesos / extras comunes
    for fname in (
        "rbm.pt",
        "head.pt",
        "model.pt",
        "weights.pt",
        "vectorizer.json",
        "encoder.json",
        "meta.json",
    ):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)


def _ensure_source_run_id(champ: Dict[str, Any]) -> Dict[str, Any]:
    if not champ.get("source_run_id"):
        metrics = champ.get("metrics")
        rid = None
        if isinstance(metrics, dict):
            rid = metrics.get("run_id")
        rid = rid or champ.get("run_id")
        if rid:
            champ["source_run_id"] = str(rid)
    return champ


def load_dataset_champion(dataset_id: str, *, family: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Carga champion.json (layout nuevo por family, con fallback legacy)."""

    candidates: List[Path] = []

    if family:
        candidates.append(_champions_ds_dir(dataset_id, family=family) / "champion.json")
        # fallback: family slugging ya es interno, pero probamos también sin family

    candidates.append(_champions_ds_dir(dataset_id, family=None) / "champion.json")

    champ = None
    champ_path = None
    for p in candidates:
        payload = _try_read_json(p)
        if isinstance(payload, dict) and payload:
            champ = payload
            champ_path = p
            break

    if not champ:
        return None

    champ = _ensure_source_run_id(champ)

    # Hidratar metrics si el champion es liviano
    if not isinstance(champ.get("metrics"), dict) or not champ["metrics"].get("run_id"):
        src = champ.get("source_run_id")
        if src:
            rm = _try_read_json(RUNS_DIR / str(src) / "metrics.json")
            if isinstance(rm, dict) and rm.get("run_id"):
                champ["metrics"] = rm

    if champ_path and not champ.get("path"):
        champ["path"] = str(champ_path.parent.resolve())

    return champ


def maybe_update_champion(
    *,
    dataset_id: str,
    model_name: str,
    metrics: Dict[str, Any],
    source_run_id: Optional[str] = None,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """Compara contra champion actual y actualiza si mejora."""

    req = _extract_req(metrics)
    fam = (family or metrics.get("family") or req.get("family"))
    fam = _norm_str(fam)

    ds_dir = _ensure_champions_ds_dir(dataset_id, family=fam)
    model_dir = ensure_dir(ds_dir / _slug(model_name))

    current = load_dataset_champion(dataset_id, family=fam)
    old_score = _champion_score((current or {}).get("metrics") or {}) if current else (-1, float("-inf"))
    new_score = _champion_score(metrics)

    promoted = (current is None) or (new_score > old_score)

    if promoted:
        # snapshot mínimo
        _write_json(model_dir / "metrics.json", metrics)

        if source_run_id:
            run_dir = (RUNS_DIR / str(source_run_id)).resolve()
            if run_dir.exists():
                _copy_run_artifacts_to_dir(run_dir, model_dir)

        champion_payload: Dict[str, Any] = {
            "family": fam,
            "dataset_id": str(dataset_id),
            "model_name": str(model_name),
            "source_run_id": str(source_run_id) if source_run_id else None,
            "metrics": metrics,
            "path": str(model_dir.resolve()),
            "updated_at": now_utc_iso(),
        }
        _write_json(ds_dir / "champion.json", champion_payload)

        # mirror legacy best-effort
        try:
            legacy_ds_dir = _ensure_champions_ds_dir(dataset_id, family=None)
            legacy_model_dir = ensure_dir(legacy_ds_dir / _slug(model_name))
            _write_json(legacy_model_dir / "metrics.json", metrics)
            if source_run_id:
                run_dir = (RUNS_DIR / str(source_run_id)).resolve()
                if run_dir.exists():
                    _copy_run_artifacts_to_dir(run_dir, legacy_model_dir)
            payload_legacy = dict(champion_payload)
            payload_legacy["path"] = str(legacy_model_dir.resolve())
            _write_json(legacy_ds_dir / "champion.json", payload_legacy)
        except Exception:
            pass

    return {
        "promoted": bool(promoted),
        "old_score": old_score,
        "new_score": new_score,
        "champion_path": str((ds_dir / "champion.json").resolve()),
    }


def promote_run_to_champion(
    dataset_id: str,
    run_id: str,
    model_name: Optional[str] = None,
    *,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """Promueve un run existente a champion (manual)."""

    rid = str(run_id or "").strip()
    if rid.lower() in _INVALID_RUN_IDS:
        raise ValueError("run_id inválido")

    run_dir = (RUNS_DIR / rid).resolve()
    metrics_path = run_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe metrics.json para run_id={rid}")

    metrics = _read_json(metrics_path)
    req = _extract_req(metrics)

    inferred_family = _norm_str(family or metrics.get("family") or req.get("family"))
    if not inferred_family:
        raise ValueError("family es requerida para promover champion (no se pudo inferir desde metrics.json).")

    inferred_model = _norm_str(model_name or metrics.get("model_name") or req.get("modelo") or "model")
    inferred_model = inferred_model or "model"

    ds_dir = _ensure_champions_ds_dir(dataset_id, family=inferred_family)
    dst_dir = ensure_dir(ds_dir / _slug(inferred_model))

    # Copia artifacts run -> champion
    _copy_run_artifacts_to_dir(run_dir, dst_dir)

    champion = {
        "family": inferred_family,
        "dataset_id": str(dataset_id),
        "model_name": inferred_model,
        "task_type": metrics.get("task_type") or req.get("task_type"),
        "input_level": metrics.get("input_level") or req.get("input_level"),
        "target_col": metrics.get("target_col") or req.get("target_col"),
        "data_plan": metrics.get("data_plan") or req.get("data_plan"),
        "source_run_id": rid,
        "created_at": metrics.get("created_at") or now_utc_iso(),
        "path": str(dst_dir.resolve()),
    }

    _write_json(ds_dir / "champion.json", champion)

    # legacy mirror best-effort
    try:
        legacy_ds_dir = _ensure_champions_ds_dir(dataset_id, family=None)
        legacy_model_dir = ensure_dir(legacy_ds_dir / _slug(inferred_model))
        _copy_run_artifacts_to_dir(run_dir, legacy_model_dir)
        legacy_payload = dict(champion)
        legacy_payload["path"] = str(legacy_model_dir.resolve())
        _write_json(legacy_ds_dir / "champion.json", legacy_payload)
    except Exception:
        pass

    champion_api = dict(champion)
    champion_api["metrics"] = metrics
    return champion_api


def load_current_champion(
    *,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    ds = dataset_id or periodo
    if not ds:
        return None

    champ = load_dataset_champion(str(ds), family=family)
    if not champ:
        return None

    if model_name:
        req_model = _slug(model_name)
        champ_model = champ.get("model_name") or champ.get("model")
        if champ_model and _slug(champ_model) != req_model:
            return None

    return _ensure_source_run_id(champ)
