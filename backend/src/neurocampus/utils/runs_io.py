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

from neurocampus.utils.model_context import fill_context


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

CHAMPION_SCHEMA_VERSION = 1


def _json_sanitize(payload: Any) -> Any:
    """Convierte `payload` a algo serializable por JSON (best-effort).

    Decisión:
    - Para métricas y metadatos puede llegar a haber tipos numpy, Path, datetime, etc.
    - Convertimos usando `default=str` y re-hidratamos con `json.loads`.
    - Si aún falla, caemos a `str(payload)` para nunca romper promote/champion.
    """
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        return str(payload)


def _artifacts_ref(p: Path) -> str:
    """Devuelve una referencia lógica estable tipo `artifacts/...` si vive bajo ARTIFACTS_DIR.

    Esto permite que el contrato sea portable aunque `NC_ARTIFACTS_DIR` apunte a otro disco.
    """
    p_res = Path(p).expanduser().resolve()
    try:
        rel = p_res.relative_to(ARTIFACTS_DIR)
        return str((Path("artifacts") / rel)).replace("\\", "/")
    except Exception:
        return str(p_res).replace("\\", "/")


_INVALID_RUN_IDS = {"", "null", "none", "nil"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_artifacts_dirs() -> None:
    ensure_dir(RUNS_DIR)
    ensure_dir(CHAMPIONS_DIR)


def now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")



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
    # Fallback: parsear dataset_id desde el nombre del run_dir/run_id:
    # <dataset_id>__<model_name>__<timestamp>__<job>
    rid = _norm_str(metrics.get("run_id")) or run_dir.name
    parts = str(rid).split("__")
    if len(parts) >= 1 and parts[0]:
        return str(parts[0])
    
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
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    job8 = _slug(job_id)[:8]
    return f"{_slug(dataset_id)}__{_slug(model_name)}__{ts}__{job8}"

def _find_metrics_path(run_dir: Path) -> Optional[Path]:
    """Encuentra el metrics.json soportando layouts legacy.

    Layout nuevo:
      artifacts/runs/<run_id>/metrics.json

    Layout legacy:
      artifacts/runs/<run_id>/model/metrics.json
    """
    candidates = [
        run_dir / "metrics.json",
        run_dir / "model" / "metrics.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _find_predictor_path(run_dir: Path) -> Optional[Path]:
    """Encuentra predictor.json soportando layouts legacy.

    Layout nuevo:
      artifacts/runs/<run_id>/predictor.json

    Layout legacy:
      artifacts/runs/<run_id>/model/predictor.json
    """
    candidates = [
        run_dir / "predictor.json",
        run_dir / "model" / "predictor.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _load_predictor_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = _find_predictor_path(run_dir)
    if not p:
        return None
    return _try_read_json(p)


def _build_champion_source_map(champions_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Mapa source_run_id -> payload de champion (family/dataset/model + metrics).

    Se usa para hidratar contexto cuando un run no trae metrics.json/predictor.json.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not champions_dir.exists():
        return out

    try:
        for p in champions_dir.glob("**/champion.json"):
            champ = _try_read_json(p)
            if not isinstance(champ, dict) or not champ:
                continue
            src = champ.get("source_run_id") or champ.get("run_id")
            if not src:
                continue
            metrics = champ.get("metrics") if isinstance(champ.get("metrics"), dict) else {}
            out[str(src)] = {
                "family": champ.get("family"),
                "dataset_id": champ.get("dataset_id"),
                "model_name": champ.get("model_name"),
                "metrics": metrics,
                "path": str(p),
            }
    except Exception:
        return out

    return out


def load_run_metrics(run_id: str) -> Dict[str, Any]:
    rd = (RUNS_DIR / str(run_id)).resolve()
    if not rd.exists():
        return {}

    mp = _find_metrics_path(rd)
    if not mp:
        return {}

    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
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

    champions_dir = (Path(base_dir) / "artifacts" / "champions").resolve() if base_dir is not None else CHAMPIONS_DIR
    champ_map = _build_champion_source_map(champions_dir)

    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for rd in run_dirs:
        # Cargar métricas (soporta layout legacy). Si no existen, intentar hidratar desde predictor/champion.
        mp = _find_metrics_path(rd)
        metrics = _try_read_json(mp) or {} if mp else {}

        run_id_val = _norm_str(metrics.get("run_id")) or rd.name

        # Fallback: hidratar métricas desde champion si coincide source_run_id
        if not metrics:
            champ = champ_map.get(str(run_id_val))
            if champ and isinstance(champ.get("metrics"), dict) and champ["metrics"]:
                metrics = champ["metrics"]

        predictor = _load_predictor_manifest(rd) or {}

        # Resolver dataset_id/model_name best-effort para filtros y contexto
        ds = (
            _norm_str(metrics.get("dataset_id"))
            or _norm_str(predictor.get("dataset_id"))
            or _infer_dataset_id(rd, metrics if metrics else {"run_id": run_id_val})
        )
        if ds_filter and ds != ds_filter:
            continue

        # Candidatos para contexto (precedencia final la aplica fill_context)
        predictor_family = None
        if isinstance(predictor.get("extra"), dict):
            predictor_family = predictor["extra"].get("family")
        predictor_family = predictor_family or predictor.get("family")

        champ = champ_map.get(str(run_id_val)) or {}
        family_guess = (
            _norm_str(_extract_ctx(metrics).get("family")) if metrics else None
        ) or _norm_str(predictor_family) or _norm_str(champ.get("family")) or _norm_str(family) or "sentiment_desempeno"

        model_guess = (
            _norm_str(metrics.get("model_name"))
            or _norm_str(metrics.get("model"))
            or _norm_str(predictor.get("model_name"))
            or _norm_str(champ.get("model_name"))
        )
        if not model_guess:
            parts = str(run_id_val).split("__")
            if len(parts) >= 2 and parts[1]:
                model_guess = str(parts[1])
        model_guess = model_guess or str(rd.name)

        # Contexto unificado (P2.1): evita null/unknown con precedencia definida
        ctx = fill_context(
            family=family_guess,
            dataset_id=ds,
            model_name=model_guess,
            metrics=metrics if metrics else None,
            predictor_manifest=predictor if predictor else None,
        )

        # Filtro family/model basado en contexto ya normalizado
        if fam_norm and _slug(ctx.get("family")) != fam_norm:
            continue
        if model_norm and _slug(ctx.get("model_name")) != model_norm:
            continue

        parts = str(run_id_val).split("__")

        created_at_val = _norm_str(metrics.get("created_at"))
        if not created_at_val and len(parts) >= 3 and parts[2]:
            try:
                dt_ = dt.datetime.strptime(str(parts[2]), "%Y%m%dT%H%M%SZ")
                created_at_val = dt_.replace(tzinfo=dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception:
                created_at_val = None
        if not created_at_val:
            try:
                created_at_val = dt.datetime.fromtimestamp(rd.stat().st_mtime, tz=dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception:
                created_at_val = now_utc_iso()


        summary: Dict[str, Any] = {
            "run_id": run_id_val,
            "dataset_id": str(ctx.get("dataset_id") or ds or ""),
            "model_name": str(ctx.get("model_name") or model_guess),
            "created_at": created_at_val,
            "artifact_path": str(rd),
            "family": ctx.get("family"),
            "task_type": ctx.get("task_type"),
            "input_level": ctx.get("input_level"),
            "target_col": ctx.get("target_col"),
            "data_plan": ctx.get("data_plan"),
            "data_source": ctx.get("data_source") or "feature_pack",
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

    mp = _find_metrics_path(rd)
    metrics = _try_read_json(mp) or {} if mp else {}

    # Fallback: hidratar desde champion/predictor si no existen métricas en el run
    if not metrics:
        champ_map = _build_champion_source_map(CHAMPIONS_DIR)
        champ = champ_map.get(str(run_id)) or champ_map.get(rd.name)
        if champ and isinstance(champ.get("metrics"), dict) and champ["metrics"]:
            metrics = champ["metrics"]

    predictor = _load_predictor_manifest(rd) or {}
    if not metrics:
        metrics = {"run_id": rd.name}

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
    # Si el run tiene contrato estandarizado (P2 Parte 4): usarlo directamente.
    pm = metrics.get("primary_metric")
    pm_mode = str(metrics.get("primary_metric_mode") or "min").lower()
    pm_value = metrics.get("primary_metric_value")
    if isinstance(pm, str) and pm and isinstance(pm_value, (int, float)):
        score_val = float(pm_value) if pm_mode == "max" else -float(pm_value)
        return (100, score_val)  # tier 100: contrato estandarizado, siempre gana a heurísticas

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
    """Copia artifacts relevantes del run al directorio del champion.

    Nota: los pesos reales viven en `run_dir/model/` (P2.x). Aquí copiamos:
    - metadata root del run (metrics/history/config/job_meta/predictor/preprocess/model.bin)
    - pesos de `run_dir/model/` (RBM: rbm.pt/head.pt/meta.json; DBM: dbm_state.npz/meta.json; + extras)
    """
    ensure_dir(target_dir)

    # 1) Root metadata del run
    for fname in (
        "metrics.json",
        "history.json",
        "config.snapshot.yaml",
        "job_meta.json",
        "predictor.json",
        "preprocess.json",
        "model.bin",
    ):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)

    # 2) Pesos y archivos dentro de run_dir/model/
    model_src = run_dir / "model"
    if model_src.is_dir():
        for p in model_src.iterdir():
            if p.is_file():
                shutil.copy2(p, target_dir / p.name)



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

def _build_champion_payload(
    *,
    dataset_id: str,
    family: str,
    model_name: str,
    source_run_id: str,
    metrics: Dict[str, Any],
    ds_dir: Path,
    model_dir: Path,
    promoted_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Construye un `champion.json` consistente (schema estable).

    Campos clave:
    - schema_version: versión de contrato del JSON.
    - source_run_id: run fuente (obligatorio).
    - metrics: métricas completas del run (para auditoría offline).
    - score: tupla (tier, value) usada para comparar champions.
    - paths: referencias lógicas `artifacts/...` + path físico (compat).

    Nota: mantenemos `created_at` por compatibilidad con versiones anteriores.
    """
    ts = promoted_at or now_utc_iso()
    tier, score_val = _champion_score(metrics or {})

    payload: Dict[str, Any] = {
        "schema_version": CHAMPION_SCHEMA_VERSION,
        "family": str(family),
        "dataset_id": str(dataset_id),
        "model_name": str(model_name),
        "source_run_id": str(source_run_id),
        # Compat: algunos consumers esperan created_at
        "created_at": ts,
        "promoted_at": ts,
        "updated_at": ts,
        "score": {"tier": int(tier), "value": float(score_val)},
        "metrics": _json_sanitize(metrics or {}),
        "paths": {
            "champion_ds_dir": _artifacts_ref(ds_dir),
            "champion_model_dir": _artifacts_ref(model_dir),
            "run_dir": _artifacts_ref(RUNS_DIR / str(source_run_id)),
            "run_metrics": _artifacts_ref(RUNS_DIR / str(source_run_id) / "metrics.json"),
        },
        # Mantener `path` por compat (algunos clientes lo muestran)
        "path": str(Path(model_dir).resolve()),
    }

    # Si el run ya trae created_at, lo preservamos aparte para auditoría
    run_created_at = (metrics or {}).get("created_at")
    if run_created_at:
        payload["run_created_at"] = run_created_at

    return payload


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
    """Compara contra champion actual y actualiza si mejora.

    - Escribe SIEMPRE champion POR MODELO en:
      artifacts/champions/<family>/<dataset>/<model>/champion.json
    - Mantiene champion GLOBAL del dataset (ds_dir/champion.json) solo si:
      (a) este run es el mejor de su modelo y
      (b) supera el score del champion global actual
    """

    req = _extract_req(metrics)
    fam = (family or metrics.get("family") or req.get("family"))
    fam = _norm_str(fam)

    ds_dir = _ensure_champions_ds_dir(dataset_id, family=fam)
    model_dir = ensure_dir(ds_dir / _slug(model_name))

    # source_run_id obligatorio en payload: inferir si no viene
    if not source_run_id:
        source_run_id = str(metrics.get("run_id") or metrics.get("source_run_id") or "")

    new_score = _champion_score(metrics)

    # ----------------------------
    # 1) Champion POR MODELO
    # ----------------------------
    model_champ_path = model_dir / "champion.json"
    current_model = _try_read_json(model_champ_path)
    old_model_score = (
        _champion_score((current_model or {}).get("metrics") or {})
        if current_model
        else (-1, float("-inf"))
    )
    promoted_model = (current_model is None) or (new_score > old_model_score)

    if promoted_model:
        # snapshot mínimo del modelo
        _write_json(model_dir / "metrics.json", metrics)

        # copiar artifacts del run al directorio del modelo champion
        if source_run_id:
            run_dir = (RUNS_DIR / str(source_run_id)).resolve()
            if run_dir.exists():
                _copy_run_artifacts_to_dir(run_dir, model_dir)

        payload_model = _build_champion_payload(
            dataset_id=str(dataset_id),
            family=str(fam or ""),
            model_name=str(model_name),
            source_run_id=str(source_run_id),
            metrics=metrics,
            ds_dir=ds_dir,
            model_dir=model_dir,
        )
        _write_json(model_champ_path, payload_model)

    # ----------------------------
    # 2) Champion GLOBAL (dataset)
    #    Solo si el run es campeón de su modelo y mejora el global
    # ----------------------------
    current_ds = load_dataset_champion(dataset_id, family=fam)
    old_ds_score = (
        _champion_score((current_ds or {}).get("metrics") or {})
        if current_ds
        else (-1, float("-inf"))
    )

    promoted_ds = bool(promoted_model) and ((current_ds is None) or (new_score > old_ds_score))

    if promoted_ds:
        # Reusamos el payload del modelo si ya lo construimos, o lo reconstruimos
        if promoted_model:
            payload_ds = payload_model
        else:
            payload_ds = _build_champion_payload(
                dataset_id=str(dataset_id),
                family=str(fam or ""),
                model_name=str(model_name),
                source_run_id=str(source_run_id),
                metrics=metrics,
                ds_dir=ds_dir,
                model_dir=model_dir,
            )

        _write_json(ds_dir / "champion.json", payload_ds)

        # mirror legacy best-effort
        try:
            legacy_ds_dir = _ensure_champions_ds_dir(dataset_id, family=None)
            legacy_model_dir = ensure_dir(legacy_ds_dir / _slug(model_name))

            _write_json(legacy_model_dir / "metrics.json", metrics)
            if source_run_id:
                run_dir = (RUNS_DIR / str(source_run_id)).resolve()
                if run_dir.exists():
                    _copy_run_artifacts_to_dir(run_dir, legacy_model_dir)

            payload_legacy = dict(payload_ds)
            payload_legacy["path"] = str(legacy_model_dir.resolve())
            if isinstance(payload_legacy.get("paths"), dict):
                payload_legacy["paths"] = dict(payload_legacy["paths"])
                payload_legacy["paths"]["champion_ds_dir"] = _artifacts_ref(legacy_ds_dir)
                payload_legacy["paths"]["champion_model_dir"] = _artifacts_ref(legacy_model_dir)

            _write_json(legacy_ds_dir / "champion.json", payload_legacy)

            # También mirror del champion por modelo en legacy (opcional pero útil)
            try:
                payload_legacy_model = dict(payload_legacy)
                payload_legacy_model["path"] = str(legacy_model_dir.resolve())
                _write_json(legacy_model_dir / "champion.json", payload_legacy_model)
            except Exception:
                pass

        except Exception:
            pass

    # Para compatibilidad con lo que ya consumía el job/UI:
    # - "promoted" mantiene el sentido histórico (promoción del champion global del dataset)
    return {
        "promoted": bool(promoted_ds),
        "promoted_model": bool(promoted_model),
        "old_score": old_ds_score,
        "new_score": new_score,
        "old_model_score": old_model_score,
        "champion_path": str((ds_dir / "champion.json").resolve()),
        "model_champion_path": str((model_dir / "champion.json").resolve()),
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

    champion_payload = _build_champion_payload(
        dataset_id=str(dataset_id),
        family=str(inferred_family),
        model_name=str(inferred_model),
        source_run_id=str(rid),
        metrics=metrics,
        ds_dir=ds_dir,
        model_dir=dst_dir,
    )
    _write_json(ds_dir / "champion.json", champion_payload)

    # legacy mirror best-effort
    try:
        legacy_ds_dir = _ensure_champions_ds_dir(dataset_id, family=None)
        legacy_model_dir = ensure_dir(legacy_ds_dir / _slug(inferred_model))
        _copy_run_artifacts_to_dir(run_dir, legacy_model_dir)
        legacy_payload = dict(champion_payload)
        legacy_payload["path"] = str(legacy_model_dir.resolve())
        if isinstance(legacy_payload.get("paths"), dict):
            legacy_payload["paths"] = dict(legacy_payload["paths"])
            legacy_payload["paths"]["champion_ds_dir"] = _artifacts_ref(legacy_ds_dir)
            legacy_payload["paths"]["champion_model_dir"] = _artifacts_ref(legacy_model_dir)
        _write_json(legacy_ds_dir / "champion.json", legacy_payload)
    except Exception:
        pass

    champion_api = dict(champion_payload)
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
