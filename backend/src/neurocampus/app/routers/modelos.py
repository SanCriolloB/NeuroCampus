"""
neurocampus.app.routers.modelos
================================

Router de **Modelos** (FastAPI) para NeuroCampus.

Incluye:
- Entrenamiento async (BackgroundTasks)
- Estado de jobs (polling)
- Readiness (insumos disponibles)
- Runs y champion (auditoría y selección del mejor modelo)
- Promote manual (opcional)

Correcciones clave
------------------
  - Evitar reutilización de instancias de estrategia entre jobs.
    Se usan CLASES (factory) y se crea una instancia NUEVA por entrenamiento.

  - Resetear runtime-state si el strategy expone un método de reset
    (reset / _reset_runtime_state / reset_state / clear_state).
  - Esto mitiga estados “fantasma” por hot-reload o referencias persistentes.

  - FIX: No pasar valores None dentro de hparams hacia el training.
    Especialmente `teacher_materia_mode`, para evitar que el strategy reciba None
    y lo convierta en 'none' (string), deshabilitando teacher/materia por accidente.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import re
import os
import time
import uuid
import math
import inspect
import logging
import json
import datetime as dt
logger = logging.getLogger(__name__)
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ..schemas.modelos import (
    EntrenarRequest,
    EntrenarResponse,
    EstadoResponse,
    ReadinessResponse,
    PromoteChampionRequest,
    RunSummary,
    RunDetails,
    ChampionInfo,
    SweepEntrenarRequest,
    SweepEntrenarResponse,
    SweepSummary,
    SweepCandidate,
    ModelSweepRequest,
    ModelSweepCandidateResult,
    ModelSweepResponse,
)

from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida
from ...models.strategies.dbm_manual_strategy import DBMManualPlantillaStrategy
from neurocampus.predictions.bundle import build_predictor_manifest, bundle_paths, write_json
from ...utils.model_context import fill_context
from ...utils.warm_start import resolve_warm_start_path
from ...models.utils.metrics_contract import standardize_run_metrics, primary_metric_for_family


from ...observability.bus_eventos import BUS
from ...models.observer.eventos_entrenamiento import emit_training_persisted

# Selección de datos por metodología (periodo_actual / acumulado / ventana)
try:
    from ...models.strategies.metodologia import SeleccionConfig, resolver_metodologia
except Exception:
    # Shim defensivo (no debería ocurrir si el repo está completo).
    class SeleccionConfig:  # type: ignore
        def __init__(self, periodo_actual=None, ventana_n=4): ...
    def resolver_metodologia(nombre: str):  # type: ignore
        raise RuntimeError(
            "El módulo de metodologías no está disponible. "
            "Asegúrate de tener neurocampus/models/strategies/metodologia.py"
        )

# Resolver labeled (heurística BETO/teacher)
from ...data.datos_dashboard import resolve_labeled_path

router = APIRouter()


# ---------------------------------------------------------------------------
# Base path (raíz del repo)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Encuentra la raíz del repo de NeuroCampus de forma robusta.

    Criterio:
    - Un directorio que contenga `data/` y `datasets/`.

    Esto evita errores cuando el servidor se lanza desde `backend/` u otra ruta.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data").exists() and (p / "datasets").exists():
            return p
    # Fallback defensivo.
    return here.parents[5]


BASE_DIR: Path = _find_project_root()

# Asegurar que runs_io escriba en el artifacts del mismo BASE_DIR
# (importante si uvicorn se ejecuta desde otra carpeta).
if "NC_ARTIFACTS_DIR" not in os.environ:
    os.environ["NC_ARTIFACTS_DIR"] = str((BASE_DIR / "artifacts").resolve())

ARTIFACTS_DIR: Path = Path(os.environ["NC_ARTIFACTS_DIR"]).expanduser().resolve()

# Importar runs_io DESPUÉS de fijar NC_ARTIFACTS_DIR
from ...utils.runs_io import (  # noqa: E402
    build_run_id,
    save_run,
    maybe_update_champion,
    promote_run_to_champion,
    list_runs,
    load_run_details,
    load_current_champion,
    load_dataset_champion,
    is_deployable_for_predictions,
)


def _relpath(p: Path) -> str:
    """Devuelve una ruta *lógica* estable para UI/contratos.

    Regla:
    - Si el path vive bajo NC_ARTIFACTS_DIR => devuelve "artifacts/<...>" (lógico).
    - Si vive bajo BASE_DIR => devuelve path relativo a BASE_DIR.
    - Si no => devuelve absoluto.

    Nota: normaliza separadores a "/" para estabilidad cross-platform.
    """
    p_res = Path(p).expanduser().resolve()

    # 1) Prioridad: artifacts lógicos (si el path vive en ARTIFACTS_DIR)
    try:
        rel_art = p_res.relative_to(ARTIFACTS_DIR)
        return str((Path("artifacts") / rel_art)).replace("\\", "/")
    except Exception:
        pass

    # 2) Fallback: relativo a BASE_DIR
    try:
        rel_base = p_res.relative_to(BASE_DIR.resolve())
        return str(rel_base).replace("\\", "/")
    except Exception:
        return str(p_res).replace("\\", "/")



def _strip_localfs(uri: str) -> str:
    """
    Convierte un URI estilo ``localfs://...`` a una ruta local.
    """
    if isinstance(uri, str) and uri.startswith("localfs://"):
        return uri.replace("localfs://", "", 1)
    return uri


def _abs_path(ref: str) -> Path:
    """Resuelve un ref (relativo/absoluto o localfs://) a un Path absoluto.

    Regla:
    - Si es absoluto => se respeta.
    - Si empieza por "artifacts/..." => se resuelve bajo ARTIFACTS_DIR (NC_ARTIFACTS_DIR).
    - En otro caso => se resuelve bajo BASE_DIR.

    Esto permite que los contratos sigan usando "artifacts/..." aunque el storage
    real esté fuera del repo (p. ej. volumen montado).
    """
    raw = _strip_localfs(ref)
    p = Path(raw)

    if p.is_absolute():
        return p.resolve()

    norm = str(raw).replace("\\", "/").lstrip("/")
    if norm.startswith("artifacts/"):
        sub = Path(norm).relative_to("artifacts")
        return (ARTIFACTS_DIR / sub).resolve()

    return (BASE_DIR / p).resolve()

def _try_write_predictor_bundle(
    *,
    run_dir: "Path | str",
    req_norm: Any,
    metrics: dict[str, Any] | None,
    strategy: Any | None = None,
) -> None:
    """Best-effort: escribe el predictor bundle en el run_dir.

    Contrato P2.x:
    - `predictor.json` SIEMPRE (si el run llegó a completed).
    - `preprocess.json` placeholder si no hay pipeline formal.
    - `model.bin`:
        - si existe persistencia real -> NO placeholder (marca “READY”)
        - si no -> placeholder

    Decisiones:
    - No debe romper P0/P1/P2: cualquier error aquí se loguea y se ignora.
    - Si no hay `save()` en la estrategia, dejamos placeholder y /predicciones/predict responderá 422.
    """
    try:
        from pathlib import Path as _Path

        run_path = _Path(run_dir).expanduser().resolve()
        bp = bundle_paths(run_path)

        # Preferir el metrics.json persistido (incluye params.req).
        # Esto evita que predictor.json quede con extra=null aunque el request sí lo tenía.
        metrics_payload = metrics or {}
        try:
            mp = run_path / "metrics.json"
            if mp.exists():
                metrics_payload = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("No se pudo leer metrics.json para contexto; se usa metrics in-memory")


        # ------------------------------------------------------------
        # 1) Persistencia real del modelo (si la estrategia soporta save())
        # ------------------------------------------------------------
        # P2.3 FIX: Definir ``model_dir`` — subdirectorio ``model/`` dentro
        # del run donde se persisten los artefactos serializados del modelo
        # (meta.json, dbm_state.npz, rbm.pt, head.pt, etc.).
        # Sin esta definición, ``strategy.save()`` fallaba con NameError y
        # dejaba la carpeta ``model/`` vacía, rompiendo DBM y warm-start.
        model_dir = run_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            save_fn = getattr(strategy, "save", None)
            if callable(save_fn):
                # Llamada robusta (posibles diferencias de firma entre estrategias)
                _call_with_accepted_kwargs(save_fn, out_dir=str(model_dir))

                # Validación mínima: si quedó vacío, lo tratamos como fallo de export
                present = [p.name for p in Path(model_dir).iterdir() if p.is_file()]
                if not present:
                    raise RuntimeError(
                        f"Export de modelo dejó model/ vacío en {model_dir}. "
                        "Revisa strategy.save() y logs del backend."
                    )
        except Exception:
            logger.exception("No se pudo persistir modelo en run_dir=%s (best-effort)", str(run_path))

        # ------------------------------------------------------------
        # 1b) Garantía de contrato de export (tests/dev)
        # ------------------------------------------------------------
        # En producción, un export incompleto debe marcar el job como failed
        # vía ``_require_exported_model()``. Sin embargo, en tests de contrato
        # (donde se monkeypatch-ea el entrenamiento para no crear modelos reales),
        # necesitamos un export mínimo para no romper el flujo del API.
        #
        # Regla: sólo generar placeholders cuando:
        #   - estamos corriendo bajo pytest (``PYTEST_CURRENT_TEST``), o
        #   - ``NC_ALLOW_PLACEHOLDER_EXPORT=1`` (flag explícito).
        #
        # Nota: estos placeholders NO pretenden ser cargables para inferencia; son
        # únicamente para cumplir contratos de persistencia en pruebas.
        try:
            present_set = {p.name for p in model_dir.iterdir() if p.is_file()}
        except Exception:
            present_set = set()

        model_name_hint = str(
            getattr(req_norm, "modelo", "")
            or metrics_payload.get("model_name")
            or metrics_payload.get("model")
            or ""
        )
        family_hint = str(getattr(req_norm, "family", "") or metrics_payload.get("family") or "")
        dataset_hint = str(getattr(req_norm, "dataset_id", "") or metrics_payload.get("dataset_id") or "")

        mn = model_name_hint.lower().strip()

        def _allow_placeholder_export() -> bool:
            return bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("NC_ALLOW_PLACEHOLDER_EXPORT") == "1"

        def _write_meta_placeholder() -> None:
            meta_path = model_dir / "meta.json"
            if meta_path.exists():
                return

            payload = {
                "schema_version": 1,
                "placeholder_export": True,
                "model_name": model_name_hint,
                "family": family_hint,
                "dataset_id": dataset_hint,
                "exported_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
                "notes": (
                    "Artefactos placeholder generados en contexto de pruebas/dev "
                    "(sin entrenamiento real). En producción, esto NO debería ocurrir."
                ),
            }
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        def _touch(path: Path, content: bytes) -> None:
            if not path.exists():
                path.write_bytes(content)

        needs = False
        if mn.startswith("rbm"):
            needs = ("meta.json" not in present_set) or not ({"rbm.pt", "head.pt"} & present_set)
            if needs and _allow_placeholder_export():
                _write_meta_placeholder()
                _touch(model_dir / "rbm.pt", b"PLACEHOLDER_RBM_WEIGHTS_P2")
                _touch(model_dir / "head.pt", b"PLACEHOLDER_HEAD_WEIGHTS_P2")

        elif mn.startswith("dbm"):
            needs = not {"meta.json", "dbm_state.npz"} <= present_set
            if needs and _allow_placeholder_export():
                _write_meta_placeholder()
                npz_path = model_dir / "dbm_state.npz"
                if not npz_path.exists():
                    try:
                        np.savez_compressed(npz_path, placeholder=np.array([], dtype=np.float32))
                    except Exception:
                        _touch(npz_path, b"PLACEHOLDER_DBM_STATE_NPZ_P2")

        # Refrescar presencia tras placeholders (para trazabilidad downstream)
        if needs and _allow_placeholder_export():
            try:
                present_set = {p.name for p in model_dir.iterdir() if p.is_file()}
            except Exception:
                pass

        # ------------------------------------------------------------
        # 2) Manifest predictor.json (siempre)
        # ------------------------------------------------------------
        dataset_id = str(getattr(req_norm, "dataset_id", "") or metrics.get("dataset_id") or "")
        model_name = str(getattr(req_norm, "modelo", "") or metrics.get("model_name") or metrics.get("model") or "")
        family = str(getattr(req_norm, "family", "") or metrics.get("family") or "")

        # Completar contexto (regla única: metrics.params.req -> metrics.* -> predictor.json -> fallback por family)
        ctx = fill_context(
            family=family or None,
            dataset_id=dataset_id or None,
            model_name=model_name or None,
            metrics=metrics_payload,
            predictor_manifest=None,
        )
        dataset_id = str((ctx.get("dataset_id") or dataset_id) or "")
        model_name = str((ctx.get("model_name") or model_name) or "")
        family = str((ctx.get("family") or family) or "")

        # P2.6: Leer trazabilidad de features del modelo serializado
        # si la estrategia persistió meta.json con conteos de texto.
        _feat_trace: dict[str, Any] = {}
        try:
            _model_meta_path = model_dir / "meta.json"
            if _model_meta_path.exists():
                _model_meta = json.loads(_model_meta_path.read_text(encoding="utf-8"))
                # Extraer campos de trazabilidad P2.6
                for _fk in ("n_features", "n_text_features", "has_text_features",
                            "text_embed_prefix", "text_feat_cols", "text_prob_cols",
                            "feat_cols_", "feat_cols"):
                    if _fk in _model_meta and _model_meta[_fk] is not None:
                        _feat_trace[_fk] = _model_meta[_fk]
        except Exception:
            pass  # best-effort: no rompe el bundle

        manifest = build_predictor_manifest(
            run_id=str(metrics.get("run_id") or ""),
            dataset_id=dataset_id,
            model_name=model_name,
            task_type=str(ctx.get("task_type") or "classification"),
            input_level=str(ctx.get("input_level") or "row"),
            target_col=str(ctx.get("target_col")) if ctx.get("target_col") else None,
            extra={
                k: v
                for k, v in {
                    "family": family,
                    "data_source": ctx.get("data_source"),
                    "data_plan": ctx.get("data_plan"),
                    "split_mode": ctx.get("split_mode"),
                    "val_ratio": ctx.get("val_ratio"),
                    "target_mode": ctx.get("target_mode"),
                    # P2.6: conteos de features de texto (si existen)
                    "n_features_total": _feat_trace.get("n_features"),
                    "n_text_features": _feat_trace.get("n_text_features"),
                    "text_features_present": _feat_trace.get("has_text_features"),
                    "text_embed_prefix": _feat_trace.get("text_embed_prefix"),
                    "note": "P2.3+: si model.bin != placeholder, el modelo se considera listo para inferencia.",
                }.items()
                if v is not None
            },
        )
        write_json(bp.predictor_json, manifest)

        # ------------------------------------------------------------
        # 3) preprocess.json placeholder (si no existe)
        # ------------------------------------------------------------
        if not bp.preprocess_json.exists():
            write_json(bp.preprocess_json, {"schema_version": 1, "notes": "placeholder P2.x"})

        # ------------------------------------------------------------
        # 4) Fallback: si NO hubo persistencia real, dejar placeholder
        # ------------------------------------------------------------
        if not bp.model_bin.exists():
            bp.model_bin.write_bytes(b"PLACEHOLDER_MODEL_BIN_P2_1")

    except Exception:
        logger.exception("No se pudo escribir predictor bundle (best-effort)")


def _call_with_accepted_kwargs(fn, **kwargs):
    """
    Llama `fn(**kwargs)` pero filtrando claves que `fn` no acepta.
    Esto evita errores tipo: got an unexpected keyword argument 'family'.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_varkw:
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return fn(**filtered)
    except Exception:
        # fallback: si no se puede inspeccionar, intenta con kwargs tal cual
        return fn(**kwargs)


def _read_json_if_exists(ref: str) -> Optional[Dict[str, Any]]:
    p = _abs_path(ref)
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_labeled_score_meta(labeled_ref: str) -> Optional[Dict[str, Any]]:
    """Extrae meta del score_total desde el labeled (si existe).

    P0: prioriza el sidecar ``*.parquet.meta.json`` (generado por BETO), porque
    la calibración/β no necesariamente vive como columnas dentro del parquet.

    Compat:
    - Si no hay sidecar, intenta leer columnas dentro del parquet (legacy).
    """
    p = _abs_path(labeled_ref)
    if not p.exists():
        return None

    # 1) Preferir sidecar: <archivo>.parquet.meta.json
    sidecar = p.with_suffix(p.suffix + ".meta.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            payload = None

        if isinstance(payload, dict):
            # Mantener un subset estable para UI (y evitar ruido de campos extras)
            keys = [
                "score_delta_max",
                "score_calib_q",
                "score_beta",
                "score_beta_source",
                "score_calib_abs_q",
            ]
            meta = {k: payload.get(k) for k in keys if k in payload}
            return meta or payload  # fallback al dict completo si no contiene las keys

    # 2) Fallback legacy: intentar columnas dentro del parquet
    cols_wanted = [
        "score_delta_max",
        "score_calib_q",
        "score_beta",
        "score_beta_source",
        "score_calib_abs_q",
    ]

    cols_existing: list[str] = []
    try:
        import pyarrow.parquet as pq  # type: ignore
        schema_cols = pq.ParquetFile(p).schema.names
        cols_existing = [c for c in cols_wanted if c in schema_cols]
        if not cols_existing:
            return None
        df = pq.read_table(p, columns=cols_existing).to_pandas()
    except Exception:
        # Fallback: lee completo (datasets suelen ser manejables)
        try:
            df = pd.read_parquet(p)
        except Exception:
            return None
        cols_existing = [c for c in cols_wanted if c in df.columns]
        if not cols_existing:
            return None
        df = df[cols_existing]

    meta: Dict[str, Any] = {}
    for c in cols_existing:
        try:
            s = df[c].dropna()
            if len(s) == 0:
                continue
            val = s.iloc[0]
            if isinstance(val, (np.generic,)):
                val = val.item()
            meta[c] = val
        except Exception:
            continue

    return meta or None



# ---------------------------------------------------------------------------
# FIX A: Factory de estrategias (CLASES, no instancias)
# ---------------------------------------------------------------------------

_STRATEGY_CLASSES: Dict[str, Type[Any]] = {
    "rbm_general": RBMGeneral,
    "rbm_restringida": RBMRestringida,
    "dbm_manual": DBMManualPlantillaStrategy,
}

def _expand_grid(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # ya viene como lista de dicts -> lo tratamos como combinaciones explícitas
    out = []
    for g in (grid or []):
        if isinstance(g, dict):
            out.append(g)
    return out


def _default_sweep_grid() -> list[dict[str, Any]]:
    # Grid seguro basado en el config legacy (hidden_units/lr/batch_size/cd_k)
    return [
        {"hidden_units": 64, "lr": 0.01},
        {"hidden_units": 64, "lr": 0.05},
        {"hidden_units": 128, "lr": 0.01},
        {"hidden_units": 128, "lr": 0.05},
    ]


def _sweeps_dir() -> Path:
    return (ARTIFACTS_DIR / "sweeps").resolve()

def _write_sweep_summary(sweep_id: str, payload: dict[str, Any]) -> Path:
    d = _sweeps_dir() / str(sweep_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "summary.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def _recompute_sweep_winners(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """
    Recomputar best_overall y best_by_model leyendo metrics.json de cada run.
    - Si un candidato no tiene métricas comparables (p.ej. sin val_rmse en regresión),
      su score cae al peor valor.
    """
    from ...utils.runs_io import champion_score, load_run_metrics  # noqa

    best_overall: dict[str, Any] | None = None
    best_by_model: dict[str, dict[str, Any]] = {}

    for it in candidates:
        if it.get("status") != "completed" or not it.get("run_id"):
            continue

        metrics = load_run_metrics(str(it["run_id"]))
        it["metrics"] = metrics

        tier, sc = champion_score(metrics or {})
        # Normalizar a float finito (JSON/UI)
        try:
            sc = float(sc)
        except Exception:
            sc = -1e30
        if not math.isfinite(sc):
            sc = -1e30

        it["score"] = [int(tier), float(sc)]

        m = str(it.get("model_name") or "")
        prev = best_by_model.get(m)
        if (prev is None) or (tuple(it["score"]) > tuple(prev.get("score") or (-999, -1e30))):
            best_by_model[m] = dict(it)

        if (best_overall is None) or (tuple(it["score"]) > tuple(best_overall.get("score") or (-999, -1e30))):
            best_overall = dict(it)

    return best_overall, best_by_model


def _create_strategy(
    modelo: str | None = None,
    *,
    model_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Factory de estrategias.
    - Compatibilidad: acepta `modelo` (histórico) y `model_name` (nuevo).
    - Tolera kwargs extra (hparams, job_id, dataset_id, family, etc.) para evitar
      fallos si cambia el caller.
    """
    name = (modelo or model_name or "").lower().strip()
    cls = _STRATEGY_CLASSES.get(name)
    if cls is None:
        raise HTTPException(status_code=400, detail=f"Modelo '{name}' no soportado")

    # Instancia de forma segura: filtra kwargs según la firma del constructor
    return _call_with_accepted_kwargs(cls, **kwargs)



def _safe_reset_strategy(strategy: Any) -> None:
    """
    Intenta resetear estado runtime del strategy (FIX B defensivo).

    Esto NO sustituye el arreglo definitivo dentro del strategy (setup/fit),
    pero reduce la probabilidad de contaminación por hot-reload u otras causas.

    Métodos que intenta:
      - reset()
      - _reset_runtime_state()
      - reset_state()
      - clear_state()
    """
    for m in ("reset", "_reset_runtime_state", "reset_state", "clear_state"):
        fn = getattr(strategy, m, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                # Reset nunca debe tumbar el entrenamiento.
                pass
            break


# ---------------------------------------------------------------------------
# Estado in-memory de jobs (para polling de la UI)
# ---------------------------------------------------------------------------

_ESTADOS: Dict[str, Dict[str, Any]] = {}
_OBS_WIRED_JOBS: set[str] = set()


def _normalize_hparams(hparams: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normaliza claves a minúsculas y retorna dict seguro (no None)."""
    if not hparams:
        return {}
    return {str(k).lower(): v for k, v in hparams.items()}

def _maybe_set(d: Dict[str, Any], key: str, value: Any) -> None:
    """Setea d[key]=value solo si value no es None y key no existe."""
    if value is None:
        return
    if key not in d:
        d[key] = value


def _infer_target_col(req: "EntrenarRequest", resolved_run_hparams: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Inferencia robusta de target_col para que:
    - El training/evaluación sepan qué columna usar
    - El snapshot (metrics.json params.req.target_col) no quede en null
    """
    # 1) si viene explícito, respétalo
    explicit = getattr(req, "target_col", None)
    if explicit:
        return explicit

    rh = resolved_run_hparams or {}

    family = str(getattr(req, "family", None) or "sentiment_desempeno").lower()
    task_type = str(getattr(req, "task_type", None) or rh.get("task_type") or "").lower()
    target_mode = str(getattr(req, "target_mode", None) or rh.get("target_mode") or "").lower()
    data_source = str(getattr(req, "data_source", None) or rh.get("data_source") or "").lower()

    # 2) reglas por family (prioritario)
    if family == "sentiment_desempeno":
        # tu feature-pack ya construye y_sentimiento (y también p_neg/p_neu/p_pos)
        # y_sentimiento es el target "clase" para evaluación
        return "y_sentimiento"

    if family == "score_docente":
        # pair_matrix usa target_score (como ya viste en metrics.json de regression)
        return "target_score"

    # 3) fallback por task_type (si family llega vacío)
    if task_type == "classification":
        return "y_sentimiento"
    if task_type == "regression":
        return "target_score"

    return None

def _prune_hparams_for_ui(hparams_norm: Dict[str, Any]) -> Dict[str, Any]:
    """Elimina claves 'reservadas' de hparams para mostrarlas en UI sin pisar campos del request.

    Problema real detectado:
      - Si el usuario manda ``hparams.epochs`` (p. ej. 10) pero el request usa ``epochs`` (p. ej. 5),
        al construir ``params`` se terminaba mostrando 10 en ``/modelos/estado``.
      - Además, el evento ``training.started`` puede traer de vuelta esos hparams y volver a pisar
        ``params`` si hacemos un reemplazo completo.

    Esta función elimina claves que deben venir del request (no de hparams) en el bloque que se expone
    en el estado para la UI.
    """
    hp = dict(hparams_norm or {})
    for k in [
        # Request-level / control de entrenamiento
        "epochs",
        "val_ratio",
        "split_mode",
        "target_mode",
        "data_source",
        "include_teacher_materia",
        "teacher_materia_mode",
        "job_id",
        # Metodología / selección
        "metodologia",
        "ventana_n",
        "dataset_id",
        "periodo_actual",
        "auto_prepare",
        "data_ref",
    ]:
        hp.pop(k, None)
    return hp


def _flatten_metrics_from_payload(payload: Dict[str, Any], allow_loss: bool = True) -> Dict[str, float]:
    """
    Aplana métricas numéricas desde un payload de evento.
    """
    if not payload:
        return {}
    ctrl = {"correlation_id", "epoch", "loss", "event", "model", "params", "final_metrics", "metrics"}
    out: Dict[str, float] = {}
    for k, v in payload.items():
        if k in ctrl:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    if allow_loss and "loss" in payload and isinstance(payload["loss"], (int, float)):
        out.setdefault("loss", float(payload["loss"]))
    return out


def _wire_job_observers(job_id: str) -> None:
    """
    Suscribe handlers al BUS para capturar eventos ``training.*`` de un job.
    """
    if job_id in _OBS_WIRED_JOBS:
        return

    def _match(evt) -> bool:
        try:
            return evt.payload.get("correlation_id") == job_id
        except Exception:
            return False

    def _on_started(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])
        st.setdefault("progress", 0.0)
        st["status"] = "running"
        st["model"] = evt.payload.get("model", st.get("model"))

        # IMPORTANTE:
        # No reemplazar st["params"] por completo, porque el evento `training.started`
        # suele reflejar hparams (y puede incluir `epochs`), lo cual podría pisar el
        # valor correcto `req.epochs` guardado previamente por el router.
        params_evt = evt.payload.get("params")
        if isinstance(params_evt, dict):
            incoming = _normalize_hparams(params_evt)
            existing = st.get("params", {}) or {}
            keep_epochs = existing.get("epochs")
            # Merge de incoming -> existing
            for k, v in incoming.items():
                if k == "epochs":
                    continue
                existing[k] = v
            # Restaurar epochs correcto si existía
            if keep_epochs is not None:
                existing["epochs"] = keep_epochs
            st["params"] = existing

    def _on_epoch_end(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])
        st.setdefault("progress", 0.0)

        payload = evt.payload or {}
        epoch = payload.get("epoch")
        loss = payload.get("loss")

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = _flatten_metrics_from_payload(payload, allow_loss=True)

        point: Dict[str, Any] = {"epoch": epoch}
        if isinstance(loss, (int, float)):
            point["loss"] = float(loss)

        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)) and k not in ("epoch",):
                point[k] = float(v)

        st["history"].append(point)
        st["metrics"] = {k: v for k, v in point.items() if k not in ("epoch",)}

        # Item 2: progress = epoch / epochs_total (si se puede calcular)
        try:
            epochs_total = st.get("params", {}).get("epochs") or 1
            e = float(epoch) if isinstance(epoch, (int, float)) else None
            et = float(epochs_total)
            if e is not None and et > 0:
                st["progress"] = min(1.0, max(0.0, e / et))
        except Exception:
            # Nunca romper el job-state por progress
            pass


    def _on_completed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        payload = evt.payload or {}
        final_metrics = payload.get("final_metrics")
        if not isinstance(final_metrics, dict):
            final_metrics = _flatten_metrics_from_payload(payload, allow_loss=True) or st.get("metrics", {})
        st["metrics"] = final_metrics
        st["status"] = "completed"
        st["progress"] = 1.0

    def _on_persisted(evt) -> None:
        """Actualiza el estado del job cuando el run ya fue persistido.

        Notas
        -----
        - El evento ``training.persisted`` se emite desde el router justo después
          de ``save_run``.
        - No marca el job como *completed* (eso lo hace ``training.completed``)
          sino que asegura que ``run_id``/``artifact_path`` sean navegables.
        """

        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        payload = evt.payload or {}
        if payload.get("run_id"):
            st["run_id"] = str(payload.get("run_id"))
        if payload.get("artifact_path"):
            st["artifact_path"] = str(payload.get("artifact_path"))
        # Flag interno (no rompe esquema) útil para UI/debug.
        st["artifact_ready"] = bool(payload.get("artifact_ready", True))
        _ESTADOS[job_id] = st

    def _on_failed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st["status"] = "failed"
        st["error"] = evt.payload.get("error", "unknown error")
        st.setdefault("progress", 0.0)

    BUS.subscribe("training.started", _on_started)
    BUS.subscribe("training.epoch_end", _on_epoch_end)
    BUS.subscribe("training.completed", _on_completed)
    BUS.subscribe("training.persisted", _on_persisted)
    BUS.subscribe("training.failed", _on_failed)

    _OBS_WIRED_JOBS.add(job_id)


# ---------------------------------------------------------------------------
# Readiness + resolver data_source + auto_prepare
# ---------------------------------------------------------------------------

def _dataset_id(req: EntrenarRequest) -> Optional[str]:
    """Obtiene dataset_id/periodo desde el request."""
    return getattr(req, "dataset_id", None) or getattr(req, "periodo_actual", None)


def _resolve_by_data_source(req: EntrenarRequest) -> str:
    """
    Resuelve el input principal del entrenamiento según `data_source`.

    Extensión Ruta 2:
    - Si `family=score_docente`, el default lógico es consumir `pair_matrix.parquet`
      (1 fila = 1 par docente–materia).
    """
    data_ref = getattr(req, "data_ref", None)
    if data_ref:
        return _strip_localfs(str(data_ref))

    ds = _dataset_id(req)
    if not ds:
        raise HTTPException(status_code=400, detail="Falta dataset_id/periodo_actual para resolver data_source.")

    family = str(getattr(req, "family", "sentiment_desempeno") or "sentiment_desempeno").lower()
    data_source = str(getattr(req, "data_source", "feature_pack")).lower()

    if data_source in ("pair_matrix", "pairs", "pair"):
        return f"artifacts/features/{ds}/pair_matrix.parquet"

    if data_source == "feature_pack":
        # Para score_docente, el "pack" relevante es el pair_matrix (Ruta 2)
        if family == "score_docente":
            return f"artifacts/features/{ds}/pair_matrix.parquet"
        return f"artifacts/features/{ds}/train_matrix.parquet"

    if data_source == "unified_labeled":
        preferred = BASE_DIR / "historico" / "unificado_labeled.parquet"
        if preferred.exists():
            return "historico/unificado_labeled.parquet"
        legacy = BASE_DIR / "historico" / "unificado.parquet"
        if legacy.exists():
            return "historico/unificado.parquet"
        return "historico/unificado_labeled.parquet"

    try:
        p = resolve_labeled_path(str(ds))
        return _relpath(p)
    except Exception:
        return f"data/labeled/{ds}_beto.parquet"



def _ensure_unified_labeled() -> None:
    """Asegura `historico/unificado_labeled.parquet`."""
    out = BASE_DIR / "historico" / "unificado_labeled.parquet"
    if out.exists():
        return
    try:
        from neurocampus.data.strategies.unificacion import UnificacionStrategy
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo importar UnificacionStrategy para auto_prepare. "
                "Ejecuta manualmente el job: POST /jobs/data/unify/run (mode=acumulado_labeled)."
            ),
        ) from e

    strat = UnificacionStrategy(base_uri=f"localfs://{BASE_DIR.as_posix()}")
    strat.acumulado_labeled()


def _ensure_feature_pack(
    dataset_id: str,
    input_uri: str,
    *,
    force: bool = False,
    text_feats_mode: str = 'none',
    text_col: Optional[str] = None,
    text_n_components: int = 64,
    text_min_df: int = 2,
    text_max_features: int = 20000,
    text_random_state: int = 42,
) -> Dict[str, str]:
    """Asegura `artifacts/features/<dataset_id>/train_matrix.parquet`.

    El **feature-pack** es un conjunto de artefactos derivados del dataset que permite
    entrenar modelos (en especial la RBM restringida) leyendo una *matriz de entrenamiento*
    ya materializada en disco (``train_matrix.parquet``) más índices auxiliares.

    Esta función es *idempotente*:

    - Si el archivo ya existe y ``force=False`` (default), no recalcula.
    - Si ``force=True``, vuelve a construir el feature-pack.

    :param dataset_id: Identificador del dataset (ej. ``"2025-1"``).
    :param input_uri: Ruta/URI (relativa o absoluta) del dataset fuente (parquet/csv).
    :param force: Recalcular incluso si ya existe.
    :returns: Diccionario con rutas *relativas* a los artefactos generados.
    :raises HTTPException: Si no se puede importar el builder o si falla el build.
    """
    out_dir = _abs_path(f"artifacts/features/{dataset_id}")
    out = out_dir / "train_matrix.parquet"

    # Rutas esperadas (las devolvemos siempre, existan o no, para UI/debug).
    pair_path = out_dir / "pair_matrix.parquet"
    pair_meta_path = out_dir / "pair_meta.json"

    artifacts_rel: Dict[str, str] = {
        "train_matrix": _relpath(out),
        "teacher_index": _relpath(out_dir / "teacher_index.json"),
        "materia_index": _relpath(out_dir / "materia_index.json"),
        "bins": _relpath(out_dir / "bins.json"),
        "meta": _relpath(out_dir / "meta.json"),
        # Ruta 2
        "pair_matrix": _relpath(pair_path),
        "pair_meta": _relpath(pair_meta_path),
    }


    if out.exists() and pair_path.exists() and pair_meta_path.exists() and not force:
        return artifacts_rel


    try:
        from neurocampus.data.features_prepare import prepare_feature_pack
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo importar prepare_feature_pack para auto_prepare. "
                "Ejecuta manualmente el job: POST /jobs/data/features/prepare/run "
                "o llama a POST /modelos/feature-pack/prepare."
            ),
        ) from e

    out_dir_abs = str(out_dir.resolve())

    # El builder requiere base_dir explícito para poder resolver rutas relativas.
    try:
        prepare_feature_pack(
            base_dir=BASE_DIR,
            dataset_id=dataset_id,
            input_uri=input_uri,
            output_dir=out_dir_abs,
            text_feats_mode=text_feats_mode,
            text_col=text_col,
            text_n_components=int(text_n_components),
            text_min_df=int(text_min_df),
            text_max_features=int(text_max_features),
            text_random_state=int(text_random_state),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo feature-pack: {e}") from e

    if not out.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "prepare_feature_pack no creó train_matrix.parquet. "
                "Revisa logs y valida que input_uri apunte a un parquet válido."
            ),
        )

    # Ruta 2: pair artifacts deben existir también (compat con runs score_docente)
    if not pair_path.exists() or not pair_meta_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "prepare_feature_pack no creó pair_matrix.parquet/pair_meta.json. "
                "Asegúrate de tener implementado el builder de pair_matrix en features_prepare.py."
            ),
        )


    return artifacts_rel

def _req_get(req, name: str, default=None):
    v = getattr(req, name, None)
    if v is not None:
        return v
    h = getattr(req, "hparams", None) or {}
    return h.get(name, default)

def _read_json_safe(p: Path):
    try:
        if not p.exists():
            return None
        import json
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _feature_pack_meta_path(ds: str) -> Path:
    return _abs_path(f"artifacts/features/{ds}/meta.json")

def _feature_pack_has_sentiment(ds: str) -> bool:
    meta = _read_json_safe(_feature_pack_meta_path(ds)) or {}
    sent_cols = meta.get("sentiment_cols") or []
    cols = meta.get("columns") or []
    return (len(sent_cols) >= 3) and ("y_sentimiento" in cols)

def _should_rebuild_feature_pack(dataset_id: str, *, family: str, data_source: str) -> bool:
    """Decide si hay que reconstruir artifacts/features/<ds>/... por incompatibilidad.

    Regla robusta:
    - sentiment_desempeno requiere blocks.sentiment=True en meta.json (si existe labeled).
    - score_docente requiere que pair_meta.target_col NO sea score_base_0_50 si tenemos labeled disponible.
    """
    try:
        ds = str(dataset_id)
        fam = (family or "").lower()
        src = (data_source or "").lower()

        feat_dir = _abs_path(f"artifacts/features/{ds}")
        meta_path = feat_dir / "meta.json"
        pair_meta_path = feat_dir / "pair_meta.json"

        # 1) Sentiment: si el pack no tiene sentiment block, no hay labels -> rebuild
        if fam == "sentiment_desempeno" and src in ("feature_pack",):
            if meta_path.exists():
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                blocks = meta.get("blocks") or {}
                if not bool(blocks.get("sentiment", False)):
                    return True
            return False

        # 2) Score docente: si pair_meta usa score_base pero existe labeled, rebuild para intentar score_total
        if fam == "score_docente" and src in ("pair_matrix", "pairs", "pair"):
            labeled_path = None
            try:
                labeled_path = resolve_labeled_path(ds)
            except Exception:
                labeled_path = None
            if labeled_path is not None and labeled_path.exists() and pair_meta_path.exists():
                pm = json.load(open(pair_meta_path, "r", encoding="utf-8"))
                if (pm.get("target_col") or "").lower() == "score_base_0_50":
                    return True
            return False

        return False
    except Exception:
        # Si algo falla leyendo meta, NO forzamos rebuild por defecto.
        return False


def _auto_prepare_if_needed(req: EntrenarRequest, data_ref: str) -> None:
    """Ejecuta preparación automática si `auto_prepare=True` y el artefacto requerido no existe
    (o existe pero es incompatible/incompleto para la family solicitada).
    """
    auto_prepare = bool(getattr(req, "auto_prepare", False))
    if not auto_prepare:
        return

    ds = _dataset_id(req)
    data_source = str(getattr(req, "data_source", "feature_pack")).lower()
    metodologia = str(getattr(req, "metodologia", "periodo_actual")).lower()

    # family puede venir top-level o dentro de hparams (retro-compat)
    hparams = getattr(req, "hparams", None) or {}
    family = str(getattr(req, "family", None) or hparams.get("family") or "").lower()

    # Si ya existe, solo reconstruimos si detectamos “pack incompleto/incompatible”
    p = _abs_path(data_ref)
    if p.exists():
        if data_source in ("feature_pack", "pair_matrix", "pairs", "pair") and _should_rebuild_feature_pack(str(ds), family=family, data_source=data_source):
            pass  # seguimos para reconstruir con force=True
        else:
            return

    if data_source == "unified_labeled":
        _ensure_unified_labeled()
        return

    if data_source in ("feature_pack", "pair_matrix", "pairs", "pair"):
        if not ds:
            raise HTTPException(status_code=400, detail="auto_prepare requiere dataset_id/periodo_actual.")

        # -------------------------------------------------------------------
        # P2.6: plumb de parámetros de texto hacia el builder del feature-pack
        # -------------------------------------------------------------------
        # IMPORTANTE:
        # - Por defecto text_feats_mode='none' => no cambia el comportamiento existente.
        # - Si el usuario activa 'tfidf_lsa', se generarán columnas numéricas feat_t_*.
        # - Estos parámetros se pasan SOLO a la construcción del feature-pack (no al training).
        text_feats_mode = str(getattr(req, "text_feats_mode", "none") or "none").lower()
        auto_text_feats = bool(getattr(req, "auto_text_feats", True))

        # Auto-enable (P2.6): para sentiment_desempeno, es común que existan columnas de
        # texto libre (p.ej. `comentario`) que aportan señal adicional.
        #
        # - Para no romper compatibilidad, este comportamiento se puede desactivar
        #   enviando auto_text_feats=false en el request.
        # - Si el parquet ya trae feat_t_* (BETO/previo), el builder NO regenerará.
        if family == "sentiment_desempeno" and auto_text_feats and text_feats_mode == "none":
            text_feats_mode = "tfidf_lsa"

        text_col = getattr(req, "text_col", None)
        text_n_components = int(getattr(req, "text_n_components", 64) or 64)
        text_min_df = int(getattr(req, "text_min_df", 2) or 2)
        text_max_features = int(getattr(req, "text_max_features", 20000) or 20000)
        text_random_state = int(getattr(req, "text_random_state", 42) or 42)

        # Caso histórico (acumulado / ventana)
        if metodologia in ("acumulado", "ventana"):
            _ensure_unified_labeled()
            input_uri = "historico/unificado_labeled.parquet"
            force_fp = False
        else:
            # Caso normal: preferimos LABELED si existe (trae BETO y score_total_0_50 cuando aplica)
            force_fp = False

            labeled_path = None
            try:
                labeled_path = resolve_labeled_path(str(ds))
            except Exception:
                labeled_path = None

            processed = BASE_DIR / "data" / "processed" / f"{ds}.parquet"
            raw = BASE_DIR / "datasets" / f"{ds}.parquet"

            if labeled_path is not None and labeled_path.exists():
                input_uri = _relpath(labeled_path)
            elif processed.exists():
                input_uri = _relpath(processed)
            elif raw.exists():
                input_uri = _relpath(raw)
            else:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"No se encontró un dataset fuente para construir feature-pack de {ds}. "
                        "Opciones:\n"
                        "- Procesa/carga el dataset en la pestaña Data (data/processed/<ds>.parquet)\n"
                        "- O genera labeled BETO (data/labeled/<ds>_beto.parquet)\n"
                        "- O asegúrate de tener datasets/<ds>.parquet"
                    ),
                )

            # Si el pack actual no sirve para la family, forzamos rebuild.
            if _should_rebuild_feature_pack(str(ds), family=family, data_source=data_source):
                force_fp = True

        _ensure_feature_pack(
            str(ds),
            input_uri=input_uri,
            force=force_fp,
            text_feats_mode=text_feats_mode,
            text_col=text_col,
            text_n_components=text_n_components,
            text_min_df=text_min_df,
            text_max_features=text_max_features,
            text_random_state=text_random_state,
        )
        return

    raise HTTPException(
        status_code=409,
        detail=(
            "data_source='labeled' no puede auto-prepararse desde Modelos. "
            "Genera primero data/labeled/<dataset>_beto.parquet desde la pestaña Datos."
        ),
    )



def _read_dataframe_any(path_or_uri: str) -> pd.DataFrame:
    """Lee dataset desde ruta local (parquet/csv)."""
    p = _abs_path(path_or_uri)
    try:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        return pd.read_parquet(p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el dataset ({p.name}): {e}") from e

def _period_key(ds: str) -> tuple[int, int]:
    """
    Ordena dataset_id tipo 'YYYY-N' (ej. 2025-1, 2024-3).
    Si no parsea, lo manda al inicio.
    """
    m = re.match(r"^\s*(\d{4})-(\d{1,2})\s*$", str(ds))
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _list_pair_matrix_datasets() -> list[str]:
    base = _abs_path("artifacts/features")
    if not base.exists():
        return []
    out: list[str] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if (p / "pair_matrix.parquet").exists():
            out.append(p.name)
    return sorted(out, key=_period_key)


def _materialize_score_docente_pair_selection(req: EntrenarRequest, job_id: str) -> str:
    """
    Construye un parquet temporal uniendo pair_matrix de varios periodos según data_plan:
      - recent_window: concat de últimos window_k periodos (<= dataset_id actual)
      - recent_window_plus_replay: concat de ventana + muestra de periodos antiguos (replay_size)
    """
    dataset_id = _dataset_id(req)
    plan = str(getattr(req, "data_plan", "dataset_only") or "dataset_only").lower()
    window_k = int(getattr(req, "window_k", None) or 4)
    replay_size = int(getattr(req, "replay_size", None) or 0)
    replay_strategy = str(getattr(req, "replay_strategy", "uniform") or "uniform").lower()

    all_ds = _list_pair_matrix_datasets()
    if not all_ds:
        raise HTTPException(status_code=404, detail="No hay pair_matrix disponibles en artifacts/features/*/pair_matrix.parquet")

    cur_k = _period_key(dataset_id)
    eligible = [d for d in all_ds if _period_key(d) <= cur_k]
    if not eligible:
        eligible = all_ds[:]  # fallback

    recent = eligible[-window_k:] if window_k > 0 else eligible[-1:]
    if dataset_id not in recent and dataset_id in eligible:
        recent = (recent + [dataset_id])[-window_k:]

    older = [d for d in eligible if d not in recent]

    def _read_pair(ds: str) -> pd.DataFrame:
        p = _abs_path(f"artifacts/features/{ds}/pair_matrix.parquet")
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"pair_matrix no encontrado para {ds}: {p}")
        df = pd.read_parquet(p)
        if "periodo" not in df.columns:
            df = df.copy()
            df["periodo"] = ds
        return df

    df_recent = pd.concat([_read_pair(d) for d in recent], ignore_index=True)

    df_replay = None
    if plan == "recent_window_plus_replay" and replay_size > 0 and older:
        df_pool = pd.concat([_read_pair(d) for d in older], ignore_index=True)
        if len(df_pool) > 0:
            n = min(replay_size, len(df_pool))
            if replay_strategy == "by_period" and "periodo" in df_pool.columns:
                chunks = []
                periods = sorted(df_pool["periodo"].astype(str).unique().tolist(), key=_period_key)
                per = max(1, n // max(1, len(periods)))
                for per_ds in periods:
                    sub = df_pool[df_pool["periodo"].astype(str) == per_ds]
                    if len(sub) == 0:
                        continue
                    take = min(per, len(sub))
                    chunks.append(sub.sample(n=take, random_state=7, replace=False))
                df_replay = pd.concat(chunks, ignore_index=True) if chunks else df_pool.sample(n=n, random_state=7, replace=False)
            else:
                df_replay = df_pool.sample(n=n, random_state=7, replace=False)

    df_sel = df_recent if df_replay is None else pd.concat([df_recent, df_replay], ignore_index=True)

    # Alinear columnas (union) para evitar errores si algún periodo trae columnas extra
    cols = sorted(set(df_sel.columns.tolist()))
    df_sel = df_sel.reindex(columns=cols)

    tmp_dir = (BASE_DIR / "data" / ".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref = tmp_dir / f"pair_sel_{job_id}.parquet"
    df_sel.to_parquet(tmp_ref, index=False)
    return str(tmp_ref.resolve())



def _prepare_selected_data(req: EntrenarRequest, job_id: str) -> str:
    """
    Resuelve fuente de datos + auto_prepare + (si aplica) metodología.
    """
    data_ref = _resolve_by_data_source(req)
    _auto_prepare_if_needed(req, data_ref)

    data_source = str(getattr(req, "data_source", "feature_pack")).lower()

    if data_source in ("feature_pack", "pair_matrix", "pairs", "pair"):
        pack_path = _abs_path(data_ref)
        if not pack_path.exists():
            kind = "pair_matrix" if data_source in ("pair_matrix", "pairs", "pair") else "feature_pack"
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Artefacto de features no encontrado ({kind}): {pack_path}. "
                    "Activa auto_prepare=true al entrenar o llama a POST /modelos/feature-pack/prepare."
                ),
            )

        # score_docente: materializar selección multi-periodo si aplica
        fam = str(getattr(req, "family", "") or "").lower()
        plan = str(getattr(req, "data_plan", "dataset_only") or "dataset_only").lower()
        if fam == "score_docente" and data_source in ("pair_matrix", "pairs", "pair") and plan in ("recent_window", "recent_window_plus_replay"):
            return _materialize_score_docente_pair_selection(req, job_id)

        return str(pack_path.resolve())



    df = _read_dataframe_any(data_ref)

    metodologia = getattr(req, "metodologia", None) or "periodo_actual"
    periodo_actual = getattr(req, "periodo_actual", None) or _dataset_id(req)
    ventana_n = getattr(req, "ventana_n", None) or 4

    metodo = resolver_metodologia(str(metodologia).lower())
    cfg = SeleccionConfig(periodo_actual=str(periodo_actual) if periodo_actual else None, ventana_n=int(ventana_n))

    df_sel = metodo.seleccionar(df, cfg)
    if df_sel.empty:
        raise HTTPException(status_code=400, detail="Selección de datos vacía según metodología/periodo.")

    tmp_dir = (BASE_DIR / "data" / ".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref = tmp_dir / f"df_sel_{job_id}.parquet"

    try:
        df_sel.to_parquet(tmp_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo materializar el subconjunto: {e}") from e

    return str(tmp_ref.resolve())


# ---------------------------------------------------------------------------
# FIX: Construcción robusta de hparams para el training (NO None + defaults)
# ---------------------------------------------------------------------------

def _build_run_hparams(req: EntrenarRequest, job_id: str) -> Dict[str, Any]:
    """
    Construye hparams para el training garantizando:

    - NO incluir valores None (para evitar que el strategy reciba None y lo degrade a 'none').
    - Defaults consistentes con el flujo nuevo.
    - Los campos explícitos del request tienen prioridad sobre req.hparams.

    Extensión Ruta 2 (score_docente):
    - default data_source = pair_matrix
    - default target_mode = score_total_0_50 (solo informativo; el target real lo dicta pair_meta)
    - se pasan flags family/task/input_level/target_col e incremental config (window/replay/warm-start)
    """
    hp = _normalize_hparams(getattr(req, "hparams", None))

    # Evitar que hparams contenga claves reservadas del request (p.ej. epochs)
    hp.pop("epochs", None)

    def put(key: str, value: Any) -> None:
        if value is None:
            return
        hp[key] = value

    put("job_id", job_id)

    family = str(getattr(req, "family", "sentiment_desempeno") or "sentiment_desempeno").lower()
    put("family", family)
    put("task_type", getattr(req, "task_type", None))
    put("input_level", getattr(req, "input_level", None))
    put("target_col", getattr(req, "target_col", None))

    # Incremental config (si existe en el schema)
    put("data_plan", getattr(req, "data_plan", None))
    put("window_k", getattr(req, "window_k", None))
    put("replay_size", getattr(req, "replay_size", None))
    put("replay_strategy", getattr(req, "replay_strategy", None))
    put("recency_lambda", getattr(req, "recency_lambda", None))
    put("warm_start_from", getattr(req, "warm_start_from", None))
    put("warm_start_run_id", getattr(req, "warm_start_run_id", None))

    # Compatibilidad (P2.2/P2.3): algunos clientes envían warm start como objeto:
    #
    #   "warm_start": {"mode": "champion"}
    #   "warm_start": {"mode": "run_id", "run_id": "<RUN_ID>"}
    #
    # En esos casos `warm_start_from` puede venir vacío/None. Normalizamos aquí para que
    # el job trace y los artifacts reflejen correctamente la intención del request.
    try:
        _ws_mode = getattr(req, "warm_start_from", None)
        _ws_run_id = getattr(req, "warm_start_run_id", None)
        _ws_obj = getattr(req, "warm_start", None)
        if (_ws_mode is None) or (str(_ws_mode).strip().lower() in {"", "none", "null"}):
            if _ws_obj:
                if isinstance(_ws_obj, dict):
                    _ws_mode = _ws_obj.get("mode") or _ws_obj.get("warm_start_from")
                    _ws_run_id = _ws_run_id or _ws_obj.get("run_id") or _ws_obj.get("warm_start_run_id")
                else:
                    _ws_mode = getattr(_ws_obj, "mode", None) or getattr(_ws_obj, "warm_start_from", None)
                    _ws_run_id = _ws_run_id or getattr(_ws_obj, "run_id", None) or getattr(_ws_obj, "warm_start_run_id", None)
        if _ws_mode is not None:
            put("warm_start_from", str(_ws_mode).lower())
        if _ws_run_id is not None:
            put("warm_start_run_id", _ws_run_id)
    except Exception:
        # No bloquear entrenamiento si el objeto warm_start viene en formato inesperado.
        pass


    # Defaults defensivos (si el request viene con None)
    data_source = getattr(req, "data_source", None)
    if data_source is None:
        data_source = "pair_matrix" if family == "score_docente" else "feature_pack"

    target_mode = getattr(req, "target_mode", None)

    # Evitar confusión UI:
    # - score_docente debe quedar con target_mode=score_only (aunque el schema default sea sentiment_probs)
    if family == "score_docente":
        if (target_mode is None) or (str(target_mode).lower() in ("sentiment_probs", "sentiment_label")):
            target_mode = "score_only"
    else:
        # sentiment_desempeno: si alguien envía score_only por error, normalizamos al default
        if (target_mode is None) or (str(target_mode).lower() == "score_only"):
            target_mode = "sentiment_probs"


    split_mode = getattr(req, "split_mode", None) or "temporal"
    val_ratio = getattr(req, "val_ratio", None)
    if val_ratio is None:
        val_ratio = 0.2

    include_tm = getattr(req, "include_teacher_materia", None)
    if include_tm is None:
        include_tm = True

    tm_mode = getattr(req, "teacher_materia_mode", None)
    # Si se desea teacher/materia y el modo no viene, default a 'embed'
    if (tm_mode is None) and bool(include_tm):
        tm_mode = "embed"

    put("data_source", str(data_source).lower())
    put("target_mode", target_mode)
    put("split_mode", split_mode)
    put("val_ratio", float(val_ratio))
    put("include_teacher_materia", bool(include_tm))
    # Importante: SOLO poner teacher_materia_mode si no es None
    put("teacher_materia_mode", tm_mode)

    # P2.6 (trazabilidad): registrar intención de features de texto del feature-pack.
    # Nota: esto NO obliga al entrenamiento a usar texto; solo deja auditable cómo se
    # pidió construir el feature-pack cuando auto_prepare está activo.
    put("text_feats_mode", getattr(req, "text_feats_mode", None))
    put("text_col", getattr(req, "text_col", None))
    put("text_n_components", getattr(req, "text_n_components", None))
    put("text_min_df", getattr(req, "text_min_df", None))
    put("text_max_features", getattr(req, "text_max_features", None))
    put("text_random_state", getattr(req, "text_random_state", None))

    return hp


def _evaluate_model_metrics(
    model: Any,
    data_ref: str,
    *,
    split_mode: str,
    val_ratio: float,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación para guardar en metrics.json del run
    y para que /modelos/runs y /modelos/champion muestren métricas reales.

    - Usa el MISMO filtrado que el modelo hace en _prepare_xy(...)
    - Aplica split_mode (temporal/random/stratified) y val_ratio
    - Devuelve:
        accuracy, f1_macro, val_accuracy, val_f1_macro,
        confusion_matrix (VAL),
        labels,
        train{n,acc,f1_macro,confusion_matrix},
        val{n,acc,f1_macro,confusion_matrix},
        n_train, n_val
    """
    try:
        # Normaliza path (puede venir con file:// o rutas relativas)
        p = _abs_path(_strip_localfs(str(data_ref)))
        df = pd.read_parquet(p)

        # Construye kwargs compatibles con distintas firmas de _prepare_xy
        base_kwargs = {
            "accept_teacher": bool(hparams.get("accept_teacher", True)),
            "threshold": float(hparams.get("accept_threshold", 0.8)),
            "max_calif": int(hparams.get("max_calif", 10)),
            "include_text_probs": bool(hparams.get("use_text_probs", False)),
            "include_text_embeds": bool(hparams.get("use_text_embeds", False)),
            "text_embed_prefix": str(hparams.get("text_embed_prefix", "x_text_")),
        }

        sig = inspect.signature(model._prepare_xy)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

        prep_out = model._prepare_xy(df, **kwargs)  # type: ignore[attr-defined]

        # Soporta versiones viejas que retornaban más cosas (mask/meta)
        if isinstance(prep_out, tuple):
            if len(prep_out) == 3:
                X, y, _feat_cols = prep_out
            elif len(prep_out) >= 3:
                X, y = prep_out[0], prep_out[1]
            else:
                return {}
        else:
            return {}

        labels = list(getattr(model, "classes_", ["neg", "neu", "pos"]))
        y_idx = np.asarray(y, dtype=int)
        y_true = np.asarray([labels[int(i)] for i in y_idx], dtype=object)

        n = int(len(y_true))
        if n == 0:
            return {}

        # Split
        val_ratio = float(val_ratio)
        val_ratio = min(max(val_ratio, 0.0), 0.9)
        n_val = int(round(n * val_ratio))
        n_val = max(1, n_val) if n >= 2 else 0

        if n_val == 0:
            return {}

        idx = np.arange(n)

        if (split_mode or "").lower() == "temporal":
            idx_tr = idx[: n - n_val]
            idx_va = idx[n - n_val :]
        else:
            seed = int(hparams.get("seed", 42))
            strat = y_true if (split_mode or "").lower() == "stratified" else None
            try:
                idx_tr, idx_va = train_test_split(
                    idx, test_size=val_ratio, random_state=seed, shuffle=True, stratify=strat
                )
            except Exception:
                idx_tr, idx_va = train_test_split(
                    idx, test_size=val_ratio, random_state=seed, shuffle=True, stratify=None
                )

        X_tr, y_tr = X[idx_tr], y_true[idx_tr]
        X_va, y_va = X[idx_va], y_true[idx_va]

        y_pred_tr = np.asarray(model.predict(X_tr), dtype=object)  # type: ignore[attr-defined]
        y_pred_va = np.asarray(model.predict(X_va), dtype=object)  # type: ignore[attr-defined]

        def pack(y_t, y_p):
            if len(y_t) == 0:
                return {"n": 0, "acc": None, "f1_macro": None, "confusion_matrix": None}
            return {
                "n": int(len(y_t)),
                "acc": float(accuracy_score(y_t, y_p)),
                "f1_macro": float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_t, y_p, labels=labels).tolist(),
            }

        tr_pack = pack(y_tr, y_pred_tr)
        va_pack = pack(y_va, y_pred_va)

        return {
            "labels": labels,
            "n_train": int(tr_pack["n"]),
            "n_val": int(va_pack["n"]),
            "accuracy": tr_pack["acc"],
            "f1_macro": tr_pack["f1_macro"],
            "val_accuracy": va_pack["acc"],
            "val_f1_macro": va_pack["f1_macro"],
            "train": tr_pack,
            "val": va_pack,
            # Por conveniencia, deja también la CM final como la de validación
            "confusion_matrix": va_pack["confusion_matrix"],
        }
    except Exception as e:
        logger.exception("Eval metrics failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Endpoints: readiness
# ---------------------------------------------------------------------------

@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Verifica insumos para entrenar (labeled / unified_labeled / feature_pack)",
)
def readiness(dataset_id: str) -> ReadinessResponse:
    """Verifica existencia de artefactos mínimos para entrenar un dataset_id.

    Extensión Ruta 2:
    - Reporta pair_matrix/pair_meta (1 fila = 1 par docente–materia)
    - Reporta score_col (target) desde pair_meta/meta.json
    - Expone meta de calibración del score_total desde el labeled (si existe)
    """
    try:
        labeled_path = resolve_labeled_path(dataset_id)
        labeled_ref = _relpath(labeled_path)
        labeled_ok = _abs_path(labeled_ref).exists()
    except Exception:
        labeled_ref = f"data/labeled/{dataset_id}_beto.parquet"
        labeled_ok = _abs_path(labeled_ref).exists()

    unified_ref = "historico/unificado_labeled.parquet"
    unified_ok = _abs_path(unified_ref).exists()

    feat_ref = f"artifacts/features/{dataset_id}/train_matrix.parquet"
    feat_meta_ref = f"artifacts/features/{dataset_id}/meta.json"
    feat_ok = _abs_path(feat_ref).exists()

    pair_ref = f"artifacts/features/{dataset_id}/pair_matrix.parquet"
    pair_meta_ref = f"artifacts/features/{dataset_id}/pair_meta.json"
    pair_ok = _abs_path(pair_ref).exists()

    pair_meta = _read_json_if_exists(pair_meta_ref) if _abs_path(pair_meta_ref).exists() else None
    pack_meta = _read_json_if_exists(feat_meta_ref) if _abs_path(feat_meta_ref).exists() else None

    score_col = None
    if isinstance(pair_meta, dict):
        score_col = pair_meta.get("target_col")
    if not score_col and isinstance(pack_meta, dict):
        score_col = pack_meta.get("score_col")

    labeled_score_meta = _extract_labeled_score_meta(labeled_ref) if labeled_ok else None

    return ReadinessResponse(
        dataset_id=dataset_id,
        labeled_exists=bool(labeled_ok),
        unified_labeled_exists=bool(unified_ok),
        feature_pack_exists=bool(feat_ok),
        pair_matrix_exists=bool(pair_ok),
        score_col=score_col,
        pair_meta=pair_meta,
        labeled_score_meta=labeled_score_meta,
        paths={
            "labeled": labeled_ref,
            "unified_labeled": unified_ref,
            "feature_pack": feat_ref,
            "feature_pack_meta": feat_meta_ref,
            "pair_matrix": pair_ref,
            "pair_meta": pair_meta_ref,
        },
    )



def _temporal_split(n: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split temporal: train = primeras filas, val = últimas filas."""
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    vr = float(val_ratio or 0.2)
    n_val = int(round(n * vr))
    # defensivo
    n_val = max(1, min(n - 1, n_val)) if n > 1 else 0

    idx_tr = np.arange(0, n - n_val, dtype=int)
    idx_va = np.arange(n - n_val, n, dtype=int)
    return idx_tr, idx_va


def _evaluate_post_training_metrics(estrategia, df: "pd.DataFrame", hparams: dict) -> dict:
    """
    Calcula métricas REALES (accuracy/f1/confusion) usando:
      - el mismo _prepare_xy del modelo
      - el mismo split temporal (val_ratio)
      - el mismo esquema de labels [neg, neu, pos]
    """
    # defaults alineados con lo que tu modelo ya usa / espera
    accept_teacher = bool(hparams.get("accept_teacher", True))
    threshold = float(hparams.get("accept_threshold", 0.8))
    max_calif = int(hparams.get("max_calif", 10))

    include_text_probs = bool(hparams.get("use_text_probs", False))
    include_text_embeds = bool(hparams.get("use_text_embeds", False))
    text_embed_prefix = str(hparams.get("text_embed_prefix", "x_text_"))

    # Firma REAL de tus strategies RBM (según tu prueba en terminal)
    X, y, feat_cols = estrategia._prepare_xy(
        df,
        accept_teacher=accept_teacher,
        threshold=threshold,
        max_calif=max_calif,
        include_text_probs=include_text_probs,
        include_text_embeds=include_text_embeds,
        text_embed_prefix=text_embed_prefix,
    )

    labels = ["neg", "neu", "pos"]
    y_true = np.array([labels[int(i)] for i in y.tolist()])
    y_pred = np.array(estrategia.predict(X))  # predict(X) devuelve strings

    idx_tr, idx_va = _temporal_split(len(y_true), float(hparams.get("val_ratio", 0.2)))

    y_true_tr, y_pred_tr = y_true[idx_tr], y_pred[idx_tr]
    y_true_va, y_pred_va = y_true[idx_va], y_pred[idx_va]

    acc_tr = float(accuracy_score(y_true_tr, y_pred_tr)) if len(idx_tr) else None
    f1_tr = float(f1_score(y_true_tr, y_pred_tr, labels=labels, average="macro", zero_division=0)) if len(idx_tr) else None

    acc_va = float(accuracy_score(y_true_va, y_pred_va)) if len(idx_va) else None
    f1_va = float(f1_score(y_true_va, y_pred_va, labels=labels, average="macro", zero_division=0)) if len(idx_va) else None

    cm_tr = confusion_matrix(y_true_tr, y_pred_tr, labels=labels).tolist() if len(idx_tr) else None
    cm_va = confusion_matrix(y_true_va, y_pred_va, labels=labels).tolist() if len(idx_va) else None

    return {
        "accuracy": acc_tr,
        "f1_macro": f1_tr,
        "val_accuracy": acc_va,
        "val_f1_macro": f1_va,
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "labels": labels,
        "train": {
            "n": int(len(idx_tr)),
            "acc": acc_tr,
            "f1_macro": f1_tr,
            "confusion_matrix": cm_tr,
        },
        "val": {
            "n": int(len(idx_va)),
            "acc": acc_va,
            "f1_macro": f1_va,
            "confusion_matrix": cm_va,
        },
        # compat: muchos consumers esperan confusion_matrix a nivel raíz = VAL
        "confusion_matrix": cm_va,
    }

def _require_exported_model(run_dir: str | Path, model_name: str) -> None:
    run_dir = Path(run_dir)
    model_dir = run_dir / "model"
    present = {p.name for p in model_dir.iterdir() if p.is_file()} if model_dir.exists() else set()

    mn = (model_name or "").lower().strip()

    if mn.startswith("rbm"):
        if "meta.json" not in present or not ({"rbm.pt", "head.pt"} & present):
            raise RuntimeError(
                f"Run {run_dir.name}: export RBM incompleto en {model_dir}. "
                f"Se esperaba meta.json + rbm.pt/head.pt. Presentes: {sorted(present)}"
            )

    if mn.startswith("dbm"):
        if not {"meta.json", "dbm_state.npz"} <= present:
            raise RuntimeError(
                f"Run {run_dir.name}: export DBM incompleto en {model_dir}. "
                f"Se esperaba meta.json + dbm_state.npz. Presentes: {sorted(present)}"
            )

# ---------------------------------------------------------------------------
# Entrenamiento (persistencia vía runs_io)
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: EntrenarRequest) -> None:
    """
    Ejecuta un entrenamiento (job_type=train) y persiste artifacts de run.

    P0: este método debe ser compatible con la firma REAL de:
      - PlantillaEntrenamiento(estrategia)
      - PlantillaEntrenamiento.run(data_ref=..., epochs=..., hparams=..., model_name=...)
      - runs_io.save_run(...)

    También completa el estado in-memory para que /modelos/estado/{job_id} devuelva
    un EstadoResponse completo.
    """
    t0 = time.perf_counter()

    # estado base (siempre completo)
    st = _ESTADOS.get(job_id, {}) if isinstance(_ESTADOS.get(job_id), dict) else {}
    st.setdefault("job_id", job_id)
    st["job_type"] = "train"
    st["status"] = "running"
    st["progress"] = 0.0
    st["error"] = None
    st.setdefault("model", getattr(req, "modelo", None))
    st.setdefault("params", {})
    st["metrics"] = {}
    st["history"] = []
    st["run_id"] = None
    st["artifact_path"] = None
    st["champion_promoted"] = None
    st["time_total_ms"] = None
    _ESTADOS[job_id] = st

    # Observers (training.* -> estado en memoria)
    _wire_job_observers(job_id)

    try:
        # 1) Resolver hparams/plan (fuente de verdad para ejecución)
        run_hparams = _build_run_hparams(req, job_id)
        # Asegurar job_id para PlantillaEntrenamiento (usa hparams.job_id como override)
        run_hparams["job_id"] = job_id

        # 1b) Resolver warm_start_path (RBM) si el request lo solicita.
        #     Debe ocurrir ANTES de crear la estrategia para que hparams
        #     llegue con warm_start_path ya seteado.
        # Compatibilidad: aceptar warm start también como objeto "warm_start": {"mode": "..."}
        # (p.ej. payload del frontend). Si no viene warm_start_from explícito, lo inferimos.
        _ws_mode = getattr(req, "warm_start_from", None)
        _ws_run_id = getattr(req, "warm_start_run_id", None)
        _ws_obj = getattr(req, "warm_start", None)
        if (_ws_mode is None) or (str(_ws_mode).strip().lower() in {"", "none", "null"}):
            if _ws_obj:
                if isinstance(_ws_obj, dict):
                    _ws_mode = _ws_obj.get("mode") or _ws_obj.get("warm_start_from")
                    _ws_run_id = _ws_run_id or _ws_obj.get("run_id") or _ws_obj.get("warm_start_run_id")
                else:
                    _ws_mode = getattr(_ws_obj, "mode", None) or getattr(_ws_obj, "warm_start_from", None)
                    _ws_run_id = _ws_run_id or getattr(_ws_obj, "run_id", None) or getattr(_ws_obj, "warm_start_run_id", None)
        _ws_mode = str(_ws_mode or "none").lower()
        _ws_path, _ws_trace = resolve_warm_start_path(
            artifacts_dir=ARTIFACTS_DIR,
            dataset_id=str(req.dataset_id or ""),
            family=str(getattr(req, "family", "") or ""),
            model_name=str(getattr(req, "modelo", "") or ""),
            warm_start_from=_ws_mode,
            warm_start_run_id=_ws_run_id,
        )
        if _ws_path is not None:
            run_hparams["warm_start_path"] = str(_ws_path)
        else:
            run_hparams.pop("warm_start_path", None)

        # 2) Normalizar request para snapshot/UI (que "params.req" sea consistente)
        inferred_target_col = _infer_target_col(req, run_hparams)

        update_payload: dict[str, Any] = {
            "data_source": run_hparams.get("data_source"),
            "target_mode": run_hparams.get("target_mode"),
            "split_mode": run_hparams.get("split_mode"),
            "val_ratio": run_hparams.get("val_ratio"),
            "include_teacher_materia": run_hparams.get("include_teacher_materia"),
            "teacher_materia_mode": run_hparams.get("teacher_materia_mode"),
        }
        if inferred_target_col is not None:
            update_payload["target_col"] = inferred_target_col

        # Filtrar solo campos existentes (evita problemas si el schema cambia).
        #
        # Nota (Pydantic v2.11+): acceder a `model_fields` desde la instancia está deprecado.
        # Se debe acceder desde la clase.
        model_fields = getattr(type(req), "model_fields", None)
        if isinstance(model_fields, dict):
            update_payload = {k: v for k, v in update_payload.items() if k in model_fields}

        req_norm = req.model_copy(update=update_payload)

        # -----------------------------------------------------------------
        # (Pre-reserva) run_id + ruta lógica del artifact
        # -----------------------------------------------------------------
        #
        # La PlantillaEntrenamiento emite ``training.completed`` antes de que el router
        # persista el run (save_run). En ese intervalo la UI puede ver status=completed
        # pero run_id/artifact_path en null, lo que dificulta debugging.
        #
        # Para evitarlo, reservamos run_id *una sola vez* antes de correr la plantilla,
        # y exponemos la ruta esperada ``artifacts/runs/<run_id>``.
        #
        # Nota: el directorio real se crea en ``save_run``. Si el job falla antes,
        # la ruta puede no existir; el campo ``artifact_ready`` indica si hubo persistencia.
        run_id = build_run_id(
            dataset_id=str(req_norm.dataset_id),
            model_name=str(req_norm.modelo),
            job_id=str(job_id),
        )
        st["run_id"] = str(run_id)
        st["artifact_path"] = _relpath((ARTIFACTS_DIR / "runs" / str(run_id)).resolve())
        st["artifact_ready"] = False
        _ESTADOS[job_id] = st  # flush early

        # 3) Seleccionar/preparar data (si req_norm.data_ref ya viene seteado, se reutiliza)
        selected_ref = _prepare_selected_data(req_norm, job_id)

        # 4) Crear estrategia (instancia nueva por job)
        strategy = _create_strategy(
            model_name=req_norm.modelo,
            hparams=run_hparams,
            job_id=job_id,
            dataset_id=req_norm.dataset_id,
            family=req_norm.family,
        )
        _safe_reset_strategy(strategy)

        # 5) Entrenar (Plantilla) - firma real: PlantillaEntrenamiento(estrategia)
        tpl = PlantillaEntrenamiento(estrategia=strategy)

        result = tpl.run(
            data_ref=str(selected_ref),
            epochs=int(req_norm.epochs or 5),
            hparams=run_hparams,
            model_name=str(req_norm.modelo),
        )

        if isinstance(result, dict) and result.get("status") == "failed":
            raise RuntimeError(result.get("error") or "Entrenamiento falló (sin detalle).")

        final_metrics = (result or {}).get("metrics") or {}
        history = (result or {}).get("history") or []

        # Enriquecer métricas mínimas para trazabilidad
        final_metrics.setdefault("family", str(req_norm.family))
        final_metrics.setdefault("dataset_id", str(req_norm.dataset_id))
        final_metrics.setdefault("model_name", str(req_norm.modelo))

        # -----------------------------------------------------------------
        # Warm-start: distinguir entre
        # - "resuelto" (se encontró un model_dir base), y
        # - "aplicado" (la estrategia efectivamente cargó pesos).
        #
        # Esto evita falsos positivos donde el path existe pero hay mismatch
        # de tarea/columnas/arquitectura y el warm-start se omite.
        # -----------------------------------------------------------------
        final_metrics["warm_start_resolved"] = bool(_ws_path is not None)

        # Copiar trazabilidad de resolución (para debugging/UI)
        if _ws_trace.get("warm_start_from"):
            final_metrics["warm_start_from"] = _ws_trace["warm_start_from"]
        if _ws_trace.get("warm_start_source_run_id"):
            final_metrics["warm_start_source_run_id"] = _ws_trace["warm_start_source_run_id"]
        if _ws_trace.get("warm_start_path"):
            final_metrics["warm_start_path"] = _ws_trace["warm_start_path"]

        # Determinar si el warm-start se aplicó realmente (según estrategia)
        ws_obj = final_metrics.get("warm_start")
        ws_applied = False
        if isinstance(ws_obj, dict):
            ws_applied = str(ws_obj.get("warm_start") or "").lower() == "ok"
        final_metrics["warm_started"] = bool(ws_applied)

        # 6) Guardar run en artifacts (fuente de verdad)
        req_snapshot = req_norm.model_dump()
        params = {"req": req_snapshot, "hparams": run_hparams}

        # ``run_id`` ya fue reservado antes de correr la plantilla.
        # Asegurar run_id en métricas para que predictor.json quede consistente.
        final_metrics.setdefault("run_id", str(run_id))

        # Estandarizar métricas según contrato por family (P2 Parte 4)
        _task_type_hint = str(final_metrics.get("task_type") or "").lower() or str(getattr(req_norm, "task_type", "") or "")
        final_metrics = standardize_run_metrics(
            final_metrics,
            family=str(req_norm.family or ""),
            task_type=_task_type_hint,
        )

        run_dir = save_run(
            run_id=str(run_id),
            job_id=str(job_id),
            dataset_id=str(req_norm.dataset_id),
            model_name=str(req_norm.modelo),
            data_ref=str(selected_ref),
            params=params,
            final_metrics=final_metrics,
            history=history,
        )

        # A partir de este punto, el run existe en disco.
        # Actualizar estado y emitir un evento explícito para UI/debug.
        st["artifact_ready"] = True
        st["artifact_path"] = _relpath(Path(run_dir))
        _ESTADOS[job_id] = st

        # Evento limpio (no altera training.completed): marca que artifacts ya son navegables.
        emit_training_persisted(
            job_id,
            run_id=str(run_id),
            artifact_path=str(st["artifact_path"]),
        )

        # (best-effort): persistir predictor bundle + modelo serializado para inferencia.
        # Importante: NO debe romper P0 si falla; solo deja bundle en placeholder.
        # Cargar el metrics.json completo (incluye params.req) para evitar nulls en predictor.json
        metrics_payload: dict[str, Any] = {}
        try:
            metrics_payload = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            # Fallback defensivo si por alguna razón no se puede leer el archivo recién escrito
            metrics_payload = {
                "run_id": str(run_id),
                "job_id": str(job_id),
                "dataset_id": str(req_norm.dataset_id),
                "model_name": str(req_norm.modelo),
                "params": params,
                **(final_metrics or {}),
            }

        _try_write_predictor_bundle(
            run_dir=run_dir,
            req_norm=req_norm,
            metrics=final_metrics,
            strategy=strategy,
        )

        _require_exported_model(run_dir, str(req_norm.modelo))

        # 7) Champion (si aplica) - usar metrics.json como contrato (incluye params.req)
        champion_promoted = None
        try:
            metrics_payload = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
            champ = maybe_update_champion(
                dataset_id=str(req_norm.dataset_id),
                model_name=str(req_norm.modelo),
                metrics=metrics_payload,
                source_run_id=str(run_id),
                family=str(req_norm.family) if req_norm.family else None,
            )
            if isinstance(champ, dict):
                champion_promoted = bool(champ.get("promoted"))
        except Exception:
            logger.exception("No se pudo evaluar/promover champion para run_id=%s", run_id)

        # 8) Estado final (completo)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        st.update(
            {
                "status": "completed",
                "progress": 1.0,
                "model": str(req_norm.modelo),
                "params": dict(run_hparams or {}),
                "run_id": str(run_id),
                "artifact_path": _relpath(Path(run_dir)),
                "metrics": final_metrics,
                "history": history,
                "champion_promoted": champion_promoted,
                "time_total_ms": float(dt_ms),
                "warm_start_trace": _ws_trace,
            }
        )
        _ESTADOS[job_id] = st

    except Exception as e:
        logger.exception("Falló entrenamiento job_id=%s", job_id)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # P2.3 FIX: Preservar ``run_id`` en estado fallido cuando ya fue
        # generado (e.g. si el error ocurrió en _try_write_predictor_bundle
        # o _require_exported_model, después de ``save_run``).  Permite
        # diagnóstico post-mortem del run parcial vía /modelos/estado/{id}.
        _failed_run_id = locals().get("run_id")
        _failed_run_dir = locals().get("run_dir")

        # Si alcanzamos a persistir el run, dejar artifact_path apuntando al directorio real.
        # Si no, mantener la ruta “reservada” (si existe) para trazabilidad.
        if _failed_run_dir is not None:
            try:
                st["artifact_path"] = _relpath(Path(_failed_run_dir))
                st["artifact_ready"] = True
            except Exception:
                # No romper el manejo de error por fallos de path.
                pass
        st.update(
            {
                "status": "failed",
                "error": str(e),
                "time_total_ms": float(dt_ms),
                "run_id": str(_failed_run_id) if _failed_run_id else st.get("run_id"),
            }
        )
        _ESTADOS[job_id] = st



def _run_sweep_training(sweep_id: str, req: SweepEntrenarRequest) -> None:
    t0 = time.perf_counter()

    st = _ESTADOS.get(sweep_id, {}) if isinstance(_ESTADOS.get(sweep_id), dict) else {}
    st.setdefault("job_id", sweep_id)
    st["job_type"] = "sweep"
    st["status"] = "running"
    st["progress"] = 0.0
    st["error"] = None

    started_at = dt.datetime.utcnow().isoformat() + "Z"

    # 1) Construir una sola selección de datos para TODO el sweep (comparabilidad)
    base_req = EntrenarRequest(
        modelo="rbm_restringida",  # placeholder; se overridea por candidato
        dataset_id=req.dataset_id,
        family=req.family,
        task_type=req.task_type,
        input_level=req.input_level,
        data_source=req.data_source,
        epochs=req.epochs,
        data_plan=req.data_plan,
        window_k=req.window_k,
        replay_size=req.replay_size,
        replay_strategy=req.replay_strategy,
        recency_lambda=req.recency_lambda,
        warm_start_from=req.warm_start_from,
        warm_start_run_id=req.warm_start_run_id,
        hparams=req.base_hparams,
        auto_prepare=True,
        # P2.6: plumb de texto hacia auto_prepare del feature-pack (sweep async).
        auto_text_feats=getattr(req, 'auto_text_feats', True),
        text_feats_mode=getattr(req, 'text_feats_mode', 'none'),
        text_col=getattr(req, 'text_col', None),
        text_n_components=int(getattr(req, 'text_n_components', 64) or 64),
        text_min_df=int(getattr(req, 'text_min_df', 2) or 2),
        text_max_features=int(getattr(req, 'text_max_features', 20000) or 20000),
        text_random_state=int(getattr(req, 'text_random_state', 42) or 42),
    )

    selected_ref = _prepare_selected_data(base_req, sweep_id)

    # 2) Armar candidatos (modelo × grid)
    modelos = [str(m) for m in (req.modelos or [])]
    grid_global = _expand_grid(req.hparams_grid or _default_sweep_grid())
    grid_by_model = req.hparams_by_model or {}

    candidates: list[dict[str, Any]] = []
    for m in modelos:
        grid = grid_by_model.get(m) or grid_global
        for g in grid:
            candidates.append({"model_name": m, "hparams": {**(req.base_hparams or {}), **(g or {})}})

    # cap
    candidates = candidates[: int(req.max_total_runs or 50)]

    # estado en memoria
    cand_state: list[dict[str, Any]] = []
    for c in candidates:
        cand_state.append(
            {
                "model_name": c["model_name"],
                "hparams": c["hparams"],
                "status": "queued",
                "child_job_id": None,
                "run_id": None,
                "metrics": None,
                "score": None,
                "error": None,
            }
        )

    st["params"] = {
        "dataset_id": req.dataset_id,
        "family": req.family,
        "n_candidates": len(cand_state),
        "selected_ref": str(selected_ref),
        # P2.6: trazabilidad de texto (útil para depurar sweep vs entrenamiento individual).
        "auto_text_feats": getattr(req, 'auto_text_feats', True),
        "text_feats_mode": getattr(req, 'text_feats_mode', 'none'),
        "text_col": getattr(req, 'text_col', None),
        "text_n_components": getattr(req, 'text_n_components', 64),
        "text_min_df": getattr(req, 'text_min_df', 2),
        "text_max_features": getattr(req, 'text_max_features', 20000),
        "text_random_state": getattr(req, 'text_random_state', 42),
    }

    best_overall: dict[str, Any] | None = None
    best_by_model: dict[str, dict[str, Any]] = {}

    from ...utils.runs_io import champion_score, load_run_metrics, load_current_champion, promote_run_to_champion  # noqa

    # 3) Ejecutar secuencial (robusto y determinista)
    for i, item in enumerate(cand_state, start=1):
        child_job_id = str(uuid.uuid4())
        item["child_job_id"] = child_job_id
        item["status"] = "running"
        _ESTADOS[sweep_id] = st  # flush

        _ESTADOS[child_job_id] = {
            "job_id": child_job_id,
            "job_type": "train",
            "status": "running",
            "progress": 0.0,
            "metrics": {},
            "history": [],
            "run_id": None,
            "error": None,
        }

        # request por candidato (reusa selected_ref para evitar re-sampling)
        cand_req = base_req.model_copy(
            update={
                "modelo": item["model_name"],
                "hparams": item["hparams"],
                "data_ref": str(selected_ref),
                "auto_prepare": False,
            }
        )

        # Normalización igual que /modelos/entrenar
        resolved = _build_run_hparams(cand_req, child_job_id)
        inferred_target_col = _infer_target_col(cand_req, resolved)

        update_payload = {
            "hparams": (cand_req.hparams or {}),
            "data_source": resolved.get("data_source"),
            "target_mode": resolved.get("target_mode"),
            "split_mode": resolved.get("split_mode"),
            "val_ratio": resolved.get("val_ratio"),
            "include_teacher_materia": resolved.get("include_teacher_materia"),
            "teacher_materia_mode": resolved.get("teacher_materia_mode"),
        }
        if inferred_target_col is not None:
            update_payload["target_col"] = inferred_target_col

        # Nota (Pydantic v2.11+): acceder a `model_fields` desde la instancia está deprecado.
        # Se debe acceder desde la clase.
        model_fields = getattr(type(cand_req), "model_fields", None)
        if isinstance(model_fields, dict):
            update_payload = {k: v for k, v in update_payload.items() if k in model_fields}

        cand_req_norm = cand_req.model_copy(update=update_payload)

        _run_training(child_job_id, cand_req_norm)

        child = _ESTADOS.get(child_job_id) or {}
        status = child.get("status")
        if status != "completed" or not child.get("run_id"):
            item["status"] = "failed"
            item["error"] = child.get("error") or "Entrenamiento falló (sin detalle)."
        else:
            item["status"] = "completed"
            item["run_id"] = child.get("run_id")

            metrics = load_run_metrics(str(item["run_id"]))
            item["metrics"] = metrics

            tier, score = champion_score(metrics or {})
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                score = -1e30
            item["score"] = [int(tier), float(score)]

            # best por modelo
            m = str(item["model_name"])
            prev = best_by_model.get(m)
            if (prev is None) or (tuple(item["score"]) > tuple(prev.get("score") or (-999, -1e30))):
                best_by_model[m] = dict(item)

            # best overall
            if (best_overall is None) or (tuple(item["score"]) > tuple(best_overall.get("score") or (-999, -1e30))):
                best_overall = dict(item)

        # progreso
        st["progress"] = float(i) / float(max(1, len(cand_state)))
        st["status"] = "running"
        _ESTADOS[sweep_id] = st

    # Hardening por si algo raro dejó best_overall vacío
    if best_overall is None:
        best_overall, best_by_model = _recompute_sweep_winners(cand_state)

    finished_at = dt.datetime.utcnow().isoformat() + "Z"

    summary_payload = {
        "sweep_id": sweep_id,
        "status": "completed",
        "family": req.family,
        "dataset_id": req.dataset_id,
        "created_at": started_at,
        "finished_at": finished_at,
        "n_candidates": len(cand_state),
        "n_completed": sum(1 for c in cand_state if c.get("status") == "completed"),
        "n_failed": sum(1 for c in cand_state if c.get("status") == "failed"),
        "best_overall": best_overall,
        "best_deployable": None,
        "best_by_model": best_by_model,
        "candidates": cand_state,
    }

    # Elegir best deployable (para promoción a champion global consumido por Predictions)
    best_deployable = None
    try:
        deployable_completed = [
            c
            for c in cand_state
            if c.get("status") == "completed"
            and c.get("run_id")
            and is_deployable_for_predictions(str(c.get("model_name")), str(req.family or ""))
        ]
        if deployable_completed:
            best_deployable = max(deployable_completed, key=lambda x: tuple(x.get("score") or (-999, -1e30)))
    except Exception:
        best_deployable = None

    summary_payload["best_deployable"] = best_deployable

    summary_path = _write_sweep_summary(sweep_id, summary_payload)

    # opcional: champion promotion
    try:
        best_for_promotion = best_deployable or best_overall
        current = load_current_champion(dataset_id=req.dataset_id)
        if current and best_for_promotion and current.get("run_id"):
            current_score = champion_score(load_run_metrics(str(current["run_id"])))
            if tuple(best_for_promotion["score"]) > tuple(current_score):
                promote_run_to_champion(
                    dataset_id=req.dataset_id,
                    run_id=str(best_for_promotion["run_id"]),
                    model_name=str(best_for_promotion["model_name"]),
                    family=str(req.family) if req.family else None,
                )
    except Exception:
        logger.exception("No se pudo evaluar/promover champion en sweep=%s", sweep_id)

    st.update(
        {
            "status": "completed",
            "progress": 1.0,
            "elapsed_s": float(time.perf_counter() - t0),
            "sweep_summary_path": str(summary_path),
            "sweep_best_overall": best_overall,
            "sweep_best_by_model": best_by_model,
        }
    )
    _ESTADOS[sweep_id] = st




@router.post(
    "/feature-pack/prepare",
    summary="Construye el feature-pack para un dataset",
)
def prepare_feature_pack_endpoint(
    dataset_id: str,
    input_uri: Optional[str] = None,
    force: bool = False,
    text_feats_mode: str = 'none',
    text_col: Optional[str] = None,
    text_n_components: int = 64,
    text_min_df: int = 2,
    text_max_features: int = 20000,
) -> Dict[str, str]:
    """Construye (o re-construye) el **feature-pack** de un dataset.

    Este endpoint habilita el modo *automático* desde la pestaña **Data**:

    - Tras subir/procesar un dataset, el frontend puede llamar a este endpoint
      para dejar listo ``artifacts/features/<dataset_id>/train_matrix.parquet``.

    También sirve como herramienta manual para debug/operación.

    Resolución de ``input_uri`` (si no se envía):

    1. ``data/processed/<dataset_id>.parquet``
    2. ``data/labeled/<dataset_id>_beto.parquet`` (vía :func:`neurocampus.data.datos_dashboard.resolve_labeled_path`)
    3. ``datasets/<dataset_id>.parquet``

    :param dataset_id: Identificador del dataset (ej. ``"2025-1"``).
    :param input_uri: Ruta/URI del dataset origen.
    :param force: Si True, re-genera incluso si el feature-pack ya existe.
    :param text_feats_mode: "none" (default) o "tfidf_lsa" para generar feat_t_* desde texto libre.
    :param text_col: Nombre de la columna de texto. Si None, se intenta detectar automáticamente.
    :param text_n_components: Dimensión máxima de LSA (solo tfidf_lsa).
    :param text_min_df: Frecuencia mínima de documento TF-IDF (solo tfidf_lsa).
    :param text_max_features: Tamaño máximo del vocabulario TF-IDF (solo tfidf_lsa).
    :returns: Diccionario de rutas relativas a los artefactos del feature-pack.
    """
    ds = str(dataset_id or "").strip()
    if not ds:
        raise HTTPException(status_code=400, detail="dataset_id es requerido")

    if input_uri:
        src_ref = _strip_localfs(str(input_uri))
        if not _abs_path(src_ref).exists():
            raise HTTPException(status_code=404, detail=f"input_uri no existe: {_abs_path(src_ref)}")
    else:
        # Resolver automáticamente el origen. Preferimos:
        # 1) labeled BETO (si existe)  -> incluye p_neg/p_neu/p_pos y permite evaluación real
        # 2) processed (Data Tab)
        # 3) datasets/<ds>.parquet
        candidates = []
        try:
            labeled = resolve_labeled_path(str(ds))
            candidates.append(labeled)
        except Exception:
            pass
        candidates.append(BASE_DIR / "data" / "processed" / f"{ds}.parquet")
        candidates.append(BASE_DIR / "datasets" / f"{ds}.parquet")

        src = next((p for p in candidates if p.exists()), None)
        if not src:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No se encontró dataset fuente para feature-pack de {ds}. Opciones:\n"
                    "- Genera labeled BETO (data/labeled/<ds>_beto.parquet)\n"
                    "- O procesa/carga el dataset en Data (data/processed/<ds>.parquet)\n"
                    "- O asegúrate de tener datasets/<ds>.parquet"
                ),
            )
        src_ref = _relpath(src)

    return _ensure_feature_pack(
        str(ds),
        input_uri=src_ref,
        force=force,
        text_feats_mode=text_feats_mode,
        text_col=text_col,
        text_n_components=int(text_n_components),
        text_min_df=int(text_min_df),
        text_max_features=int(text_max_features),
    )



@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks) -> EntrenarResponse:
    """Lanza un entrenamiento en background y retorna job_id."""
    job_id = str(uuid.uuid4())

    # hparams crudo normalizado (para pasarlo al training tal cual, excepto None)
    hp_norm_raw = _normalize_hparams(req.hparams)
    # versión "limpia" para UI (no debe pisar campos del request)
    hp_norm_ui = _prune_hparams_for_ui(hp_norm_raw)

    base_ref = _resolve_by_data_source(req)

    # Defaults consistentes (y sin None) para reflejar en UI
    resolved_run_hparams = _build_run_hparams(req, job_id)

    # --- NUEVO: inferir target_col de forma consistente ---
    inferred_target_col = _infer_target_col(req, resolved_run_hparams)

    # --- NUEVO: asegurar que el training reciba metadata mínima en hparams (sin meter None) ---
    _maybe_set(hp_norm_raw, "family", getattr(req, "family", None))
    _maybe_set(hp_norm_raw, "task_type", getattr(req, "task_type", None))
    _maybe_set(hp_norm_raw, "input_level", getattr(req, "input_level", None))
    _maybe_set(hp_norm_raw, "data_plan", getattr(req, "data_plan", None))

    # defaults ya resueltos (si aplican)
    _maybe_set(hp_norm_raw, "data_source", resolved_run_hparams.get("data_source"))
    _maybe_set(hp_norm_raw, "target_mode", resolved_run_hparams.get("target_mode"))
    _maybe_set(hp_norm_raw, "split_mode", resolved_run_hparams.get("split_mode"))

    # target_col inferido (clave para evaluación/snapshot)
    _maybe_set(hp_norm_raw, "target_col", inferred_target_col)



    # IMPORTANTE (Item 1):
    # - No permitir que hp_norm_ui contenga 'epochs' (u otros reservados) que pisen req.epochs.
    # - Colocar epochs AL FINAL del dict params para que siempre sea el valor del request.
    params_ui: Dict[str, Any] = {
        **hp_norm_ui,
        "dataset_id": _dataset_id(req),
        "periodo_actual": getattr(req, "periodo_actual", None),
        "metodologia": getattr(req, "metodologia", "periodo_actual"),
        "ventana_n": getattr(req, "ventana_n", None),
        # Ruta 2 (families)
        "family": getattr(req, "family", "sentiment_desempeno"),
        "task_type": getattr(req, "task_type", None),
        "input_level": getattr(req, "input_level", None),
        "target_col": inferred_target_col,
        # incremental (solo aplica a score_docente; se muestra si viene)
        "data_plan": getattr(req, "data_plan", None),
        "window_k": getattr(req, "window_k", None),
        "replay_size": getattr(req, "replay_size", None),
        "replay_strategy": getattr(req, "replay_strategy", None),
        "warm_start_from": getattr(req, "warm_start_from", None),
        "warm_start_run_id": getattr(req, "warm_start_run_id", None),
        # reflejar defaults consistentes (y ya “limpios”)
        "data_source": resolved_run_hparams.get("data_source", "feature_pack"),
        "target_mode": resolved_run_hparams.get("target_mode", "sentiment_probs"),
        "split_mode": resolved_run_hparams.get("split_mode", "temporal"),
        "val_ratio": resolved_run_hparams.get("val_ratio", 0.2),
        "include_teacher_materia": resolved_run_hparams.get("include_teacher_materia", True),
        "teacher_materia_mode": resolved_run_hparams.get("teacher_materia_mode", "embed"),
        "auto_prepare": getattr(req, "auto_prepare", True),
        "data_ref": getattr(req, "data_ref", None) or base_ref,
        "job_id": job_id,
        # Item 1: epochs del request SIEMPRE
        "epochs": req.epochs,
    }


    _ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.0,  # Item 2
        "metrics": {},
        "history": [],
        "model": req.modelo,
        "params": params_ui,
        "error": None,
        "run_id": None,
        "artifact_path": None,
        "champion_promoted": False,
    }

    # Normaliza hparams, preservando el resto del request intacto
    # + persistir defaults resueltos para trazabilidad (params.req.*)
    update_payload: Dict[str, Any] = {
        "hparams": hp_norm_raw,
        # defaults “efectivos” (evita que params.req.target_mode quede como el default del schema)
        "data_source": resolved_run_hparams.get("data_source"),
        "target_mode": resolved_run_hparams.get("target_mode"),
        "split_mode": resolved_run_hparams.get("split_mode"),
        "val_ratio": resolved_run_hparams.get("val_ratio"),
        "include_teacher_materia": resolved_run_hparams.get("include_teacher_materia"),
        "teacher_materia_mode": resolved_run_hparams.get("teacher_materia_mode"),
    }

    # Persistimos también target_col inferido en el request (para que quede en params.req.target_col)
    if inferred_target_col is not None:
        update_payload["target_col"] = inferred_target_col

    try:
        req_norm = req.model_copy(update=update_payload)
    except AttributeError:
        req_norm = req.copy(update=update_payload)

    bg.add_task(_run_training, job_id, req_norm)
    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


# ---------------------------------------------------------------------------
# Sweep determinístico P2 Parte 5
# ---------------------------------------------------------------------------

_SWEEP_MODEL_ORDER: list[str] = ["rbm_general", "rbm_restringida", "dbm_manual"]


def _pick_best_deterministic(
    candidates: list[dict],
    *,
    primary_metric: str,
    mode: str,
) -> dict | None:
    """
    Elige el mejor candidato de forma determinística.

    Criterios (en orden de prioridad):
    1. primary_metric_value (mayor si mode=max, menor si mode=min)
    2. Tie-breaker: model_name en _SWEEP_MODEL_ORDER (primero en el orden canónico)
    3. Tie-breaker final: run_id (orden lexicográfico, más pequeño = más temprano)
    """
    completed = [c for c in candidates if c.get("status") == "completed" and c.get("run_id")]
    if not completed:
        return None

    def _sort_key(c: dict):
        val = c.get("primary_metric_value")
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            # Peores posibles: mayor = peor para max, menor = peor para min
            metric_key = float("-inf") if mode == "max" else float("inf")
        else:
            metric_key = float(val) if mode == "max" else -float(val)

        model_idx = _SWEEP_MODEL_ORDER.index(c["model_name"]) if c["model_name"] in _SWEEP_MODEL_ORDER else 999
        run_id_key = str(c.get("run_id") or "")
        return (metric_key, -model_idx, run_id_key)  # mayor métrica mejor; menor model_idx mejor

    return max(completed, key=_sort_key)


def _run_model_sweep(sweep_id: str, req: "ModelSweepRequest") -> dict:
    """
    Ejecuta sweep determinístico sobre los modelos en req.models.

    Reutiliza _run_training() por candidato (mismo flujo que /modelos/entrenar).
    Devuelve payload del resumen para construir ModelSweepResponse.
    """
    t0 = time.perf_counter()
    from ...models.utils.metrics_contract import primary_metric_for_family

    # Contrato de métricas para esta family
    primary_metric, pm_mode = primary_metric_for_family(str(req.family or ""), task_type="")

    # Orden canónico de modelos (fijo para determinismo)
    models_ordered = [m for m in _SWEEP_MODEL_ORDER if m in req.models]
    # Añadir modelos no estándar al final (para extensibilidad futura)
    models_ordered += [m for m in req.models if m not in _SWEEP_MODEL_ORDER]

    # Construir request base compartido (mismos datos para todos los candidatos)
    base_req = EntrenarRequest(
        modelo="rbm_restringida",  # placeholder
        dataset_id=req.dataset_id,
        family=req.family,
        data_source=req.data_source,
        epochs=req.epochs,
        data_plan=req.data_plan,
        window_k=req.window_k,
        replay_size=req.replay_size,
        replay_strategy=req.replay_strategy,
        warm_start_from=req.warm_start_from,
        warm_start_run_id=req.warm_start_run_id,
        hparams={**req.base_hparams, "seed": req.seed},
        auto_prepare=req.auto_prepare,
        # P2.6: parámetros opcionales de texto para auto_prepare del feature-pack.
        auto_text_feats=getattr(req, "auto_text_feats", True),
        text_feats_mode=getattr(req, "text_feats_mode", "none"),
        text_col=getattr(req, "text_col", None),
        text_n_components=getattr(req, "text_n_components", 64),
        text_min_df=getattr(req, "text_min_df", 2),
        text_max_features=getattr(req, "text_max_features", 20000),
        text_random_state=getattr(req, "text_random_state", 42),
    )

    # Selección de datos única (comparabilidad)
    selected_ref = _prepare_selected_data(base_req, sweep_id)

    candidates_out: list[dict] = []

    for model_name in models_ordered[: int(req.max_candidates)]:
        child_job_id = str(uuid.uuid4())

        # Hparams = base + override por modelo
        merged_hparams = {
            **req.base_hparams,
            "seed": req.seed,
            **(req.hparams_overrides.get(model_name) or {}),
        }

        cand_req = base_req.model_copy(
            update={
                "modelo": model_name,
                "hparams": merged_hparams,
                "data_ref": str(selected_ref),
                "auto_prepare": False,
            }
        )

        # Normalización igual que /modelos/entrenar
        resolved = _build_run_hparams(cand_req, child_job_id)
        inferred_target_col = _infer_target_col(cand_req, resolved)

        update_payload: dict = {
            "hparams": merged_hparams,
            "data_source": resolved.get("data_source"),
            "target_mode": resolved.get("target_mode"),
            "split_mode": resolved.get("split_mode"),
            "val_ratio": resolved.get("val_ratio"),
            "include_teacher_materia": resolved.get("include_teacher_materia"),
            "teacher_materia_mode": resolved.get("teacher_materia_mode"),
        }
        if inferred_target_col is not None:
            update_payload["target_col"] = inferred_target_col

        model_fields = getattr(type(cand_req), "model_fields", None)
        if isinstance(model_fields, dict):
            update_payload = {k: v for k, v in update_payload.items() if k in model_fields}

        cand_req_norm = cand_req.model_copy(update=update_payload)

        # Estado hijo
        _ESTADOS[child_job_id] = {
            "job_id": child_job_id,
            "job_type": "train",
            "status": "running",
            "progress": 0.0,
            "metrics": {},
            "history": [],
            "run_id": None,
            "error": None,
        }

        _run_training(child_job_id, cand_req_norm)

        child = _ESTADOS.get(child_job_id) or {}
        cand: dict = {
            "model_name": model_name,
            "run_id": child.get("run_id"),
            "status": child.get("status", "failed"),
            "primary_metric_value": None,
            "metrics": None,
            "error": None,
        }

        if child.get("status") == "completed" and child.get("run_id"):
            from ...utils.runs_io import load_run_metrics as _lrm
            metrics = _lrm(str(child["run_id"]))
            cand["metrics"] = metrics
            pmv = metrics.get("primary_metric_value")
            if isinstance(pmv, (int, float)) and math.isfinite(float(pmv)):
                cand["primary_metric_value"] = float(pmv)
        else:
            cand["status"] = "failed"
            cand["error"] = child.get("error") or "Entrenamiento falló"

        candidates_out.append(cand)

    # Elegir best determinístico.
    # Importante: el champion GLOBAL es consumido por Predictions, por lo que
    # preferimos modelos "deployable" para esa family.
    best_overall = _pick_best_deterministic(candidates_out, primary_metric=primary_metric, mode=pm_mode)
    eligible = [
        c for c in candidates_out
        if is_deployable_for_predictions(str(c.get("model_name")), str(req.family or ""))
    ]
    best = _pick_best_deterministic(eligible, primary_metric=primary_metric, mode=pm_mode) or best_overall

    # Champion promotion
    champion_promoted = False
    champion_run_id: str | None = None

    if req.auto_promote_champion and best and best.get("run_id"):
        try:
            from ...utils.runs_io import (
                maybe_update_champion as _muc,
                load_run_metrics as _lrm2,
            )
            best_metrics = best.get("metrics") or _lrm2(str(best["run_id"]))
            result = _muc(
                dataset_id=req.dataset_id,
                model_name=str(best["model_name"]),
                metrics=best_metrics,
                source_run_id=str(best["run_id"]),
                family=str(req.family),
            )
            if isinstance(result, dict) and result.get("promoted"):
                champion_promoted = True
                champion_run_id = str(best["run_id"])
        except Exception:
            logger.exception("No se pudo promover champion en sweep=%s", sweep_id)

    elapsed = float(time.perf_counter() - t0)

    return {
        "sweep_id": sweep_id,
        "status": "completed",
        "dataset_id": req.dataset_id,
        "family": str(req.family),
        "primary_metric": primary_metric,
        "primary_metric_mode": pm_mode,
        "candidates": candidates_out,
        "best": best,
        "best_overall": best_overall,
        "champion_promoted": champion_promoted,
        "champion_run_id": champion_run_id,
        "n_completed": sum(1 for c in candidates_out if c["status"] == "completed"),
        "n_failed": sum(1 for c in candidates_out if c["status"] == "failed"),
        "elapsed_s": elapsed,
    }


@router.post(
    "/sweep",
    response_model=ModelSweepResponse,
    summary="Sweep determinístico: entrena N modelos y elige el mejor",
)
def model_sweep(req: ModelSweepRequest) -> ModelSweepResponse:
    """
    Entrena los modelos en ``req.models`` con los mismos datos y elige el mejor
    por ``primary_metric`` (contrato P2 Parte 4).

    - Ejecución **síncrona** (espera hasta completar todos los candidatos).
    - Tie-breaker determinístico: primary_metric_value → model_name → run_id.
    - Si ``auto_promote_champion=true``, promueve el best a champion.

    Usa el mismo flujo de entrenamiento que ``POST /modelos/entrenar``,
    garantizando que las métricas sean comparables.
    """
    # Validar modelos
    allowed = set(_STRATEGY_CLASSES.keys())
    invalid = [m for m in req.models if m not in allowed]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Modelos no soportados: {invalid}. Permitidos: {sorted(allowed)}",
        )
    if not req.dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id es requerido")
    if not req.family:
        raise HTTPException(status_code=422, detail="family es requerida")

    sweep_id = str(uuid.uuid4())

    result = _run_model_sweep(sweep_id, req)

    # Persistir resumen
    try:
        summary_path = _write_sweep_summary(sweep_id, result)
        result["summary_path"] = str(summary_path)
    except Exception:
        logger.exception("No se pudo persistir sweep summary sweep=%s", sweep_id)

    # Construir response (mapear candidatos a schema)
    candidates_resp = [
        ModelSweepCandidateResult(
            model_name=c["model_name"],
            run_id=c.get("run_id"),
            status=c.get("status", "failed"),
            primary_metric_value=c.get("primary_metric_value"),
            metrics=c.get("metrics"),
            error=c.get("error"),
        )
        for c in result["candidates"]
    ]

    best_resp: ModelSweepCandidateResult | None = None
    if result["best"]:
        b = result["best"]
        best_resp = ModelSweepCandidateResult(
            model_name=b["model_name"],
            run_id=b.get("run_id"),
            status=b.get("status", "completed"),
            primary_metric_value=b.get("primary_metric_value"),
            metrics=b.get("metrics"),
        )

    return ModelSweepResponse(
        sweep_id=sweep_id,
        status="completed",
        dataset_id=req.dataset_id,
        family=str(req.family),
        primary_metric=result["primary_metric"],
        primary_metric_mode=result["primary_metric_mode"],
        candidates=candidates_resp,
        best=best_resp,
        champion_promoted=result["champion_promoted"],
        champion_run_id=result.get("champion_run_id"),
        n_completed=result["n_completed"],
        n_failed=result["n_failed"],
        summary_path=result.get("summary_path"),
        elapsed_s=result.get("elapsed_s"),
    )


@router.post("/entrenar/sweep", response_model=SweepEntrenarResponse)
def entrenar_sweep(req: SweepEntrenarRequest, bg: BackgroundTasks) -> SweepEntrenarResponse:
    sweep_id = str(uuid.uuid4())

    _ESTADOS[sweep_id] = {
        "job_id": sweep_id,
        "job_type": "sweep",
        "status": "running",
        "progress": 0.0,
        "metrics": {},
        "history": [],
        "params": {
            "dataset_id": req.dataset_id,
            "family": req.family,
            "modelos": req.modelos,
        },
        "error": None,
    }

    bg.add_task(_run_sweep_training, sweep_id, req)
    return SweepEntrenarResponse(sweep_id=sweep_id, status="running", message="Sweep lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str):
    st = _ESTADOS.get(job_id)

    # 1) Si está en memoria, devuélvelo normal
    if isinstance(st, dict):
        payload = dict(st)
        payload.setdefault("job_id", job_id)
        payload.setdefault("status", "unknown")
        payload.setdefault("progress", 0.0)
        payload.setdefault("model", st.get("model"))
        payload.setdefault("params", st.get("params") or {})
        payload.setdefault("metrics", st.get("metrics") or {})
        payload.setdefault("history", st.get("history") or [])
        payload.setdefault("run_id", st.get("run_id"))
        payload.setdefault("artifact_path", st.get("artifact_path"))
        payload.setdefault("champion_promoted", st.get("champion_promoted"))
        payload.setdefault("job_type", st.get("job_type"))
        payload.setdefault("sweep_summary_path", st.get("sweep_summary_path"))
        payload.setdefault("sweep_best_overall", st.get("sweep_best_overall"))
        payload.setdefault("sweep_best_by_model", st.get("sweep_best_by_model"))
        payload.setdefault("warm_start_trace", st.get("warm_start_trace"))

        # time_total_ms: preferido si existe, sino derivar de elapsed_s
        if payload.get("time_total_ms") is None and payload.get("elapsed_s") is not None:
            try:
                payload["time_total_ms"] = float(payload["elapsed_s"]) * 1000.0
            except Exception:
                payload["time_total_ms"] = None

        return payload

    # 2) Fallback: si es sweep y existe summary.json, úsalo como fuente de verdad
    summary_path = _sweeps_dir() / job_id / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        status = payload.get("status") or "completed"
        return {
            "job_id": job_id,
            "status": status,
            "progress": 1.0 if status == "completed" else 0.0,
            "error": payload.get("error"),
            "sweep_summary_path": str(summary_path),
        }

    return {"job_id": job_id, "status": "unknown", "progress": 0.0, "error": None, "sweep_summary_path": None}



@router.post(
    "/champion/promote",
    response_model=ChampionInfo,
    summary="Promueve un run existente a champion (manual)",
)
def promote_champion(req: PromoteChampionRequest) -> ChampionInfo:
    """
    Promueve un run a champion.

    Semántica P0:
      - 422 si run_id es inválido (null / vacío / "null" / "none")
      - 404 si no existe el run o falta metrics.json
      - 200 si promueve correctamente
    """
    # Validación defensiva (además de pydantic, por compat legacy)
    rid = str(getattr(req, "run_id", "") or "").strip()
    if (not rid) or (rid.lower() in {"null", "none", "nil"}):
        raise HTTPException(status_code=422, detail="run_id inválido")

    # ------------------------------------------------------------------
    # P2.1 FIX: permitir promote con payload mínimo (run_id [+family]).
    #
    # El frontend de Modelos puede enviar sólo `run_id` (y opcionalmente `family`)
    # para promover. Históricamente el schema exigía dataset_id/model_name, lo que
    # resultaba en 422 aunque el run existiera.
    #
    # Regla:
    # - Si falta dataset_id, inferirlo desde metrics.json (o, como fallback, desde
    #   el formato <dataset>__<model>__...).
    # - Si falta model_name, se deja como None y `promote_run_to_champion` lo infiere
    #   de metrics.json/params.req.
    # - Si no existe metrics.json, responder 404 (semántica P0) incluso si faltan
    #   otros campos.
    # ------------------------------------------------------------------

    dataset_id = str(getattr(req, "dataset_id", "") or "").strip() or None
    model_name = str(getattr(req, "model_name", "") or "").strip() or None

    if not dataset_id:
        run_dir = (ARTIFACTS_DIR / "runs" / rid).resolve()
        mp = run_dir / "metrics.json"
        legacy_mp = run_dir / "model" / "metrics.json"

        # Semántica P0: si no hay métricas persistidas para el run, es 404.
        if not mp.exists() and not legacy_mp.exists():
            raise HTTPException(status_code=404, detail=f"No existe metrics.json para run_id={rid}")

        # Leer métricas (best-effort) para inferir dataset_id.
        metrics_payload: dict[str, Any] = {}
        chosen = mp if mp.exists() else legacy_mp
        try:
            metrics_payload = json.loads(chosen.read_text(encoding="utf-8"))
            if not isinstance(metrics_payload, dict):
                metrics_payload = {}
        except Exception:
            metrics_payload = {}

        # Si venimos de layout legacy (model/metrics.json), hacemos mirror best-effort
        # a run_dir/metrics.json para que `promote_run_to_champion` encuentre el archivo.
        if chosen == legacy_mp and not mp.exists() and metrics_payload:
            try:
                mp.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                # No romper promote: si no se puede escribir, promote_run_to_champion
                # mantendrá su semántica (FileNotFoundError -> 404).
                pass

        req_ctx = {}
        try:
            params = metrics_payload.get("params") if isinstance(metrics_payload, dict) else None
            if isinstance(params, dict) and isinstance(params.get("req"), dict):
                req_ctx = params.get("req") or {}
        except Exception:
            req_ctx = {}

        dataset_id = (
            str(metrics_payload.get("dataset_id") or "").strip()
            or str(req_ctx.get("dataset_id") or req_ctx.get("periodo") or "").strip()
            or None
        )

        # Último fallback: parsear <dataset_id>__<model_name>__<timestamp>__<job>
        if not dataset_id:
            parts = rid.split("__")
            if len(parts) >= 1 and parts[0].strip():
                dataset_id = parts[0].strip()

    if not dataset_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "dataset_id es requerido para promover champion (no se pudo inferir desde metrics.json/run_id)."
            ),
        )

    try:
        champ = _call_with_accepted_kwargs(
            promote_run_to_champion,
            dataset_id=dataset_id,
            run_id=rid,
            model_name=model_name,
            family=getattr(req, "family", None),
        )

        # fallback defensivo para source_run_id (misma lógica que GET /champion)
        if isinstance(champ, dict) and not champ.get("source_run_id"):
            m = champ.get("metrics") or {}
            if isinstance(m, dict) and m.get("run_id"):
                champ = dict(champ)
                champ["source_run_id"] = m.get("run_id")

        return ChampionInfo(**champ)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"No se pudo promover champion: {e}")


@router.get(
    "/runs",
    response_model=list[RunSummary],
    summary="Lista runs de entrenamiento/auditoría de modelos",
)
def get_runs(
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
    family: Optional[str] = None,  # <-- NUEVO
) -> List[RunSummary]:
    """Devuelve un resumen de runs encontrados en artifacts/runs.

    Extensión Ruta 2:
    - Permite filtrar por `family` (sentiment_desempeno | score_docente).
    - Compatibilidad: si un run legacy no tiene `family`, se asume sentiment_desempeno.
    """
    ds = dataset_id or dataset or periodo

    runs = _call_with_accepted_kwargs(list_runs, model_name=model_name, dataset_id=ds, periodo=ds, family=family)

    if family:
        fam = str(family).lower()
        filtered = []
        for r in (runs or []):
            rf = (r.get("family") or "sentiment_desempeno")
            if str(rf).lower() == fam:
                filtered.append(r)
        runs = filtered


    # Backfill consistente (P2.1): completar contexto desde metrics.json + predictor.json
    def _missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "unknown", "n/a"}:
            return True
        return False

    hydrated: List[RunSummary] = []
    for r in (runs or []):
        if not isinstance(r, dict):
            continue

        run_id = str(r.get("run_id") or "")
        run_dir = None
        try:
            if r.get("artifact_path"):
                # `artifact_path` puede venir como:
                #   - absolute path
                #   - "artifacts/..." (contrato lógico)
                #   - "localfs://..." (compatibilidad)
                #
                # Usamos `_abs_path` para resolverlo de forma consistente con
                # `NC_ARTIFACTS_DIR` y evitar depender del cwd del proceso (especialmente
                # en Windows / despliegues con volumen montado).
                ref = str(r.get("artifact_path") or "").strip()
                # Expandir "~" *solo* para rutas locales (sin esquema localfs://).
                ref_local = _strip_localfs(ref)
                ref_local = str(Path(ref_local).expanduser())
                run_dir = _abs_path(ref_local)
            elif run_id:
                run_dir = (ARTIFACTS_DIR / "runs" / run_id).resolve()
        except Exception:
            run_dir = None

        # Cargar metrics.json completo si falta contexto
        full_metrics: Dict[str, Any] = {}
        try:
            if run_dir:
                mp = run_dir / "metrics.json"
                if mp.exists():
                    full_metrics = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            full_metrics = {}

        # predictor.json (si existe) como fuente de fallback
        predictor_manifest = None
        try:
            if run_dir:
                pj = run_dir / "predictor.json"
                if pj.exists():
                    predictor_manifest = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            predictor_manifest = None

        req = (full_metrics.get("params") or {}).get("req") if isinstance(full_metrics.get("params"), dict) else {}
        if not isinstance(req, dict):
            req = {}

        hint_model = r.get("model_name") or full_metrics.get("model_name") or full_metrics.get("model") or req.get("model_name") or req.get("modelo")
        hint_family = r.get("family") or family or full_metrics.get("family") or req.get("family")
        hint_ds = r.get("dataset_id") or ds or full_metrics.get("dataset_id") or req.get("dataset_id")

        ctx = fill_context(
            family=hint_family,
            dataset_id=hint_ds,
            model_name=hint_model,
            metrics=full_metrics or {},
            predictor_manifest=predictor_manifest,
        )

        # Backfill del objeto `context` (P2.1): el listado de runs debe ser autosuficiente
        # para la UI. Si `context` viene null/None desde runs_io, lo reconstruimos a partir
        # de metrics.json + predictor.json.
        if _missing(r.get("context")) and isinstance(ctx, dict) and ctx:
            r["context"] = ctx

        # Backfill de métricas resumidas (P2.4): garantizar que el listado incluya
        # primary_metric y primary_metric_value cuando existan en metrics.json.
        if not isinstance(r.get("metrics"), dict):
            r["metrics"] = {}

        def _mget(key: str) -> Any:
            # 1) top-level en metrics.json
            if isinstance(full_metrics, dict) and (key in full_metrics) and (not _missing(full_metrics.get(key))):
                return full_metrics.get(key)
            # 2) algunos writers guardan métricas bajo "final_metrics"
            fm = full_metrics.get("final_metrics") if isinstance(full_metrics, dict) and isinstance(full_metrics.get("final_metrics"), dict) else {}
            if (key in fm) and (not _missing(fm.get(key))):
                return fm.get(key)
            # 3) fallback: métricas bajo "metrics"
            mm = full_metrics.get("metrics") if isinstance(full_metrics, dict) and isinstance(full_metrics.get("metrics"), dict) else {}
            if (key in mm) and (not _missing(mm.get(key))):
                return mm.get(key)
            return None

        ms = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        for k in (
            "primary_metric",
            "primary_metric_mode",
            "primary_metric_value",
            "val_accuracy",
            "val_f1_macro",
            "accuracy",
            "f1_macro",
            "n_train",
            "n_val",
        ):
            v = _mget(k)
            if _missing(ms.get(k)) and (v is not None):
                ms[k] = v
        r["metrics"] = ms


        # Backfill top-level fields del resumen
        for key in ("family", "dataset_id", "model_name", "task_type", "input_level", "target_col", "data_plan", "data_source"):
            if _missing(r.get(key)) and (ctx.get(key) is not None):
                r[key] = ctx.get(key)


        hydrated.append(RunSummary(**r))

    return hydrated



@router.get(
    "/runs/{run_id}",
    response_model=RunDetails,
    summary="Detalles completos de un run (incluye config si existe)",
)
def get_run_details(run_id: str) -> RunDetails:
    """Devuelve detalles completos de un run leyendo artifacts del filesystem."""
    details = load_run_details(run_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Run {run_id} no encontrado")

    # Backfill consistente (P2.1): completar contexto desde metrics.json + predictor.json
    metrics = details.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    predictor_manifest = None
    try:
        # `artifact_path` puede ser relativo ("artifacts/...") o un URI `localfs://...`.
        # Resolverlo con `_abs_path` evita fallos al ejecutar desde distintos cwd.
        rd_ref = str(details.get("artifact_path") or "").strip()
        if rd_ref:
            rd_local = _strip_localfs(rd_ref)
            rd_local = str(Path(rd_local).expanduser())
            rd = _abs_path(rd_local)
        else:
            # Fallback: ubicación estándar del run
            rd = (ARTIFACTS_DIR / "runs" / run_id).resolve()

        pj = rd / "predictor.json"
        if pj.exists():
            predictor_manifest = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        predictor_manifest = None

    req = (metrics.get("params") or {}).get("req") if isinstance(metrics.get("params"), dict) else {}
    if not isinstance(req, dict):
        req = {}

    hint_family = details.get("family") or metrics.get("family") or req.get("family")
    hint_ds = details.get("dataset_id") or metrics.get("dataset_id") or req.get("dataset_id")
    hint_model = metrics.get("model_name") or metrics.get("model") or req.get("model_name") or req.get("modelo")

    ctx = fill_context(
        family=hint_family,
        dataset_id=hint_ds,
        model_name=hint_model,
        metrics=metrics,
        predictor_manifest=predictor_manifest,
    )

    def _missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "unknown", "n/a"}:
            return True
        return False


    # Backfill del objeto `context` (P2.1):
    # - `load_run_details` puede devolver context=None (runs legacy).
    # - La UI de Modelos consume `context` para renderizar metadatos sin depender
    #   de rutas adicionales.
    if _missing(details.get("context")) and isinstance(ctx, dict) and ctx:
        details["context"] = ctx

    for key in ("family", "dataset_id", "model_name", "task_type", "input_level", "target_col", "data_plan", "data_source"):
        if _missing(details.get(key)) and (ctx.get(key) is not None):
            details[key] = ctx.get(key)

    # Enriquecer métricas (sin persistir a disco)
    for key in ("family", "task_type", "input_level", "target_col", "data_plan", "data_source"):
        if _missing(metrics.get(key)) and (ctx.get(key) is not None):
            metrics[key] = ctx.get(key)
    details["metrics"] = metrics

    return RunDetails(**details)

@router.get("/sweeps/{sweep_id}", response_model=SweepSummary)
def get_sweep_summary(sweep_id: str) -> SweepSummary:
    p = _sweeps_dir() / str(sweep_id) / "summary.json"
    if not p.exists():
        # si aún corre, devolvemos lo que haya en memoria
        st = _ESTADOS.get(sweep_id) or {}
        return SweepSummary(
            sweep_id=sweep_id,
            dataset_id=str((st.get("params") or {}).get("dataset_id") or ""),
            family=str((st.get("params") or {}).get("family") or "score_docente"),
            status=str(st.get("status") or "unknown"),
            summary_path=str(p) if p.exists() else None,
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["summary_path"] = str(p)

    # Compatibilidad por si existen llaves antiguas en summary.json
    if "best_overall" not in payload and "sweep_best_overall" in payload:
        payload["best_overall"] = payload.get("sweep_best_overall")
    if "best_by_model" not in payload and "sweep_best_by_model" in payload:
        payload["best_by_model"] = payload.get("sweep_best_by_model")

    from ...utils.runs_io import champion_score  # noqa

    def _hydrate_candidate(cand: Any, default_model_name: Optional[str] = None) -> Any:
        if not isinstance(cand, dict):
            return cand

        if default_model_name and not cand.get("model_name"):
            cand["model_name"] = default_model_name

        metrics = cand.get("metrics")
        if isinstance(metrics, dict):
            if not cand.get("model_name") and metrics.get("model_name"):
                cand["model_name"] = metrics.get("model_name")
            if not cand.get("run_id") and metrics.get("run_id"):
                cand["run_id"] = metrics.get("run_id")

            if cand.get("score") is None:
                try:
                    tier, sc = champion_score(metrics or {})
                    cand["score"] = [int(tier), float(sc)]
                except Exception:
                    pass

        return cand

    bbm = payload.get("best_by_model") or {}
    if isinstance(bbm, dict):
        for k, v in list(bbm.items()):
            bbm[k] = _hydrate_candidate(v, default_model_name=k)
        payload["best_by_model"] = bbm

    bo = payload.get("best_overall")
    payload["best_overall"] = _hydrate_candidate(bo)


    # Normalización robusta:
    # - Si best_overall no viene o viene vacío, derivarlo desde best_by_model (ya calculado)
    bo = payload.get("best_overall")
    if (bo is None) or (isinstance(bo, dict) and not bo.get("run_id")):
        bbm = payload.get("best_by_model") or {}
        if isinstance(bbm, dict) and bbm:
            def _score_tuple(v: dict[str, Any]) -> tuple[int, float]:
                s = v.get("score") or [-999, -1e30]
                try:
                    return (int(s[0]), float(s[1]))
                except Exception:
                    return (-999, -1e30)

            payload["best_overall"] = max(bbm.values(), key=_score_tuple)

            # Persistir la corrección para UI/offline (idempotente)
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SweepSummary(**payload)


@router.get(
    "/champion",
    response_model=ChampionInfo,
    summary="Devuelve info del modelo campeón actual (por dataset o legacy)",
)
def get_champion(
    dataset_id: Optional[str] = None,
    dataset: Optional[str] = None,
    periodo: Optional[str] = None,
    model_name: str = "rbm_restringida",
    family: Optional[str] = None,
):
    ds = dataset_id or dataset or periodo
    if not ds:
        raise HTTPException(status_code=400, detail="dataset_id (o dataset/periodo) es requerido")

    # 1) Cargar champion (usa wrapper si existe y acepta kwargs)
    try:
        champ = _call_with_accepted_kwargs(
            load_current_champion,
            dataset_id=str(ds),
            model_name=model_name,
            family=family,
        )
        # P2.3 FIX: El fallback anterior llamaba a ``load_dataset_champion``
        # sin filtrar por ``model_name``, lo que provocaba que al pedir
        # el champion de "rbm_general" se devolviera "rbm_restringida"
        # (o viceversa).  Ahora solo se usa el fallback cuando el motivo
        # del None NO es el filtro de model_name (i.e. cuando simplemente
        # no existe champion), y se re-filtra tras obtener el resultado.
        if champ is None:
            champ = _call_with_accepted_kwargs(
                load_dataset_champion,
                dataset_id=str(ds),
                family=family,
            )
            # Re-filtrar por model_name: si el champion encontrado no coincide
            # con el modelo solicitado, descartar para evitar cruce de modelos.
            if champ is not None and model_name:
                _champ_mn = (champ.get("model_name") or champ.get("model") or "").strip().lower()
                _req_mn = model_name.strip().lower()
                if _champ_mn and _champ_mn != _req_mn:
                    champ = None
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Importante: no dejar que esto se vaya como 500 "text/plain" opaco
        raise HTTPException(status_code=500, detail=f"No se pudo cargar champion: {e}")

    if not champ:
        raise HTTPException(
            status_code=404,
            detail=f"No hay champion para dataset_id={ds}" + (f" y family={family}" if family else ""),
        )

    # 2) Backfill mínimo de campos críticos (sin tocar lo que ya viene bien)
    # family (prioridad: champ > query)
    champ_family = champ.get("family") or family
    if champ_family:
        champ["family"] = champ_family

    # source_run_id (si falta, derivar de metrics.run_id)
    if not champ.get("source_run_id"):
        champ["source_run_id"] = (champ.get("metrics") or {}).get("run_id")

    # path es OBLIGATORIO en ChampionInfo => si falta, lo calculamos
    if not champ.get("path"):
        artifacts_dir = (ARTIFACTS_DIR / "champions").resolve()
        ds_dir = (artifacts_dir / champ_family / str(ds)) if champ_family else (artifacts_dir / str(ds))

        mn = champ.get("model_name") or model_name
        model_dir = ds_dir / str(mn) if mn else None

        # Mantener semántica: si existe el dir del modelo úsalo; si no, usa ds_dir
        if model_dir and model_dir.exists():
            champ["path"] = _relpath(model_dir)
        else:
            champ["path"] = _relpath(ds_dir)

    # ------------------------------------------------------------
    # 2.b) Backfill consistente de contexto (P2.1)
    # ------------------------------------------------------------
    # Regla única: metrics.params.req -> metrics.* -> predictor.json -> fallback por family
    def _missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "unknown", "n/a"}:
            return True
        return False

    metrics = champ.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    # Best-effort: leer predictor.json desde la carpeta del champion (si existe)
    predictor_manifest = None
    try:
        champ_dir = Path(str(champ.get("path") or "")).expanduser().resolve()
        pj = champ_dir / "predictor.json"
        if pj.exists():
            predictor_manifest = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        predictor_manifest = None

    ctx = fill_context(
        family=champ.get("family") or family,
        dataset_id=str(ds),
        model_name=champ.get("model_name") or model_name,
        metrics=metrics,
        predictor_manifest=predictor_manifest,
    )

    # Rellenar solo si falta (no sobrescribir valores válidos)
    for key in ("family", "dataset_id", "model_name", "task_type", "input_level", "target_col", "data_plan", "data_source"):
        if _missing(champ.get(key)) and (ctx.get(key) is not None):
            champ[key] = ctx.get(key)

    # También backfill en el bloque `metrics` cuando aplica (útil para consumers legacy)
    for key in ("family", "task_type", "input_level", "target_col", "data_plan", "data_source"):
        if _missing(metrics.get(key)) and (ctx.get(key) is not None):
            metrics[key] = ctx.get(key)
    champ["metrics"] = metrics



    # 3) Validar explícitamente aquí para evitar response-validation 500 opaco
    try:
        return ChampionInfo(**champ)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Champion inválido para ChampionInfo: {e}")
