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

from typing import Any, Dict, Optional, Type

import os
import time
import uuid
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
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
)

from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida

from ...observability.bus_eventos import BUS

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
)


def _relpath(p: Path) -> str:
    """Devuelve una ruta relativa a BASE_DIR si es posible (si no, devuelve absoluta)."""
    try:
        return str(p.resolve().relative_to(BASE_DIR.resolve()))
    except Exception:
        return str(p.resolve())


def _strip_localfs(uri: str) -> str:
    """
    Convierte un URI estilo ``localfs://...`` a una ruta local.
    """
    if isinstance(uri, str) and uri.startswith("localfs://"):
        return uri.replace("localfs://", "", 1)
    return uri


def _abs_path(ref: str) -> Path:
    """
    Convierte un ref (relativo/absoluto o localfs://) a un Path absoluto bajo BASE_DIR.
    """
    raw = _strip_localfs(ref)
    p = Path(raw)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


# ---------------------------------------------------------------------------
# FIX A: Factory de estrategias (CLASES, no instancias)
# ---------------------------------------------------------------------------

_STRATEGY_CLASSES: Dict[str, Type[Any]] = {
    "rbm_general": RBMGeneral,
    "rbm_restringida": RBMRestringida,
}


def _create_strategy(modelo: str) -> Any:
    """
    Crea una instancia NUEVA de estrategia por job (FIX A).

    :param modelo: nombre lógico del modelo ("rbm_general" | "rbm_restringida")
    :raises HTTPException: si el modelo no está soportado
    """
    key = str(modelo or "").strip().lower()
    cls = _STRATEGY_CLASSES.get(key)
    if cls is None:
        raise HTTPException(status_code=400, detail=f"Modelo no soportado: {modelo}")
    return cls()


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
    """
    data_ref = getattr(req, "data_ref", None)
    if data_ref:
        return _strip_localfs(str(data_ref))

    ds = _dataset_id(req)
    if not ds:
        raise HTTPException(status_code=400, detail="Falta dataset_id/periodo_actual para resolver data_source.")

    data_source = str(getattr(req, "data_source", "feature_pack")).lower()

    if data_source == "feature_pack":
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


def _ensure_feature_pack(dataset_id: str, input_uri: str, *, force: bool = False) -> Dict[str, str]:
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
    out_dir = BASE_DIR / "artifacts" / "features" / dataset_id
    out = out_dir / "train_matrix.parquet"

    # Rutas esperadas (las devolvemos siempre, existan o no, para UI/debug).
    artifacts_rel: Dict[str, str] = {
        "train_matrix": _relpath(out),
        "teacher_index": _relpath(out_dir / "teacher_index.json"),
        "materia_index": _relpath(out_dir / "materia_index.json"),
        "bins": _relpath(out_dir / "bins.json"),
        "meta": _relpath(out_dir / "meta.json"),
    }

    if out.exists() and not force:
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

    return artifacts_rel


def _auto_prepare_if_needed(req: EntrenarRequest, data_ref: str) -> None:
    """Ejecuta preparación automática si `auto_prepare=True` y `data_ref` no existe.

    El objetivo es minimizar acciones manuales antes de entrenar:

    - Si ``data_source='unified_labeled'`` y falta el archivo, intenta construirlo.
    - Si ``data_source='feature_pack'`` y falta el feature-pack, intenta construirlo.

    Notas de diseño:

    - Para ``feature_pack`` preferimos usar como insumo ``data/processed/<dataset_id>.parquet``
      (salida de la pestaña **Data**) porque suele existir antes que el ``labeled``.
    - Para metodologías ``acumulado`` / ``ventana`` se usa ``historico/unificado_labeled.parquet``
      como fuente (si está disponible), porque ya contiene el histórico consolidado.

    :param req: Request de entrenamiento.
    :param data_ref: Ruta/URI principal resuelta según `data_source`.
    """
    auto_prepare = bool(getattr(req, "auto_prepare", False))
    if not auto_prepare:
        return

    p = _abs_path(data_ref)
    if p.exists():
        return

    ds = _dataset_id(req)
    data_source = str(getattr(req, "data_source", "feature_pack")).lower()
    metodologia = str(getattr(req, "metodologia", "periodo_actual")).lower()

    if data_source == "unified_labeled":
        _ensure_unified_labeled()
        return

    if data_source == "feature_pack":
        if not ds:
            raise HTTPException(status_code=400, detail="auto_prepare requiere dataset_id/periodo_actual.")

        # Caso histórico (acumulado / ventana)
        if metodologia in ("acumulado", "ventana"):
            _ensure_unified_labeled()
            input_uri = "historico/unificado_labeled.parquet"
        else:
            # Caso normal: intentamos con processed (Data tab) -> labeled -> datasets
            processed = BASE_DIR / "data" / "processed" / f"{ds}.parquet"
            if processed.exists():
                input_uri = _relpath(processed)
            else:
                try:
                    labeled_path = resolve_labeled_path(str(ds))
                    input_uri = _relpath(labeled_path)
                except Exception:
                    raw = BASE_DIR / "datasets" / f"{ds}.parquet"
                    if raw.exists():
                        input_uri = _relpath(raw)
                    else:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"No se encontró un dataset fuente para construir feature-pack de {ds}. "
                                "Opciones:\n"
                                "- Procesa/carga el dataset en la pestaña Data (debe generar data/processed/<ds>.parquet)\n"
                                "- O genera labeled (data/labeled/<ds>_beto.parquet)\n"
                                "- O asegúrate de tener datasets/<ds>.parquet"
                            ),
                        )

        _ensure_feature_pack(str(ds), input_uri=input_uri)
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


def _prepare_selected_data(req: EntrenarRequest, job_id: str) -> str:
    """
    Resuelve fuente de datos + auto_prepare + (si aplica) metodología.
    """
    data_ref = _resolve_by_data_source(req)
    _auto_prepare_if_needed(req, data_ref)

    data_source = str(getattr(req, "data_source", "feature_pack")).lower()

    if data_source == "feature_pack":
        pack_path = _abs_path(data_ref)
        if not pack_path.exists():
            raise HTTPException(
            status_code=404,
            detail=(
                f"Feature-pack no encontrado: {pack_path}. "
                "Activa auto_prepare=true al entrenar o llama a POST /modelos/feature-pack/prepare."
            ),
        )
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
    - Defaults consistentes con el flujo nuevo:
      - data_source=feature_pack
      - target_mode=sentiment_probs
      - split_mode=temporal
      - val_ratio=0.2
      - include_teacher_materia=True
      - teacher_materia_mode='embed' (si include_teacher_materia=True)
    - Los campos explícitos del request tienen prioridad sobre req.hparams.

    :param req: Request de entrenamiento.
    :param job_id: Correlation/job id.
    :return: Dict de hparams listo para pasar a PlantillaEntrenamiento.run().
    """
    hp = _normalize_hparams(getattr(req, "hparams", None))

    # Evitar que hparams contenga claves reservadas del request (p.ej. epochs)
    # ya que el número de épocas lo controla `req.epochs`.
    hp.pop("epochs", None)

    def put(key: str, value: Any) -> None:
        if value is None:
            return
        hp[key] = value

    put("job_id", job_id)

    # Defaults defensivos (si el request viene con None)
    data_source = getattr(req, "data_source", None) or "feature_pack"
    target_mode = getattr(req, "target_mode", None) or "sentiment_probs"
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

    put("data_source", data_source)
    put("target_mode", target_mode)
    put("split_mode", split_mode)
    put("val_ratio", val_ratio)
    put("include_teacher_materia", bool(include_tm))
    # Importante: SOLO poner teacher_materia_mode si no es None
    put("teacher_materia_mode", tm_mode)

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
    """Verifica existencia de artefactos mínimos para entrenar un dataset_id."""
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
    feat_ok = _abs_path(feat_ref).exists()

    return ReadinessResponse(
        dataset_id=dataset_id,
        labeled_exists=bool(labeled_ok),
        unified_labeled_exists=bool(unified_ok),
        feature_pack_exists=bool(feat_ok),
        paths={"labeled": labeled_ref, "unified_labeled": unified_ref, "feature_pack": feat_ref},
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


# ---------------------------------------------------------------------------
# Entrenamiento (persistencia vía runs_io)
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: EntrenarRequest) -> None:
    """Ejecuta el entrenamiento en background y persiste runs/champion si aplica."""
    t0 = time.perf_counter()
    try:
        # FIX A: instancia nueva SIEMPRE
        estrategia = _create_strategy(req.modelo)
        # FIX B defensivo: reset si existe
        _safe_reset_strategy(estrategia)

        tpl = PlantillaEntrenamiento(estrategia)

        _wire_job_observers(job_id)

        selected_ref = _prepare_selected_data(req, job_id)

        # hparams robusto (sin None + defaults)
        run_hparams = _build_run_hparams(req, job_id)

        out = tpl.run(
            selected_ref,
            req.epochs,
            run_hparams,
            model_name=req.modelo,
        )

        # estado base: si no existía, al menos deja job_id + contenedores
        st = _ESTADOS.get(job_id, {})
        if not isinstance(st, dict):
            st = {}
        st.setdefault("job_id", job_id)
        st.setdefault("metrics", {})
        st.setdefault("history", [])

        # mete status/metrics/history que devolvió tpl.run(...)
        if isinstance(out, dict):
            st.update(out)

        if out.get("status") == "completed":
            # 1) Calcular métricas REALES (accuracy/f1/confusion) sobre el mismo dataset
            #    y con el mismo filtrado del modelo (_prepare_xy).
            try:
                df_eval = pd.read_parquet(str(selected_ref))
                eval_metrics = _evaluate_post_training_metrics(estrategia, df_eval, run_hparams)

                merged = dict(out.get("metrics") or {})
                merged.update(eval_metrics)

                out["metrics"] = merged
                st["metrics"] = merged
            except Exception as e_eval:
                # No tumbar el run por evaluación: deja rastro para debug
                merged = dict(out.get("metrics") or {})
                merged["eval_error"] = str(e_eval)
                out["metrics"] = merged
                st["metrics"] = merged

            # 2) Persistir run (CON métricas reales ya mergeadas)
            ds = _dataset_id(req) or "unknown"
            run_id = build_run_id(dataset_id=str(ds), model_name=str(req.modelo), job_id=job_id)

            req_snapshot = (req.model_dump() if hasattr(req, "model_dump") else req.dict())
            req_snapshot.update({"job_id": job_id, "selected_ref": str(selected_ref), "base_dir": str(BASE_DIR)})

            run_dir = save_run(
                run_id=run_id,
                job_id=job_id,
                dataset_id=str(ds),
                model_name=str(req.modelo),
                data_ref=str(selected_ref),
                params={"req": req_snapshot},
                final_metrics=out.get("metrics") or {},
                history=out.get("history") or [],
            )

            # 3) Guardar artefactos del modelo en el run_dir (rbm.pt, head.pt, vectorizer.json, etc.)
            if hasattr(estrategia, "save") and callable(getattr(estrategia, "save")):
                estrategia.save(str(run_dir))

            # 4) Actualizar champion usando métricas reales
            upd = maybe_update_champion(
                dataset_id=str(ds),
                model_name=str(req.modelo),
                metrics=out.get("metrics") or {},
                source_run_id=run_id,
            )

            st["run_id"] = run_id
            st["artifact_path"] = str(run_dir)
            st["champion_promoted"] = bool(upd.get("promoted"))
            st["time_total_ms"] = float((time.perf_counter() - t0) * 1000.0)

        _ESTADOS[job_id] = st

    except Exception as e:
        st = _ESTADOS.get(job_id) or {"job_id": job_id, "metrics": {}, "history": []}
        if not isinstance(st, dict):
            st = {"job_id": job_id, "metrics": {}, "history": []}
        st["status"] = "failed"
        st["error"] = str(e)
        st["time_total_ms"] = float((time.perf_counter() - t0) * 1000.0)
        st.setdefault("progress", 0.0)
        _ESTADOS[job_id] = st



@router.post(
    "/feature-pack/prepare",
    summary="Construye el feature-pack para un dataset",
)
def prepare_feature_pack_endpoint(
    dataset_id: str,
    input_uri: Optional[str] = None,
    force: bool = False,
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
        # Resolver automáticamente el origen.
        candidates = []
        candidates.append(BASE_DIR / "data" / "processed" / f"{ds}.parquet")
        try:
            labeled = resolve_labeled_path(ds)
            candidates.append(labeled)
        except Exception:
            pass
        candidates.append(BASE_DIR / "datasets" / f"{ds}.parquet")

        src_ref = None
        for c in candidates:
            if c.exists():
                src_ref = _relpath(c)
                break

        if not src_ref:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No se pudo resolver input_uri para {ds}. "
                    "Primero procesa el dataset (data/processed), o genera labeled, o coloca el parquet en datasets/."
                ),
            )

    return _ensure_feature_pack(ds, input_uri=src_ref, force=bool(force))


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

    # IMPORTANTE (Item 1):
    # - No permitir que hp_norm_ui contenga 'epochs' (u otros reservados) que pisen req.epochs.
    # - Colocar epochs AL FINAL del dict params para que siempre sea el valor del request.
    params_ui: Dict[str, Any] = {
        **hp_norm_ui,
        "dataset_id": _dataset_id(req),
        "periodo_actual": getattr(req, "periodo_actual", None),
        "metodologia": getattr(req, "metodologia", "periodo_actual"),
        "ventana_n": getattr(req, "ventana_n", None),
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
    try:
        req_norm = req.model_copy(update={"hparams": hp_norm_raw})
    except AttributeError:
        req_norm = req.copy(update={"hparams": hp_norm_raw})

    bg.add_task(_run_training, job_id, req_norm)
    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str) -> EstadoResponse:
    """Devuelve el estado actual de un job."""
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}, "history": [], "progress": 0.0}
    return EstadoResponse(**st)


@router.post(
    "/champion/promote",
    response_model=ChampionInfo,
    summary="Promueve un run existente a champion (manual)",
)
def promote_champion(req: PromoteChampionRequest) -> ChampionInfo:
    """Promueve manualmente un run existente a champion usando runs_io."""
    try:
        champ = promote_run_to_champion(
            dataset_id=req.dataset_id,
            run_id=req.run_id,
            model_name=req.model_name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo promover champion: {e}") from e

    return ChampionInfo(**champ)


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
) -> list[RunSummary]:
    """Devuelve un resumen de runs encontrados en artifacts/runs."""
    ds = dataset_id or dataset or periodo
    runs = list_runs(model_name=model_name, dataset_id=ds)
    return [RunSummary(**r) for r in runs]


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
    return RunDetails(**details)


@router.get(
    "/champion",
    response_model=ChampionInfo,
    summary="Devuelve info del modelo campeón actual (por dataset o legacy)",
)
def get_champion(
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
) -> ChampionInfo:
    """Devuelve el campeón actual."""
    ds = dataset_id or dataset or periodo

    if ds:
        champ = load_dataset_champion(ds)
        if champ and (not model_name or champ.get("model_name") == model_name):
            return ChampionInfo(**champ)

    champ = load_current_champion(model_name=model_name, dataset_id=ds)
    if not champ:
        raise HTTPException(status_code=404, detail="No hay campeón registrado")
    return ChampionInfo(**champ)
