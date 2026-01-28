"""
neurocampus.app.routers.modelos
================================

Router de **Modelos** (FastAPI).

Responsabilidades principales
----------------------------
- Lanzar entrenamientos en background (``POST /modelos/entrenar``).
- Reportar progreso/estado (``GET /modelos/estado/{job_id}``).
- Exponer auditoría de entrenamientos (runs) y champion (en este archivo ya existe).
- Exponer readiness y resolver insumos del entrenamiento:
  - ``GET /modelos/readiness?dataset_id=...``
  - Resolver ``data_source``:
    - ``feature_pack`` (recomendado): ``artifacts/features/<dataset_id>/train_matrix.parquet``
    - ``labeled`` (fallback): ``data/labeled/<dataset_id>_beto.parquet`` (o *_teacher.parquet)
    - ``unified_labeled`` (para acumulado/ventana): ``historico/unificado_labeled.parquet``
  - ``auto_prepare``: si faltan artifacts, intentar generarlos
    - unificación labeled (``historico/unificado_labeled.parquet``)
    - feature-pack (``artifacts/features/<dataset_id>/train_matrix.parquet``)

.. important::
   Este archivo intenta ser robusto a diferentes CWDs (por ejemplo, cuando se ejecuta
   uvicorn desde ``backend/``). Por eso se detecta ``BASE_DIR`` (raíz del repo).

Notas
-----
- La persistencia de runs/champion (commit 3) se implementa en otro cambio.
- Este commit se enfoca en readiness + resolver de data_source + auto_prepare
  (sin tocar DBM ni Predicciones).
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import uuid
import pandas as pd
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas.modelos import (
    EntrenarRequest,
    EntrenarResponse,
    EstadoResponse,
    ReadinessResponse,   # <-- Commit 2: readiness response
    RunSummary,
    RunDetails,
    ChampionInfo,
)

from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida

from ...observability.bus_eventos import BUS  # capturamos eventos training.* (history/metrics)

# Selección de datos por metodología (periodo_actual / acumulado / ventana)
try:
    from ...models.strategies.metodologia import SeleccionConfig, resolver_metodologia
except Exception:
    # Shim defensivo (no debería ocurrir en el repo actual)
    class SeleccionConfig:  # type: ignore
        def __init__(self, periodo_actual=None, ventana_n=4): ...
    def resolver_metodologia(nombre: str):  # type: ignore
        raise RuntimeError(
            "El módulo de metodologías no está disponible. "
            "Asegúrate de crear neurocampus/models/strategies/metodologia.py"
        )

# Runs/champion (ya existente en el archivo base)
from ...utils.runs_io import list_runs, load_run_details, load_current_champion, load_dataset_champion

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
    # Fallback: layout típico del proyecto (defensivo)
    return here.parents[5]


BASE_DIR: Path = _find_project_root()


def _relpath(p: Path) -> str:
    """Devuelve una ruta relativa a BASE_DIR si es posible (si no, devuelve absoluta)."""
    try:
        return str(p.resolve().relative_to(BASE_DIR.resolve()))
    except Exception:
        return str(p.resolve())


def _strip_localfs(uri: str) -> str:
    """
    Convierte un URI estilo ``localfs://...`` a una ruta local.

    Ejemplo:
        - ``localfs:///tmp/x.parquet`` -> ``/tmp/x.parquet``
        - ``localfs://data/x.parquet`` -> ``data/x.parquet``
    """
    if isinstance(uri, str) and uri.startswith("localfs://"):
        return uri.replace("localfs://", "", 1)
    return uri


def _abs_path(ref: str) -> Path:
    """
    Convierte un ref (relativo/absoluto o localfs://) a un Path absoluto bajo BASE_DIR.

    :param ref: Ruta relativa/absoluta o localfs://...
    :return: Path absoluto.
    """
    raw = _strip_localfs(ref)
    p = Path(raw)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


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


def _flatten_metrics_from_payload(payload: Dict[str, Any], allow_loss: bool = True) -> Dict[str, float]:
    """
    Aplana métricas numéricas desde un payload de evento.

    Se usa cuando el evento no trae un dict `metrics` explícito.
    Filtra campos de control y conserva solo numéricos.

    :param payload: payload del evento.
    :param allow_loss: incluir `loss` si está presente.
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

    Eventos:
    - training.started: inicializa metadatos (modelo/params)
    - training.epoch_end: agrega un punto a history[] y actualiza metrics
    - training.completed: marca estado y métricas finales
    - training.failed: marca error

    Idempotente: evita resuscribir si ya fue wired (útil con --reload).
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
        st["status"] = "running"
        st["model"] = evt.payload.get("model", st.get("model"))
        params_evt = evt.payload.get("params")
        if isinstance(params_evt, dict):
            st["params"] = _normalize_hparams(params_evt) or st.get("params", {})

    def _on_epoch_end(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st.setdefault("history", [])

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

    def _on_failed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st["status"] = "failed"
        st["error"] = evt.payload.get("error", "unknown error")

    BUS.subscribe("training.started", _on_started)
    BUS.subscribe("training.epoch_end", _on_epoch_end)
    BUS.subscribe("training.completed", _on_completed)
    BUS.subscribe("training.failed", _on_failed)

    _OBS_WIRED_JOBS.add(job_id)


# ---------------------------------------------------------------------------
# Commit 2: readiness + resolver data_source + auto_prepare
# ---------------------------------------------------------------------------

def _dataset_id(req: EntrenarRequest) -> Optional[str]:
    """
    Obtiene dataset_id/periodo desde el request.

    En el schema actualizado, ``dataset_id`` y ``periodo_actual`` se sincronizan.
    Aquí usamos getattr para no romper compatibilidad si el schema aún está migrando.
    """
    return getattr(req, "dataset_id", None) or getattr(req, "periodo_actual", None)


def _resolve_by_data_source(req: EntrenarRequest) -> str:
    """
    Resuelve el input principal del entrenamiento según `data_source`.

    Prioridad:
    1) `data_ref` (override manual / legacy)
    2) `data_source`:
       - feature_pack: artifacts/features/<dataset_id>/train_matrix.parquet
       - unified_labeled: historico/unificado_labeled.parquet (fallback a historico/unificado.parquet)
       - labeled: data/labeled/<dataset_id>_beto.parquet|_teacher.parquet (heurística)

    :raises HTTPException: si falta dataset_id o no se encuentra el recurso requerido.
    """
    data_ref = getattr(req, "data_ref", None)
    if data_ref:
        return _strip_localfs(str(data_ref))

    ds = _dataset_id(req)
    if not ds:
        raise HTTPException(status_code=400, detail="Falta dataset_id/periodo_actual para resolver data_source.")

    data_source = str(getattr(req, "data_source", "labeled")).lower()

    if data_source == "feature_pack":
        return f"artifacts/features/{ds}/train_matrix.parquet"

    if data_source == "unified_labeled":
        # preferido
        preferred = BASE_DIR / "historico" / "unificado_labeled.parquet"
        if preferred.exists():
            return "historico/unificado_labeled.parquet"
        # legacy (defensivo)
        legacy = BASE_DIR / "historico" / "unificado.parquet"
        if legacy.exists():
            return "historico/unificado.parquet"
        # si no existe, dejamos que auto_prepare lo genere si está activo
        return "historico/unificado_labeled.parquet"

    # labeled (fallback)
    try:
        p = resolve_labeled_path(str(ds))
        return _relpath(p)
    except Exception:
        # fallback mínimo
        return f"data/labeled/{ds}_beto.parquet"


def _ensure_unified_labeled() -> None:
    """
    Asegura la existencia de `historico/unificado_labeled.parquet`.

    Implementación:
    - Ejecuta `UnificacionStrategy.acumulado_labeled()` en forma síncrona.

    .. note::
       Esto se usa como parte de `auto_prepare`. Si tu operación prefiere jobs,
       puedes cambiarlo a lanzar `/jobs/data/unify/run` y esperar su finalización.
    """
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
                "Ejecuta manualmente el job de Datos: POST /jobs/data/unify/run (mode=acumulado_labeled)."
            ),
        ) from e

    strat = UnificacionStrategy(base_uri=f"localfs://{BASE_DIR.as_posix()}")
    # Genera historico/unificado_labeled.parquet
    strat.acumulado_labeled()


def _ensure_feature_pack(dataset_id: str, input_uri: str) -> None:
    """
    Asegura `artifacts/features/<dataset_id>/train_matrix.parquet`.

    :param dataset_id: id lógico del dataset (normalmente periodo).
    :param input_uri: fuente etiquetada (labeled o unificado_labeled).
    """
    out = BASE_DIR / "artifacts" / "features" / dataset_id / "train_matrix.parquet"
    if out.exists():
        return

    try:
        from neurocampus.data.features_prepare import prepare_feature_pack
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo importar prepare_feature_pack para auto_prepare. "
                "Ejecuta manualmente el job de Datos: POST /jobs/data/features/prepare/run."
            ),
        ) from e

    output_dir = str((BASE_DIR / "artifacts" / "features" / dataset_id).resolve())
    prepare_feature_pack(
        base_dir=BASE_DIR,
        dataset_id=dataset_id,
        input_uri=input_uri,
        output_dir=output_dir,
    )


def _auto_prepare_if_needed(req: EntrenarRequest, data_ref: str) -> None:
    """
    Ejecuta preparación automática de artefactos si `auto_prepare=True` y `data_ref` no existe.

    Lógica:
    - Si `data_source=unified_labeled`: asegura `historico/unificado_labeled.parquet`.
    - Si `data_source=feature_pack`: asegura `historico/unificado_labeled` (si metodología acumulado/ventana)
      y luego crea `artifacts/features/<dataset_id>/train_matrix.parquet`.
    - Si `data_source=labeled`: no se intenta auto_prepare (requeriría BETO/PLN).

    :param req: request de entrenamiento.
    :param data_ref: referencia resuelta (ruta relativa o absoluta).
    """
    auto_prepare = bool(getattr(req, "auto_prepare", False))
    if not auto_prepare:
        return

    p = _abs_path(data_ref)
    if p.exists():
        return

    ds = _dataset_id(req)
    data_source = str(getattr(req, "data_source", "labeled")).lower()
    metodologia = str(getattr(req, "metodologia", "periodo_actual")).lower()

    if data_source == "unified_labeled":
        _ensure_unified_labeled()
        return

    if data_source == "feature_pack":
        if not ds:
            raise HTTPException(status_code=400, detail="auto_prepare requiere dataset_id/periodo_actual.")

        # Si se entrena acumulado/ventana, el input típico del feature-pack es unificado_labeled
        if metodologia in ("acumulado", "ventana"):
            _ensure_unified_labeled()
            input_uri = "historico/unificado_labeled.parquet"
        else:
            # periodo_actual -> feature-pack desde labeled del periodo
            try:
                labeled_path = resolve_labeled_path(str(ds))
                input_uri = _relpath(labeled_path)
            except Exception:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"No existe labeled para {ds}. "
                        "Primero corre el pipeline de Datos (BETO) para generar data/labeled/<dataset>_beto.parquet."
                    ),
                )

        _ensure_feature_pack(str(ds), input_uri=input_uri)
        return

    # labeled: no se auto prepara aquí (requiere PLN/BETO)
    raise HTTPException(
        status_code=409,
        detail=(
            "data_source='labeled' no puede auto-prepararse desde Modelos. "
            "Genera primero data/labeled/<dataset>_beto.parquet desde la pestaña Datos."
        ),
    )


def _read_dataframe_any(path_or_uri: str) -> pd.DataFrame:
    """
    Lee dataset desde una ruta (parquet/csv) local.

    :param path_or_uri: ruta relativa/absoluta (o `localfs://...`).
    :raises HTTPException: si no se puede leer.
    """
    p = _abs_path(path_or_uri)
    try:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        # fallback: intentar parquet
        return pd.read_parquet(p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el dataset ({p.name}): {e}") from e


def _prepare_selected_data(req: EntrenarRequest, job_id: str) -> str:
    """
    Resuelve fuente de datos (data_source/data_ref) + auto_prepare + metodología
    y materializa un parquet temporal para el entrenamiento.

    Flujo:
    1) data_ref := resolver por data_source (o override)
    2) si auto_prepare y no existe -> generar artifacts necesarios
    3) leer DataFrame
    4) aplicar selección por metodología (periodo_actual/acumulado/ventana)
    5) guardar en `data/.tmp/df_sel_<job_id>.parquet`

    :return: ruta (string) al parquet temporal.
    """
    data_ref = _resolve_by_data_source(req)
    _auto_prepare_if_needed(req, data_ref)

    df = _read_dataframe_any(data_ref)

    metodologia = getattr(req, "metodologia", None) or "periodo_actual"
    periodo_actual = getattr(req, "periodo_actual", None) or _dataset_id(req)
    ventana_n = getattr(req, "ventana_n", None) or 4

    metodo = resolver_metodologia(str(metodologia).lower())
    cfg = SeleccionConfig(periodo_actual=str(periodo_actual) if periodo_actual else None, ventana_n=int(ventana_n))

    df_sel = metodo.seleccionar(df, cfg)
    if df_sel.empty:
        raise HTTPException(status_code=400, detail="Selección de datos vacía según la metodología/periodo.")

    tmp_dir = (BASE_DIR / "data" / ".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref = tmp_dir / f"df_sel_{job_id}.parquet"

    try:
        df_sel.to_parquet(tmp_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo materializar el subconjunto: {e}") from e

    return _relpath(tmp_ref)


# ---------------------------------------------------------------------------
# Endpoints (Commit 2): readiness
# ---------------------------------------------------------------------------

@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Verifica insumos para entrenar (labeled / unified_labeled / feature_pack)",
)
def readiness(dataset_id: str) -> ReadinessResponse:
    """
    Verifica si existen los artefactos mínimos para entrenar un dataset_id.

    Chequeos:
    - labeled: data/labeled/<dataset_id>_beto.parquet (o *_teacher.parquet)
    - unified_labeled: historico/unificado_labeled.parquet
    - feature_pack: artifacts/features/<dataset_id>/train_matrix.parquet

    :param dataset_id: periodo o id lógico del dataset.
    """
    # labeled (heurística)
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
        paths={
            "labeled": labeled_ref,
            "unified_labeled": unified_ref,
            "feature_pack": feat_ref,
        },
    )


# ---------------------------------------------------------------------------
# Entrenamiento (usa auto_prepare + data_source)
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: EntrenarRequest) -> None:
    """
    Ejecuta el entrenamiento en background.

    IMPORTANTE:
    - Cualquier excepción debe marcar el job como 'failed' y registrar 'error'
      para que el frontend detenga el polling.
    """
    try:
        estrategia = RBMGeneral() if req.modelo == "rbm_general" else RBMRestringida()
        tpl = PlantillaEntrenamiento(estrategia)

        _wire_job_observers(job_id)

        selected_ref = _prepare_selected_data(req, job_id)

        out = tpl.run(
            selected_ref,
            req.epochs,
            {**(_normalize_hparams(req.hparams)), "job_id": job_id,
             # pasar a estrategia (si la usa) hints de entrenamiento
             "data_source": getattr(req, "data_source", None),
             "target_mode": getattr(req, "target_mode", None),
             "split_mode": getattr(req, "split_mode", None),
             "val_ratio": getattr(req, "val_ratio", None),
             "include_teacher_materia": getattr(req, "include_teacher_materia", None),
             },
            model_name=req.modelo,
        )

        st = _ESTADOS.get(job_id, {})
        st.update(out)
        _ESTADOS[job_id] = st

    except Exception as e:
        st = _ESTADOS.get(job_id) or {"job_id": job_id, "metrics": {}, "history": []}
        st["status"] = "failed"
        st["error"] = str(e)
        _ESTADOS[job_id] = st


@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks) -> EntrenarResponse:
    """
    Lanza un entrenamiento en background y retorna `job_id`.

    Este endpoint inicializa el estado para que la UI pueda hacer polling en
    ``GET /modelos/estado/{job_id}``.
    """
    job_id = str(uuid.uuid4())

    hp_norm = _normalize_hparams(req.hparams)

    # Para trazabilidad: resolvemos el data_ref "base" (antes de selección)
    base_ref = _resolve_by_data_source(req)

    _ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "metrics": {},
        "history": [],
        "model": req.modelo,
        "params": {
            "epochs": req.epochs,
            **hp_norm,
            "dataset_id": _dataset_id(req),
            "periodo_actual": getattr(req, "periodo_actual", None),
            "metodologia": getattr(req, "metodologia", "periodo_actual"),
            "ventana_n": getattr(req, "ventana_n", None),
            "data_source": getattr(req, "data_source", "feature_pack"),
            "target_mode": getattr(req, "target_mode", "sentiment_probs"),
            "split_mode": getattr(req, "split_mode", "temporal"),
            "val_ratio": getattr(req, "val_ratio", 0.2),
            "include_teacher_materia": getattr(req, "include_teacher_materia", True),
            "auto_prepare": getattr(req, "auto_prepare", True),
            "data_ref": getattr(req, "data_ref", None) or base_ref,
        },
        "error": None,
    }

    # Asegurar que hparams viajen normalizados
    try:
        req_norm = req.model_copy(update={"hparams": hp_norm})  # Pydantic v2
    except AttributeError:
        req_norm = req.copy(update={"hparams": hp_norm})        # Pydantic v1

    bg.add_task(_run_training, job_id, req_norm)

    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str) -> EstadoResponse:
    """
    Devuelve el estado actual de un job.

    Si no existe, responde con status ``unknown`` (compatibilidad).
    """
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}, "history": []}
    return EstadoResponse(**st)


# ---------------------------------------------------------------------------
# Runs / Champion (ya existentes en tu archivo base)
# ---------------------------------------------------------------------------

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
    """
    Devuelve un resumen de los runs encontrados en artifacts/runs.

    Filtros:
      - model_name: ej 'rbm_general'
      - dataset_id / dataset / periodo: filtra por dataset asociado al run
    """
    ds = dataset_id or dataset or periodo
    runs = list_runs(model_name=model_name, dataset_id=ds)
    return [RunSummary(**r) for r in runs]


@router.get(
    "/runs/{run_id}",
    response_model=RunDetails,
    summary="Detalles completos de un run (incluye config si existe)",
)
def get_run_details(run_id: str) -> RunDetails:
    """
    Devuelve detalles completos de un run leyendo:
      - artifacts/runs/<run_id>/metrics.json
      - artifacts/runs/<run_id>/config.snapshot.yaml o config.yaml (si existe)
    """
    details = load_run_details(run_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Run {run_id} no encontrado")
    return RunDetails(**details)


@router.get(
    "/champion",
    response_model=ChampionInfo,
    summary="Devuelve info del modelo campeón actual (para dashboard/predicciones)",
)
def get_champion(
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_id: Optional[str] = None,
    periodo: Optional[str] = None,
) -> ChampionInfo:
    """
    Devuelve el modelo campeón (champion) actual.

    Soporta:
      - Champions por dataset:
        artifacts/champions/<dataset_id>/<model_name>/metrics.json
      - Champions legacy por modelo:
        artifacts/champions/<model_name>/metrics.json

    Parámetros:
      - model_name (opcional): filtra por tipo de modelo
      - dataset_id / dataset / periodo (opcional): selecciona champion del dataset
    """
    ds = dataset_id or dataset or periodo

    if ds and not model_name:
        champ = load_dataset_champion(ds)
    else:
        champ = load_current_champion(model_name=model_name, dataset_id=ds)

    if not champ:
        raise HTTPException(status_code=404, detail="No hay campeón registrado")
    return ChampionInfo(**champ)
