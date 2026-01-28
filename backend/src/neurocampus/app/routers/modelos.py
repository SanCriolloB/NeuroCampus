"""
neurocampus.app.routers.modelos
================================

Router de **Modelos** (FastAPI) para NeuroCampus.

Este módulo expone endpoints de entrenamiento, estado de jobs, auditoría (runs)
y champion. Está diseñado para ser compatible con el flujo reproducible del sistema:

- Datos produce *feature-packs* en ``artifacts/features/<dataset_id>/train_matrix.parquet``.
- Modelos entrena desde esos artefactos, persiste runs en ``artifacts/runs/<run_id>/``
  y mantiene un champion en ``artifacts/champions/<dataset_id>/...``.

--------
- ``GET /modelos/readiness?dataset_id=...`` para detectar insumos disponibles.
- Resolver de ``data_source`` (feature_pack/labeled/unified_labeled).
- ``auto_prepare``: intentar generar artefactos faltantes (unificado labeled / feature-pack).
--------
- Persistencia real de runs (run_id + metrics/history/config/job_meta + pesos si la estrategia los guarda).
- Champion auto-update por score mediante helpers del módulo :mod:`neurocampus.utils.runs_io`.
- Endpoint manual para promover un run a champion: ``POST /modelos/champion/promote``.

.. note::
   - Este router maneja jobs en memoria (``_ESTADOS``) y expone polling vía ``/estado/{job_id}``.
   - La persistencia (runs/champion) se delega a ``runs_io.py`` para evitar duplicación.

.. warning::
   Este router asume que el pipeline de Datos ya existe (DataTab) y genera los artefactos
   necesarios. Si ``data_source='labeled'`` falta, no se auto-construye BETO desde aquí.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import os
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException

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

        # Solo numéricos al history (para graficación)
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

    :raises HTTPException: si falta dataset_id.
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
        preferred = BASE_DIR / "historico" / "unificado_labeled.parquet"
        if preferred.exists():
            return "historico/unificado_labeled.parquet"
        legacy = BASE_DIR / "historico" / "unificado.parquet"
        if legacy.exists():
            return "historico/unificado.parquet"
        return "historico/unificado_labeled.parquet"

    # labeled (fallback)
    try:
        p = resolve_labeled_path(str(ds))
        return _relpath(p)
    except Exception:
        return f"data/labeled/{ds}_beto.parquet"


def _ensure_unified_labeled() -> None:
    """
    Asegura la existencia de `historico/unificado_labeled.parquet`.

    Implementación:
    - Ejecuta `UnificacionStrategy.acumulado_labeled()` en forma síncrona.

    Si en tu operación prefieres jobs, puedes cambiar esto por una llamada a
    ``POST /jobs/data/unify/run`` y esperar su finalización.
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
                "Ejecuta manualmente el job: POST /jobs/data/unify/run (mode=acumulado_labeled)."
            ),
        ) from e

    strat = UnificacionStrategy(base_uri=f"localfs://{BASE_DIR.as_posix()}")
    strat.acumulado_labeled()


def _ensure_feature_pack(dataset_id: str, input_uri: str) -> None:
    """
    Asegura `artifacts/features/<dataset_id>/train_matrix.parquet`.

    :param dataset_id: id lógico del dataset.
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
                "Ejecuta manualmente el job: POST /jobs/data/features/prepare/run."
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
    Ejecuta preparación automática si ``auto_prepare=True`` y ``data_ref`` no existe.

    - unified_labeled: crea historico/unificado_labeled.parquet
    - feature_pack:
      - acumulado/ventana: requiere unificado_labeled + crea artifacts/features/<ds>/train_matrix.parquet
      - periodo_actual: intenta crear feature-pack desde labeled del periodo
    - labeled: NO se auto-prepara aquí (requiere BETO/PLN de DataTab).

    :raises HTTPException: si no hay insumos para preparar.
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

        if metodologia in ("acumulado", "ventana"):
            _ensure_unified_labeled()
            input_uri = "historico/unificado_labeled.parquet"
        else:
            try:
                labeled_path = resolve_labeled_path(str(ds))
                input_uri = _relpath(labeled_path)
            except Exception:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"No existe labeled para {ds}. "
                        "Primero corre el pipeline de Datos para generar data/labeled/<dataset>_beto.parquet."
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
    """
    Lee dataset desde ruta local (parquet/csv).

    :param path_or_uri: ruta relativa/absoluta o localfs://...
    """
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
    Resuelve fuente de datos + auto_prepare + metodología y materializa un parquet temporal.

    Flujo:
    1) data_ref := resolver por data_source (o override)
    2) auto_prepare si falta (unificado/feature-pack)
    3) leer DF
    4) aplicar metodología (periodo_actual/acumulado/ventana)
    5) escribir ``data/.tmp/df_sel_<job_id>.parquet``

    :return: ruta relativa del parquet temporal.
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
        raise HTTPException(status_code=400, detail="Selección de datos vacía según metodología/periodo.")

    tmp_dir = (BASE_DIR / "data" / ".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref = tmp_dir / f"df_sel_{job_id}.parquet"

    try:
        df_sel.to_parquet(tmp_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo materializar el subconjunto: {e}") from e

    return _relpath(tmp_ref)


# ---------------------------------------------------------------------------
# Endpoints: readiness (Commit 2)
# ---------------------------------------------------------------------------

@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Verifica insumos para entrenar (labeled / unified_labeled / feature_pack)",
)
def readiness(dataset_id: str) -> ReadinessResponse:
    """
    Verifica existencia de artefactos mínimos para entrenar un ``dataset_id``.

    Chequeos:
    - labeled: ``data/labeled/<dataset_id>_beto.parquet`` (o *_teacher.parquet)
    - unified_labeled: ``historico/unificado_labeled.parquet``
    - feature_pack: ``artifacts/features/<dataset_id>/train_matrix.parquet``
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
        paths={"labeled": labeled_ref, "unified_labeled": unified_ref, "feature_pack": feat_ref},
    )


# ---------------------------------------------------------------------------
# Entrenamiento (Commit 3: persistencia vía runs_io)
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: EntrenarRequest) -> None:
    """
    Ejecuta el entrenamiento en background.

    Commit 3:
    - Si finaliza exitosamente:
      - Persiste un run usando :func:`neurocampus.utils.runs_io.save_run`.
      - Decide/actualiza champion usando :func:`neurocampus.utils.runs_io.maybe_update_champion`.

    Cualquier excepción marca el job como ``failed``.
    """
    t0 = time.perf_counter()
    try:
        estrategia = RBMGeneral() if req.modelo == "rbm_general" else RBMRestringida()
        tpl = PlantillaEntrenamiento(estrategia)

        _wire_job_observers(job_id)

        # 1) materializar subset (según metodología)
        selected_ref = _prepare_selected_data(req, job_id)

        # 2) ejecutar entrenamiento
        out = tpl.run(
            selected_ref,
            req.epochs,
            {
                **(_normalize_hparams(req.hparams)),
                "job_id": job_id,
                # hints (si la estrategia decide usarlos)
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

        # 3) persistir run + champion (solo si completed)
        if out.get("status") == "completed":
            ds = _dataset_id(req) or "unknown"
            run_id = build_run_id(dataset_id=str(ds), model_name=str(req.modelo), job_id=job_id)

            # snapshot reproducible del request
            req_snapshot = (req.model_dump() if hasattr(req, "model_dump") else req.dict())
            req_snapshot.update(
                {
                    "job_id": job_id,
                    "selected_ref": selected_ref,
                    "base_dir": str(BASE_DIR),
                }
            )

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

            # Guardar pesos si la estrategia lo implementa (mejor esfuerzo)
            try:
                if hasattr(estrategia, "save") and callable(getattr(estrategia, "save")):
                    estrategia.save(str(run_dir))
            except Exception:
                # no hacemos fallar el job por IO de pesos
                pass

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
        st["status"] = "failed"
        st["error"] = str(e)
        st["time_total_ms"] = float((time.perf_counter() - t0) * 1000.0)
        _ESTADOS[job_id] = st


@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks) -> EntrenarResponse:
    """
    Lanza un entrenamiento en background y retorna ``job_id``.

    Inicializa el estado en ``_ESTADOS`` para que la UI haga polling a
    ``GET /modelos/estado/{job_id}``.
    """
    job_id = str(uuid.uuid4())

    hp_norm = _normalize_hparams(req.hparams)
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
        "run_id": None,
        "artifact_path": None,
        "champion_promoted": False,
    }

    # Asegurar que hparams viajen normalizados (Pydantic v2/v1)
    try:
        req_norm = req.model_copy(update={"hparams": hp_norm})
    except AttributeError:
        req_norm = req.copy(update={"hparams": hp_norm})

    bg.add_task(_run_training, job_id, req_norm)
    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str) -> EstadoResponse:
    """
    Devuelve el estado actual de un job.

    Puede incluir ``run_id`` y ``artifact_path`` si el entrenamiento terminó exitosamente.
    """
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}, "history": []}
    return EstadoResponse(**st)


# ---------------------------------------------------------------------------
# Commit 3: promote manual
# ---------------------------------------------------------------------------

@router.post(
    "/champion/promote",
    response_model=ChampionInfo,
    summary="Promueve un run existente a champion (manual)",
)
def promote_champion(req: PromoteChampionRequest) -> ChampionInfo:
    """
    Promueve manualmente un run existente a champion usando runs_io.

    Internamente delega a :func:`neurocampus.utils.runs_io.promote_run_to_champion`.

    :raises HTTPException: si el run no existe o hay error de IO.
    """
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


# ---------------------------------------------------------------------------
# Runs / Champion (auditoría)
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
    Devuelve un resumen de runs encontrados en ``artifacts/runs``.

    Filtros:
      - ``model_name``: ej. ``rbm_general``
      - ``dataset_id`` / ``dataset`` / ``periodo``: dataset asociado
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
    Devuelve detalles completos de un run leyendo artifacts del filesystem.

    Delegación a :func:`neurocampus.utils.runs_io.load_run_details`.
    """
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
    """
    Devuelve el campeón actual.

    Prioridad:
    1) Si se pasa ``dataset_id`` (o alias), intenta champion por dataset con
       :func:`neurocampus.utils.runs_io.load_dataset_champion`.
    2) Fallback/legacy con :func:`neurocampus.utils.runs_io.load_current_champion`.

    :raises HTTPException: si no hay champion.
    """
    ds = dataset_id or dataset or periodo

    if ds:
        champ = load_dataset_champion(ds)
        if champ and (not model_name or champ.get("model_name") == model_name):
            return ChampionInfo(**champ)

    champ = load_current_champion(model_name=model_name, dataset_id=ds)
    if not champ:
        raise HTTPException(status_code=404, detail="No hay campeón registrado")
    return ChampionInfo(**champ)
