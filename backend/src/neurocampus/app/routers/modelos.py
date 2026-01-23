# backend/src/neurocampus/app/routers/modelos.py
# Ajuste Día 5 (B): integrar selección de datos por metodología antes de entrenar,
# conservando contrato y wiring existentes.

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from ..schemas.modelos import (
    EntrenarRequest, EntrenarResponse, EstadoResponse,
    RunSummary, RunDetails, ChampionInfo,
)
from ...models.templates.plantilla_entrenamiento import PlantillaEntrenamiento
from ...models.strategies.modelo_rbm_general import RBMGeneral
from ...models.strategies.modelo_rbm_restringida import RBMRestringida
from ...observability.bus_eventos import BUS  # capturamos eventos training.*
from typing import Dict, Any, Optional
import uuid

# NUEVO: utilidades para selección de datos
import pandas as pd
from pathlib import Path
try:
    # Disponible tras agregar metodologia.py (Día 5 B)
    from ...models.strategies.metodologia import SeleccionConfig, resolver_metodologia
except Exception:
    # Si aún no existe el módulo, definimos un shim mínimo para no romper import
    class SeleccionConfig:  # type: ignore
        def __init__(self, periodo_actual=None, ventana_n=4): ...
    def resolver_metodologia(nombre: str):  # type: ignore
        raise RuntimeError(
            "El módulo de metodologías no está disponible. "
            "Asegúrate de crear neurocampus/models/strategies/metodologia.py"
        )

# NUEVO: utilidades para inspeccionar runs y campeón desde artifacts/
from ...utils.runs_io import list_runs, load_run_details, load_current_champion

router = APIRouter()

# Registro in-memory de estados
# Estructura:
# _ESTADOS[job_id] = {
#   "job_id": str,
#   "status": "running" | "completed" | "failed" | "unknown",
#   "metrics": Dict[str, float],         # último snapshot
#   "history": list[dict[str, Any]],     # [{epoch, loss, ...metrics}]
#   "model": str,                        # "rbm_general" | "rbm_restringida" (si disponible)
#   "params": Dict[str, Any],            # hparams, etc. (si disponible)
#   "error": str | None
# }
_ESTADOS: Dict[str, Dict[str, Any]] = {}

# Para evitar suscribir múltiples veces los handlers de un mismo job con --reload
_OBS_WIRED_JOBS: set[str] = set()


def _normalize_hparams(hparams: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normaliza claves a minúsculas y retorna dict seguro (no None)."""
    if not hparams:
        return {}
    return {str(k).lower(): v for k, v in hparams.items()}


def _flatten_metrics_from_payload(payload: Dict[str, Any], allow_loss: bool = True) -> Dict[str, float]:
    """
    Cuando no venga `metrics` como dict explícito en el payload del evento,
    intenta aplanar los pares numéricos (excepto campos de control).
    """
    if not payload:
        return {}
    ctrl = {"correlation_id", "epoch", "loss", "event", "model", "params", "final_metrics"}
    out: Dict[str, float] = {}
    for k, v in payload.items():
        if k in ctrl:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # opcionalmente incluir loss
    if allow_loss and "loss" in payload and isinstance(payload["loss"], (int, float)):
        out.setdefault("loss", float(payload["loss"]))
    return out


def _wire_job_observers(job_id: str) -> None:
    """
    Se suscribe al BUS para capturar:
    - training.started: inicializa metadatos (modelo/params) si vienen en payload
    - training.epoch_end: agrega un punto a history[] y actualiza metrics
    - training.completed: marca estado y métricas finales
    - training.failed: marca error
    Idempotente por job_id (no re-suscribe en recargas).
    """
    if job_id in _OBS_WIRED_JOBS:
        return

    def _match(evt) -> bool:
        # Coincidir por correlation_id == job_id
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
        # Si llegan params desde el evento, preferirlos sobre los iniciales
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

        # Métricas pueden venir como bloque `metrics` o aplanadas
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = _flatten_metrics_from_payload(payload, allow_loss=True)

        # Guardar punto de la curva (epoch/loss + métricas)
        point = {"epoch": epoch}
        if isinstance(loss, (int, float)):
            point["loss"] = float(loss)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("epoch",):
                    point[k] = float(v)

        st["history"].append(point)
        # Snapshot de últimas métricas
        st["metrics"] = {k: v for k, v in point.items() if k not in ("epoch",)}

    def _on_completed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        payload = evt.payload or {}
        final_metrics = payload.get("final_metrics")

        if not isinstance(final_metrics, dict):
            # fallback: aplanar payload (sin epoch) o usar último snapshot
            final_metrics = _flatten_metrics_from_payload(payload, allow_loss=True)
            if not final_metrics:
                final_metrics = st.get("metrics", {})

        st["metrics"] = final_metrics
        st["status"] = "completed"

    def _on_failed(evt) -> None:
        if not _match(evt) or job_id not in _ESTADOS:
            return
        st = _ESTADOS[job_id]
        st["status"] = "failed"
        st["error"] = evt.payload.get("error", "unknown error")

    # Suscripciones (best-effort; el BUS no implementa unsubscribe)
    BUS.subscribe("training.started", _on_started)
    BUS.subscribe("training.epoch_end", _on_epoch_end)
    BUS.subscribe("training.completed", _on_completed)
    BUS.subscribe("training.failed", _on_failed)

    _OBS_WIRED_JOBS.add(job_id)


# -----------------------------
# Selección de datos (Día 5 B)
# -----------------------------
def _strip_localfs(uri: str) -> str:
    """Convierte 'localfs://path/to/file' a ruta local 'path/to/file'."""
    if isinstance(uri, str) and uri.startswith("localfs://"):
        return uri.replace("localfs://", "", 1)
    return uri


def _resolve_data_path(req: EntrenarRequest) -> str:
    """
    Determina la ruta al dataset a usar:
    - Si viene req.data_ref, la usa (admite esquema localfs://).
    - Si no, intenta 'historico/unificado.parquet' (Día 5 A).
    """
    if getattr(req, "data_ref", None):
        p = _strip_localfs(req.data_ref)  # type: ignore[arg-type]
        return p
    hist = Path("historico") / "unificado.parquet"
    if hist.exists():
        return str(hist)
    raise HTTPException(status_code=400, detail="No hay data_ref ni histórico unificado disponible.")


def _read_dataframe_any(path_or_uri: str) -> pd.DataFrame:
    """Lee un parquet local. (Extensible a otros esquemas si se requiere)."""
    p = _strip_localfs(path_or_uri)
    try:
        return pd.read_parquet(p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el dataset: {e}")


def _prepare_selected_data(req: EntrenarRequest, job_id: str) -> str:
    """
    Aplica la metodología de selección de datos y retorna
    una ruta temporal (parquet) con el subconjunto elegido.
    """
    # 1) Resolver dataset fuente (data_ref o histórico unificado)
    data_ref = _resolve_data_path(req)
    df = _read_dataframe_any(data_ref)

    # 2) Resolver metodología y config (con defaults robustos si el schema aún no se actualizó)
    metodologia = getattr(req, "metodologia", None) or "periodo_actual"
    periodo_actual = getattr(req, "periodo_actual", None)
    ventana_n = getattr(req, "ventana_n", None) or 4

    metodo = resolver_metodologia(str(metodologia).lower())
    cfg = SeleccionConfig(periodo_actual=periodo_actual, ventana_n=int(ventana_n))

    df_sel = metodo.seleccionar(df, cfg)
    if df_sel.empty:
        raise HTTPException(status_code=400, detail="Selección de datos vacía según la metodología/periodo.")

    # 3) Persistir subconjunto para el entrenamiento
    tmp_dir = Path("data/.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref = tmp_dir / f"df_sel_{job_id}.parquet"
    try:
        df_sel.to_parquet(tmp_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo materializar el subconjunto: {e}")

    return str(tmp_ref)


def _run_training(job_id: str, req: EntrenarRequest):
    # Elige estrategia
    estrategia = RBMGeneral() if req.modelo == "rbm_general" else RBMRestringida()
    tpl = PlantillaEntrenamiento(estrategia)

    # Asegurar wiring de observabilidad para este job antes de correr
    _wire_job_observers(job_id)

    # NUEVO: preparar subconjunto seleccionado según metodología
    selected_ref = _prepare_selected_data(req, job_id)

    # Ejecuta entrenamiento (emite training.* que recogerán los handlers)
    out = tpl.run(
        selected_ref,  # <-- usar subconjunto
        req.epochs,
        {**(_normalize_hparams(req.hparams)), "job_id": job_id},
        model_name=req.modelo,
    )

    # Consolidar estado final (por si el template devolvió info adicional)
    # Nota: out = {"job_id", "status", "metrics", "history?"} según plantilla
    st = _ESTADOS.get(job_id, {})
    st.update(out)
    _ESTADOS[job_id] = st


@router.post("/entrenar", response_model=EntrenarResponse)
def entrenar(req: EntrenarRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # Normaliza hparams para mantener contrato consistente a lo largo del flujo
    hp_norm = _normalize_hparams(req.hparams)

    # Estado inicial visible para la UI
    _ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "metrics": {},
        "history": [],  # acumularemos aquí cada epoch_end
        "model": req.modelo,
        # Guardamos también la configuración de metodología para trazabilidad
        "params": {
            "epochs": req.epochs,
            **hp_norm,
            "metodologia": getattr(req, "metodologia", "periodo_actual"),
            "periodo_actual": getattr(req, "periodo_actual", None),
            "ventana_n": getattr(req, "ventana_n", None),
            "data_ref": getattr(req, "data_ref", None) or "historico/unificado.parquet",
        },
        "error": None,
    }

    # Lanza en background con hparams normalizados, preservando otros campos del request
    try:
        # Pydantic v2
        req_norm = req.model_copy(update={"hparams": hp_norm})  # type: ignore[attr-defined]
    except AttributeError:
        # Pydantic v1
        req_norm = req.copy(update={"hparams": hp_norm})  # type: ignore

    bg.add_task(_run_training, job_id, req_norm)

    return EntrenarResponse(job_id=job_id, status="running", message="Entrenamiento lanzado")


@router.get("/estado/{job_id}", response_model=EstadoResponse)
def estado(job_id: str):
    # Mantener compat con esquema actual: si no existe, devolver "unknown"
    st = _ESTADOS.get(job_id) or {"job_id": job_id, "status": "unknown", "metrics": {}}
    # OJO: si tu EstadoResponse no define "history", FastAPI filtrará ese campo.
    return EstadoResponse(**st)


# ---------------------------------------------------------------------------
# NUEVO: Endpoints para listar runs y consultar campeón (Flujo 3)
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
      - model_name: ej 'rbm', 'rbm_general'
      - dataset_id / dataset / periodo: filtra por dataset asociado al run

    Compatibilidad:
      - Si no envías filtros, retorna todos los runs disponibles (como antes).
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
      - artifacts/runs/<run_id>/metrics.json (obligatorio)
      - artifacts/runs/<run_id>/config.snapshot.yaml o config.yaml (opcional)
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
    champ = load_current_champion(model_name=model_name, dataset_id=ds)
    if not champ:
        raise HTTPException(status_code=404, detail="No hay campeón registrado")
    return ChampionInfo(**champ)
