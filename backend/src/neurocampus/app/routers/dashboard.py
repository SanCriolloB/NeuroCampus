# backend/src/neurocampus/app/routers/dashboard.py
"""Router del Dashboard (lectura exclusiva de histórico).

Contrato de negocio (Dashboard)
------------------------------
- El Dashboard **NO** consulta datasets individuales directamente.
- Todas las métricas/series/catálogos se derivan del histórico:
  - ``historico/unificado.parquet`` (processed histórico)
  - ``historico/unificado_labeled.parquet`` (labeled histórico)

Este router expone la API ``/dashboard/*``. Por diseño:
- Los routers se mantienen delgados (HTTP/serialización).
- La lógica de lectura/filtrado vive en ``neurocampus.dashboard.queries``.

Notas
-----
- Por desempeño, el endpoint ``/dashboard/status`` **no lee** parquets completos;
  se basa en ``historico/manifest.json`` + existencia/mtime en disco.
- Los endpoints ``/dashboard/catalogos`` y ``/dashboard/kpis`` leen desde
  ``historico/unificado.parquet`` y aplican filtros estándar.
"""

from __future__ import annotations

import time
from pathlib import Path
import os
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Query

from neurocampus.app.schemas.dashboard import (
    DashboardCatalogos,
    DashboardKPIs,
    DashboardPeriodos,
    DashboardSeries,
    DashboardSeriesPoint,
    DashboardSentimiento,
    DashboardSentimientoBucket,
    DashboardRankings,
    DashboardRankingItem,
    DashboardRadar,
    DashboardRadarItem,
    DashboardStatus,
    FileStatus,
)
from neurocampus.dashboard.aggregations import (
    rankings,
    radar_preguntas,
    sentimiento_distribucion,
    series_por_periodo,
)

from neurocampus.app.schemas.dashboard import DashboardWordcloud, DashboardWordcloudItem

from neurocampus.dashboard.queries import (
    DashboardFilters,
    apply_filters,
    compute_catalogos,
    compute_kpis,
    load_processed,
)
from neurocampus.historico.manifest import load_manifest, list_periodos_from_manifest

from neurocampus.dashboard.predictions_kpis import (
    count_predicciones_total,
    resolve_dataset_ids_from_period_filters,
)


router = APIRouter()


# ---------------------------------------------------------------------------
# Resolución de rutas (misma convención que otros routers)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Devuelve la raíz del repo NeuroCampus.

    Desde `routers/` hay que subir 5 niveles para llegar a la raíz:
    [0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=repo_root
    """
    here = Path(__file__).resolve()
    return here.parents[5]


BASE_DIR = _repo_root()

HIST_DIR = BASE_DIR / "historico"
MANIFEST_PATH = HIST_DIR / "manifest.json"
UNIFICADO_PATH = HIST_DIR / "unificado.parquet"
UNIFICADO_LABELED_PATH = HIST_DIR / "unificado_labeled.parquet"


def _mtime_iso(path: Path) -> Optional[str]:
    """Devuelve mtime en ISO UTC, o None si el archivo no existe."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    # mtime en UTC (formato compacto similar al usado por jobs)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))


def _filters_from_query(
    periodo: Optional[str],
    periodo_from: Optional[str],
    periodo_to: Optional[str],
    docente: Optional[str],
    asignatura: Optional[str],
    programa: Optional[str],
) -> DashboardFilters:
    """Construye `DashboardFilters` desde query params.

    Se mantiene en el router (capa HTTP) porque representa parsing/contrato
    de entrada. La lógica de filtrado real vive en `neurocampus.dashboard.queries`.
    """
    return DashboardFilters(
        periodo=periodo,
        periodo_from=periodo_from,
        periodo_to=periodo_to,
        docente=docente,
        asignatura=asignatura,
        programa=programa,
    )


@router.get("/status", response_model=DashboardStatus)
def dashboard_status() -> DashboardStatus:
    """Estado del histórico para el Dashboard.

    Este endpoint es intencionalmente liviano: no carga parquets, solo inspecciona
    metadatos y existencia de archivos.
    """
    manifest_exists = MANIFEST_PATH.exists()
    manifest: Dict[str, Any] = load_manifest()
    periodos = list_periodos_from_manifest()

    processed_exists = UNIFICADO_PATH.exists()
    labeled_exists = UNIFICADO_LABELED_PATH.exists()

    return DashboardStatus(
        manifest_exists=manifest_exists,
        manifest_updated_at=manifest.get("updated_at"),
        manifest_corrupt=bool(manifest.get("corrupt_manifest")),
        periodos_disponibles=periodos,
        processed=FileStatus(
            path="historico/unificado.parquet",
            exists=processed_exists,
            mtime=_mtime_iso(UNIFICADO_PATH),
        ),
        labeled=FileStatus(
            path="historico/unificado_labeled.parquet",
            exists=labeled_exists,
            mtime=_mtime_iso(UNIFICADO_LABELED_PATH),
        ),
        # Para UI: processed se considera listo si existe parquet + hay al menos 1 periodo en manifest.
        ready_processed=bool(processed_exists and periodos),
        # labeled es opcional: se marca listo solo si existe el archivo.
        ready_labeled=bool(labeled_exists),
    )


@router.get("/periodos", response_model=DashboardPeriodos)
def dashboard_periodos() -> DashboardPeriodos:
    """Lista de periodos disponibles para filtros del Dashboard."""
    return DashboardPeriodos(items=list_periodos_from_manifest())


@router.get("/catalogos", response_model=DashboardCatalogos)
def dashboard_catalogos(
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardCatalogos:
    """Catálogos para poblar dropdowns del Dashboard."""
    # Leemos columnas mínimas. Si alguna columna no existe en el parquet, pandas
    # lanzará error; por compatibilidad, leemos todo y calculamos de forma defensiva.
    df = load_processed()
    df_f = apply_filters(df, _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa))
    docentes, asignaturas, programas = compute_catalogos(df_f)
    return DashboardCatalogos(docentes=docentes, asignaturas=asignaturas, programas=programas)


@router.get("/kpis", response_model=DashboardKPIs)
def dashboard_kpis(
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardKPIs:
    """KPIs básicos del Dashboard basados en histórico processed."""
    df = load_processed()
    df_f = apply_filters(df, _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa))
    base = compute_kpis(df_f)

    # KPI adicional: predicciones persistidas en artifacts/predictions.
    # Regla: artifacts está en la carpeta del repo (./artifacts), salvo override por NC_ARTIFACTS_DIR.
    artifacts_dir = Path(os.getenv("NC_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))

    # Dataset ids a considerar según periodo o rango.
    available_periodos = list_periodos_from_manifest()
    if not available_periodos:
        # Fallback defensivo si el manifest está vacío/corrupto.
        try:
            available_periodos = sorted(df["periodo"].dropna().astype(str).unique().tolist())
        except Exception:
            available_periodos = []

    dataset_ids = resolve_dataset_ids_from_period_filters(
        available_periodos=available_periodos,
        periodo=periodo,
        periodo_from=periodo_from,
        periodo_to=periodo_to,
    )

    pred = count_predicciones_total(
        artifacts_dir=artifacts_dir,
        dataset_ids=dataset_ids,
        docente=docente,
        asignatura=asignatura,
    )

    return DashboardKPIs(predicciones=pred, **base)

@router.get("/series", response_model=DashboardSeries)
def dashboard_series(
    metric: str = Query(
        "evaluaciones",
        description="Métrica solicitada (p.ej. evaluaciones, score_promedio, docentes, asignaturas).",
    ),  # noqa: B008
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardSeries:
    """Serie agregada por periodo desde histórico processed.

    El frontend usa esta serie para gráficas de evolución por periodo sin acceder
    directamente a datasets puntuales.
    """
    filters = _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa)

    try:
        points = series_por_periodo(metric, filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out_points = [
        DashboardSeriesPoint(periodo=str(p.get("periodo")), value=float(p.get("value") or 0.0))
        for p in points
    ]
    return DashboardSeries(metric=metric, points=out_points)

@router.get("/radar", response_model=DashboardRadar)
def dashboard_radar(
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardRadar:
    """Radar de indicadores (promedio de preguntas 1..10).

    El Dashboard usa este endpoint para construir el radar "Perfil Global de Indicadores".

    Notas
    -----
    - Se basa únicamente en histórico processed (``historico/unificado.parquet``).
    - El output está en la escala del histórico (típicamente 0–50).
      Si el frontend desea 0–5, puede dividir por 10.
    """
    filters = _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa)

    try:
        rows = radar_preguntas(filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = [
        DashboardRadarItem(key=str(r.get("key")), value=float(r.get("value") or 0.0))
        for r in (rows or [])
    ]
    return DashboardRadar(items=items)


@router.get("/wordcloud", response_model=DashboardWordcloud)
def dashboard_wordcloud(
    limit: int = Query(80, description="Cantidad máxima de tokens.", ge=1, le=500),  # noqa: B008
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardWordcloud:
    """Wordcloud (top términos) desde histórico labeled.

    Devuelve los tokens más frecuentes a partir del texto procesado en
    ``historico/unificado_labeled.parquet``.

    Respuestas
    ----------
    - 200: lista (posiblemente vacía) de tokens y sus frecuencias
    - 404: histórico labeled no disponible
    """
    filters = _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa)

    from neurocampus.dashboard.aggregations import wordcloud_terms

    try:
        rows = wordcloud_terms(filters, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = [
        DashboardWordcloudItem(text=str(r.get("text")), value=int(r.get("value") or 0))
        for r in (rows or [])
    ]
    return DashboardWordcloud(items=items)

@router.get("/sentimiento", response_model=DashboardSentimiento)
def dashboard_sentimiento(
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardSentimiento:
    """Distribución de sentimiento desde histórico labeled.

    Este endpoint solo está disponible cuando existe
    ``historico/unificado_labeled.parquet`` (labeled histórico).

    Respuestas
    ----------
    - 200: devuelve buckets neg/neu/pos (proporciones 0..1)
    - 404: histórico labeled no disponible
    - 400: histórico labeled existe pero no hay columnas de sentimiento compatibles
    """
    filters = _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa)

    try:
        payload = sentimiento_distribucion(filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buckets = payload.get("buckets") or []
    out = [
        DashboardSentimientoBucket(label=str(b.get("label")), value=float(b.get("value") or 0.0))
        for b in buckets
    ]
    return DashboardSentimiento(buckets=out)


@router.get("/rankings", response_model=DashboardRankings)
def dashboard_rankings(
    by: str = Query(
        ...,
        description="Dimensión del ranking: docente o asignatura.",
    ),  # noqa: B008
    metric: str = Query(
        "score_promedio",
        description="Métrica de ranking: score_promedio o evaluaciones.",
    ),  # noqa: B008
    order: str = Query(
        "desc",
        description="Orden: asc o desc.",
    ),  # noqa: B008
    limit: int = Query(
        8,
        description="Cantidad máxima de items.",
        ge=1,
        le=200,
    ),  # noqa: B008
    periodo: Optional[str] = Query(None, description="Periodo exacto (prioriza sobre rango)."),  # noqa: B008
    periodo_from: Optional[str] = Query(None, description="Inicio de rango (incl.)."),  # noqa: B008
    periodo_to: Optional[str] = Query(None, description="Fin de rango (incl.)."),  # noqa: B008
    docente: Optional[str] = Query(None, description="Filtro por docente (opcional)."),  # noqa: B008
    asignatura: Optional[str] = Query(None, description="Filtro por asignatura (opcional)."),  # noqa: B008
    programa: Optional[str] = Query(None, description="Filtro por programa (opcional)."),  # noqa: B008
) -> DashboardRankings:
    """Ranking (top/bottom) derivado del histórico processed.

    Este endpoint existe porque el Dashboard UI consume rankings para tablas como:
    - Ranking de Docentes (por score_promedio)
    - Distribución/Ranking por Asignatura (por evaluaciones u otra métrica)

    Notas
    -----
    - Solo usa histórico processed (``historico/unificado.parquet``).
    - Si falta el histórico o columnas requeridas, responde con 404/400.
    """
    filters = _filters_from_query(periodo, periodo_from, periodo_to, docente, asignatura, programa)

    try:
        rows = rankings(by=by, metric=metric, order=order, limit=limit, filters=filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Normalizamos salida a contrato estable (name/value).
    # Si por datos faltantes `value` llega como None, lo convertimos a 0.0 para no romper UI.
    items = [
        DashboardRankingItem(name=str(r.get("key")), value=float(r.get("value") or 0.0))
        for r in (rows or [])
    ]
    return DashboardRankings(by=by, metric=metric, order=order, items=items)
