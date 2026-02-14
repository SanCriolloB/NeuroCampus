# backend/src/neurocampus/app/routers/dashboard.py
"""Router del Dashboard (lectura exclusiva de histórico).

Contrato de negocio (Dashboard)
------------------------------
- El Dashboard **NO** consulta datasets individuales directamente.
- Todas las métricas/series/catálogos se derivan del histórico:
  - ``historico/unificado.parquet`` (processed histórico)
  - ``historico/unificado_labeled.parquet`` (labeled histórico)

Este router inicia la API ``/dashboard/*`` con endpoints base para que el
frontend pueda cablearse sin cambiar el diseño visual:
- ``GET /dashboard/status``: estado del histórico (existencia, timestamps, periodos).
- ``GET /dashboard/periodos``: lista de periodos disponibles (rápido, vía manifest).

Notas
-----
- Por desempeño, estos endpoints **no leen** parquets completos. Se basan en:
  - ``historico/manifest.json`` (metadatos livianos)
  - existencia/mtime de archivos en disco
- La actualización eager del manifest se realiza al finalizar la unificación
  (ver ``neurocampus.data.strategies.unificacion``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from neurocampus.historico.manifest import load_manifest, list_periodos_from_manifest


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


class FileStatus(BaseModel):
    """Estado mínimo de un artefacto del histórico."""

    path: str = Field(..., description="Ruta relativa dentro del repo.")
    exists: bool = Field(..., description="True si el archivo existe en disco.")
    mtime: Optional[str] = Field(None, description="mtime en ISO UTC si existe.")


class DashboardStatusResponse(BaseModel):
    """Respuesta del endpoint /dashboard/status."""

    manifest_exists: bool = Field(..., description="True si historico/manifest.json existe.")
    manifest_updated_at: Optional[str] = Field(
        None, description="Timestamp principal del manifest (UTC)."
    )
    manifest_corrupt: bool = Field(
        False, description="True si se detectó manifest corrupto (fallback a vacío)."
    )

    periodos_disponibles: List[str] = Field(default_factory=list)

    processed: FileStatus
    labeled: FileStatus

    ready_processed: bool = Field(..., description="True si el histórico processed está listo.")
    ready_labeled: bool = Field(..., description="True si el histórico labeled está listo.")


class PeriodosResponse(BaseModel):
    """Respuesta del endpoint /dashboard/periodos."""

    items: List[str] = Field(default_factory=list)


@router.get("/status", response_model=DashboardStatusResponse)
def dashboard_status() -> DashboardStatusResponse:
    """Estado del histórico para el Dashboard.

    Este endpoint es intencionalmente liviano: no carga parquets, solo inspecciona
    metadatos y existencia de archivos.
    """
    manifest_exists = MANIFEST_PATH.exists()
    manifest: Dict[str, Any] = load_manifest()
    periodos = list_periodos_from_manifest()

    processed_exists = UNIFICADO_PATH.exists()
    labeled_exists = UNIFICADO_LABELED_PATH.exists()

    return DashboardStatusResponse(
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


@router.get("/periodos", response_model=PeriodosResponse)
def dashboard_periodos() -> PeriodosResponse:
    """Lista de periodos disponibles para filtros del Dashboard."""
    return PeriodosResponse(items=list_periodos_from_manifest())
