# backend/src/neurocampus/app/schemas/dashboard.py
"""Esquemas (Pydantic) para la API del Dashboard.

Este módulo define contratos de respuesta **estables** para los endpoints
``/dashboard/*``.

Motivación
----------
- Mantener la compatibilidad del frontend aun cuando evolucione la implementación
  de queries/aggregations.
- Evitar que el router contenga modelos inline difíciles de reutilizar.

Fuente de verdad
----------------
El Dashboard solo consume histórico (nunca datasets puntuales):
- ``historico/unificado.parquet`` (processed histórico)
- ``historico/unificado_labeled.parquet`` (labeled histórico)

Estos esquemas se irán usando progresivamente conforme se implementen los
endpoints ``/dashboard/catalogos``, ``/dashboard/kpis``, ``/dashboard/series``,
``/dashboard/sentimiento`` y ``/dashboard/rankings``.

Notas
-----
- Los modelos están diseñados para ser simples y serializables (JSON).
- Se prioriza claridad de nombres y docstrings para documentación con Sphinx.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FileStatus(BaseModel):
    """Estado mínimo de un artefacto del histórico."""

    path: str = Field(..., description="Ruta relativa dentro del repo.")
    exists: bool = Field(..., description="True si el archivo existe en disco.")
    mtime: Optional[str] = Field(
        None,
        description="Fecha de modificación (mtime) en ISO UTC si existe.",
    )


class DashboardStatus(BaseModel):
    """Contrato para ``GET /dashboard/status``."""

    manifest_exists: bool = Field(..., description="True si historico/manifest.json existe.")
    manifest_updated_at: Optional[str] = Field(
        None,
        description="Timestamp principal del manifest (UTC).",

    )
    manifest_corrupt: bool = Field(
        False,
        description="True si se detectó manifest corrupto (fallback a vacío).",

    )

    periodos_disponibles: List[str] = Field(
        default_factory=list,
        description="Lista ordenada de periodos disponibles para filtros.",
    )

    processed: FileStatus = Field(..., description="Estado de historico/unificado.parquet.")
    labeled: FileStatus = Field(..., description="Estado de historico/unificado_labeled.parquet.")

    ready_processed: bool = Field(..., description="True si el histórico processed está listo.")
    ready_labeled: bool = Field(..., description="True si el histórico labeled está listo.")


class DashboardPeriodos(BaseModel):
    """Contrato para ``GET /dashboard/periodos``."""

    items: List[str] = Field(default_factory=list)


class DashboardCatalogos(BaseModel):
    """Contrato para ``GET /dashboard/catalogos``.

    Devuelve listas válidas para poblar dropdowns del UI. Los nombres de campos
    deben mapearse al frontend sin cambiar el diseño.
    """

    docentes: List[str] = Field(default_factory=list)
    asignaturas: List[str] = Field(default_factory=list)
    programas: List[str] = Field(default_factory=list)


class DashboardKPIs(BaseModel):
    """Contrato para ``GET /dashboard/kpis``."""

    evaluaciones: int = Field(0, description="Cantidad de evaluaciones incluidas.")
    docentes: int = Field(0, description="Cantidad de docentes distintos.")
    asignaturas: int = Field(0, description="Cantidad de asignaturas distintas.")
    score_promedio: Optional[float] = Field(
        None,
        description="Score promedio (si aplica) en el rango/periodo solicitado.",
    )


class DashboardSeriesPoint(BaseModel):
    """Punto de serie temporal/agregada por periodo."""

    periodo: str
    value: float


class DashboardSeries(BaseModel):
    """Contrato para ``GET /dashboard/series``."""

    metric: str = Field(..., description="Métrica solicitada (p.ej. score_promedio).")
    points: List[DashboardSeriesPoint] = Field(default_factory=list)


class DashboardSentimientoBucket(BaseModel):
    """Bucket de sentimiento (solo si labeled disponible)."""

    label: str
    value: float


class DashboardSentimiento(BaseModel):
    """Contrato para ``GET /dashboard/sentimiento``."""

    buckets: List[DashboardSentimientoBucket] = Field(default_factory=list)


class DashboardRankingItem(BaseModel):
    """Ítem de ranking (top/bottom) para docente o asignatura."""

    name: str
    value: float


class DashboardRankings(BaseModel):
    """Contrato para ``GET /dashboard/rankings``."""

    by: str = Field(..., description="Dimensión del ranking (docente|asignatura).")
    metric: str = Field(..., description="Métrica usada (p.ej. score_total).")
    order: str = Field(..., description="Orden (asc|desc).")
    items: List[DashboardRankingItem] = Field(default_factory=list)
