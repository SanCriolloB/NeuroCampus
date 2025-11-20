"""
Schemas (modelos Pydantic) del dominio 'datos'.

- Mantener modelos de request/response separados por dominio reduce acoplamiento.
- En documentación OpenAPI, estos modelos aparecen como componentes reutilizables.

Actualizado (Día 2):
- Se agregan los modelos para exponer el esquema consumido por la UI:
  * EsquemaCol, EsquemaResponse
- Se agrega la respuesta de carga (mock) para /datos/upload:
  * DatosUploadResponse
- Se mantiene UploadResumen (del Día 1) para métricas/uso interno.

Actualizado (Día 3):
- Se añaden contratos explícitos para /datos/validar:
  * ValidIssue, ValidSummary, DatosValidarResponse

NOTA: Los campos derivados de PLN (comentario.sent_pos, .sent_neg, .sent_neu)
NO forman parte del dataset de entrada. Se calcularán en una etapa posterior
(Día 6) y por tanto NO aparecen como columnas requeridas en el esquema.
"""

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Modelos ya existentes (Día 1)
# ---------------------------------------------------------------------------

class UploadResumen(BaseModel):
    """
    Resumen mínimo de una carga de datos:
    - total: cantidad de registros procesados
    - ok: bandera general de éxito
    """
    total: int = Field(0, description="Número total de registros procesados")
    ok: bool = Field(True, description="Indica si la operación global fue exitosa")


# ---------------------------------------------------------------------------
# Día 2 — Contratos para /datos/esquema y /datos/upload (mock)
# ---------------------------------------------------------------------------

class EsquemaCol(BaseModel):
    """
    Define una columna del esquema que la UI utilizará para construir formularios
    dinámicos y validaciones de entrada.
    """
    name: str = Field(..., description="Nombre de la columna")
    dtype: Literal["string", "number", "integer", "boolean", "date"] = Field(
        ..., description="Tipo lógico esperado por el backend/UI"
    )
    required: bool = Field(
        False, description="Indica si la columna es obligatoria en el upload"
    )
    # Conjunto cerrado de valores válidos (para listas/desplegables).
    domain: Optional[List[str]] = Field(
        default=None, description="Valores permitidos (si aplica)"
    )
    # Límite inferior y superior para valores numéricos.
    range: Optional[Tuple[float, float]] = Field(
        default=None, description="Rango permitido para números (min, max)"
    )
    # Longitud máxima para strings (si aplica).
    max_len: Optional[int] = Field(
        default=None, description="Longitud máxima permitida para strings"
    )


class EsquemaResponse(BaseModel):
    """
    Respuesta de GET /datos/esquema. Versiona el contrato para que
    la UI pueda cachear y detectar cambios.
    """
    version: str = Field("v0.2.0", description="Versión del esquema publicado")
    columns: List[EsquemaCol] = Field(
        ..., description="Definición de columnas esperadas por el upload"
    )


class DatosUploadResponse(BaseModel):
    """
    Respuesta de POST /datos/upload (mock Día 2).
    Representa el identificador lógico del dataset y metadatos de almacenamiento.
    """
    dataset_id: str = Field(..., description="Identificador lógico del dataset (p.ej. periodo)")
    rows_ingested: int = Field(..., description="Número de filas aceptadas/ingresadas")
    stored_as: str = Field(..., description="URI/Path de almacenamiento (csv/parquet/etc.)")
    warnings: List[str] = Field(
        default_factory=list,
        description="Advertencias no bloqueantes (campos vacíos, coerciones, etc.)"
    )


# ---------------------------------------------------------------------------
# Día 3 — Contratos para /datos/validar
# ---------------------------------------------------------------------------

class ValidIssue(BaseModel):
    """
    Elemento de la lista de hallazgos de validación.
    - code: identificador breve de la regla (p.ej., MISSING_COLUMN, BAD_TYPE)
    - severity: severidad del hallazgo (error | warning)
    - column: columna asociada (si aplica)
    - row: índice de fila asociado (si aplica)
    - message: descripción legible del hallazgo
    """
    code: str = Field(..., description="Código del hallazgo (p.ej., MISSING_COLUMN)")
    severity: Literal["error", "warning"] = Field(
        ..., description="Severidad del hallazgo"
    )
    column: Optional[str] = Field(
        None, description="Columna asociada al hallazgo (si aplica)"
    )
    row: Optional[int] = Field(
        None, description="Índice de fila asociado (si aplica)"
    )
    message: str = Field(..., description="Descripción legible del hallazgo")


class ValidSummary(BaseModel):
    """
    Resumen de la validación para KPIs de UI.
    - rows: número total de filas del dataset recibido
    - errors: cantidad total de hallazgos con severidad 'error'
    - warnings: cantidad total de hallazgos con severidad 'warning'
    - engine: motor de DataFrame usado ('pandas' | 'polars')
    """
    rows: int = Field(..., ge=0, description="Número de filas del dataset")
    errors: int = Field(..., ge=0, description="Cantidad de errores")
    warnings: int = Field(..., ge=0, description="Cantidad de advertencias")
    engine: str = Field(..., description="Engine de DataFrame utilizado")


class DatosValidarResponse(BaseModel):
    """
    Respuesta de POST /datos/validar.
    Contiene un resumen y el listado de issues detectados por la cadena de validación.
    """
    summary: ValidSummary = Field(..., description="KPIs de validación")
    issues: List[ValidIssue] = Field(
        default_factory=list,
        description="Listado de hallazgos (errores/advertencias)"
    )

# ---------------------------------------------------------------------------
# Resumen de dataset y análisis de sentimientos (para pestaña "Datos")
# ---------------------------------------------------------------------------

class ColumnaResumen(BaseModel):
    """
    Resumen de una columna del dataset para mostrar en la UI (tabla de columnas).
    """
    name: str = Field(..., description="Nombre de la columna")
    dtype: str = Field(..., description="Tipo lógico detectado (dtype normalizado)")
    non_nulls: int = Field(..., ge=0, description="Número de filas no nulas")
    sample_values: List[str] = Field(
        default_factory=list,
        description="Pequeña muestra de valores distintos para ayudar al usuario"
    )


class DatasetResumenResponse(BaseModel):
    """
    Respuesta de GET /datos/resumen.
    Provee KPIs generales del dataset y un resumen por columna.
    """
    dataset_id: str = Field(..., description="Identificador lógico del dataset (periodo, etc.)")
    n_rows: int = Field(..., ge=0, description="Cantidad de filas")
    n_cols: int = Field(..., ge=0, description="Cantidad de columnas")

    periodos: List[str] = Field(
        default_factory=list,
        description="Valores únicos de la columna 'periodo', si existe"
    )
    fecha_min: Optional[datetime.date] = Field(
        default=None, description="Fecha mínima detectada (si hay columna de fecha)"
    )
    fecha_max: Optional[datetime.date] = Field(
        default=None, description="Fecha máxima detectada (si hay columna de fecha)"
    )
    n_docentes: Optional[int] = Field(
        default=None, ge=0, description="Cantidad de docentes distintos (si hay columna compatible)"
    )
    n_asignaturas: Optional[int] = Field(
        default=None, ge=0, description="Cantidad de asignaturas distintas (si hay columna compatible)"
    )

    columns: List[ColumnaResumen] = Field(
        default_factory=list,
        description="Resumen de columnas (para tabla de la UI)"
    )


class SentimentBreakdown(BaseModel):
    """
    Conteo de comentarios por sentimiento.
    Las etiquetas son las usadas internamente en BETO/teacher: neg | neu | pos.
    """
    label: Literal["neg", "neu", "pos"] = Field(..., description="Etiqueta de sentimiento")
    count: int = Field(..., ge=0, description="Cantidad de comentarios")
    proportion: float = Field(..., ge=0.0, le=1.0, description="Proporción sobre el total [0,1]")


class SentimentByGroup(BaseModel):
    """
    Distribución de sentimientos por grupo (docente o asignatura).
    """
    group: str = Field(..., description="Nombre del docente/asignatura")
    counts: List[SentimentBreakdown] = Field(
        default_factory=list,
        description="Conteos de sentimientos en este grupo"
    )


class DatasetSentimientosResponse(BaseModel):
    """
    Respuesta de GET /datos/sentimientos.
    Diseñada para alimentar las gráficas de la pestaña Datos.
    """
    dataset_id: str = Field(..., description="Dataset base sobre el que se corrió BETO")
    total_comentarios: int = Field(..., ge=0, description="Total de filas con comentario no vacío")

    global_counts: List[SentimentBreakdown] = Field(
        default_factory=list,
        description="Distribución global de sentimientos (para gráfico principal)"
    )
    por_docente: List[SentimentByGroup] = Field(
        default_factory=list,
        description="Distribución de sentimientos por docente"
    )
    por_asignatura: List[SentimentByGroup] = Field(
        default_factory=list,
        description="Distribución de sentimientos por asignatura"
    )
