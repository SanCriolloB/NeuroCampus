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
# (Opcional) Placeholders para /datos/validar — previsto Día 3
# ---------------------------------------------------------------------------

class ValidacionSummary(BaseModel):
    rows: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    warnings: int = Field(..., ge=0)


class ValidacionCheck(BaseModel):
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    details: dict = Field(default_factory=dict)


class ValidacionRespuesta(BaseModel):
    summary: ValidacionSummary
    checks: List[ValidacionCheck]
    recommendations: List[str] = Field(default_factory=list)
