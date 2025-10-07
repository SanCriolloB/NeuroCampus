"""
Schemas (modelos Pydantic) del dominio 'datos'.

- Mantener modelos de request/response separados por dominio reduce acoplamiento.
- En documentación OpenAPI, estos modelos aparecen como componentes reutilizables.
"""
from pydantic import BaseModel, Field

class UploadResumen(BaseModel):
    """
    Resumen mínimo de una carga de datos:
    - total: cantidad de registros procesados
    - ok: bandera general de éxito
    """
    total: int = Field(0, description="Número total de registros procesados")
    ok: bool = Field(True, description="Indica si la operación global fue exitosa")