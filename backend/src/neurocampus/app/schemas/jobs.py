"""
Schemas (modelos Pydantic) del dominio 'jobs'.
"""
from pydantic import BaseModel, Field

class JobStatus(BaseModel):
    """
    Estado mínimo de un job:
    - id: identificador del job (uuid/string)
    - estado: valor simple 'pending', 'running', 'done', 'error'
    """
    id: str = Field(..., description="Identificador único del job")
    estado: str = Field("pending", description="Estado actual del job")