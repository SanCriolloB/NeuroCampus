"""
Router del contexto 'datos'.

Uso:
- Agrupar endpoints relacionados con ingestión, lectura y esquema de datos.
- Cada router permite mantener ordenado el código por dominio.
"""

from fastapi import APIRouter

# Cada router es independiente y luego se registra en main.py
router = APIRouter()

@router.get("/ping")
def ping() -> dict:
    """
    Comprobación rápida de vida del router /datos.
    """
    return {"datos": "pong"}