"""
Router del contexto 'jobs'.

Uso:
- Operaciones relacionadas con ejecución y estado de jobs en background.
- Proporciona puntos de extensión para colas, schedulers, etc.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
def ping() -> dict:
    """
    Comprobación rápida de vida del router /jobs.
    Ayuda a verificar que el prefijo y el registro en main.py funcionan.
    """
    return {"jobs": "pong"}