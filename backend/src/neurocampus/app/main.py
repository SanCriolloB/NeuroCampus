"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
"""

from fastapi import FastAPI

# Importamos routers del dominio (cada archivo define un APIRouter)
from .routers import datos, jobs

# Instancia de la aplicación (título visible en /docs y /openapi.json)
app = FastAPI(title="NeuroCampus API", version="0.1.0")

@app.get("/health")
def health() -> dict:
    """
    Endpoint de salud: permite saber si la API está arriba.
    Retorna un JSON muy simple con 'status: ok'.
    """
    return {"status": "ok"}

# Registro de routers bajo prefijos. Cada router agrupa rutas por contexto
# y define su propia etiqueta (tags) para la documentación automática.
app.include_router(datos.router, prefix="/datos", tags=["datos"])
app.include_router(jobs.router,  prefix="/jobs",  tags=["jobs"])