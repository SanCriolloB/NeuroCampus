"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importamos routers del dominio (cada archivo define un APIRouter)
from .routers import datos, jobs

# Instancia de la aplicación (título visible en /docs y /openapi.json)
app = FastAPI(title="NeuroCampus API", version="0.1.0")

# --- CORS (necesario para que el navegador permita las peticiones desde Vite) ---
# En desarrollo, habilitamos localhost y 127.0.0.1 (puerto 5173 por defecto en Vite)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # en desarrollo puedes usar ["*"] si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],            # importante: permite OPTIONS del preflight
    allow_headers=["*"],
)

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
