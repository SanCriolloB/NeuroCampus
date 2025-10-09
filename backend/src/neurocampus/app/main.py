"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173)
- Conectar el destino de observabilidad (logging) para eventos training.*
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Routers del dominio
from .routers import datos, jobs, modelos

# Instancia de la aplicación (título visible en /docs y /openapi.json)
app = FastAPI(title="NeuroCampus API", version="0.4.0")

# --- CORS (necesario para que el navegador permita las peticiones desde Vite) ---
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

def _wire_observability_safe() -> None:
    """
    Conecta el handler de logging a los eventos training.* si el módulo existe.
    Si no existe (aún no creado), la app sigue funcionando sin observabilidad.
    """
    log = logging.getLogger("neurocampus")
    try:
        # Preferimos import absoluto con --app-dir backend/src
        from neurocampus.observability.destinos.log_handler import wire_logging_destination
        wire_logging_destination()
        log.info("Observability wiring OK: training.* -> logging.INFO")
    except ModuleNotFoundError as e:
        log.warning(
            "Observability module not found: %s. La API arrancará sin logging de training.* "
            "(crea backend/src/neurocampus/observability/destinos/log_handler.py y __init__.py).",
            e,
        )
    except Exception as e:
        log.warning("Fallo conectando observabilidad (se ignora para no bloquear): %s", e)


@app.on_event("startup")
def _startup_observability_wiring() -> None:
    """
    Conecta una única vez el handler de logging al bus de eventos.
    De esta forma, los eventos training.* emitidos por la plantilla de entrenamiento
    quedarán registrados en logs sin necesidad de tocar el resto del código.
    """
    wire_logging_destination()

@app.get("/health")
def health() -> dict:
    """Endpoint de salud: permite saber si la API está arriba."""
    return {"status": "ok"}

# Registro de routers bajo prefijos. Cada router agrupa rutas por contexto
# y define su propia etiqueta (tags) para la documentación automática.
app.include_router(datos.router,   prefix="/datos",   tags=["datos"])
app.include_router(jobs.router,    prefix="/jobs",    tags=["jobs"])
app.include_router(modelos.router, prefix="/modelos", tags=["modelos"])
