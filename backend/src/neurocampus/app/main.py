"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173)
- Conectar el destino de observabilidad (logging) para eventos training.*
- Inyectar middleware de Correlation-Id (X-Correlation-Id) para trazabilidad
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# NEW: configuración de logging (dictConfig)
from neurocampus.app.logging_config import setup_logging

# NEW: middleware de trazabilidad y LogRecordFactory contextual
from neurocampus.observability.middleware_correlation import CorrelationIdMiddleware
from neurocampus.observability.logging_context import install_logrecord_factory

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

# --- Correlation-Id Middleware (trazabilidad end-to-end) ---
# Agrega/propaga X-Correlation-Id y lo expone en request.state.correlation_id
app.add_middleware(CorrelationIdMiddleware)


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
    - Configura logging (dictConfig con filtro cid)
    - Instala LogRecordFactory que inyecta correlation_id desde ContextVar
    - Conecta el destino de logging para eventos training.*
    """
    # 1) dictConfig con filtro cid
    setup_logging()

    # 2) LogRecordFactory que añade correlation_id automáticamente
    install_logrecord_factory()

    # 3) Wiring de observabilidad existente
    _wire_observability_safe()


@app.get("/health")
def health() -> dict:
    """Endpoint de salud: permite saber si la API está arriba."""
    return {"status": "ok"}


# Registro de routers bajo prefijos. Cada router agrupa rutas por contexto
# y define su propia etiqueta (tags) para la documentación automática.
app.include_router(datos.router,   prefix="/datos",   tags=["datos"])
app.include_router(jobs.router,    prefix="/jobs",    tags=["jobs"])
app.include_router(modelos.router, prefix="/modelos", tags=["modelos"])
