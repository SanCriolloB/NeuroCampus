"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173)
- Conectar el destino de observabilidad (logging) para eventos training.* y prediction.*
- Inyectar middleware de Correlation-Id (X-Correlation-Id) para trazabilidad
- Aplicar límite de tamaño de subida (413) según NC_MAX_UPLOAD_MB (solo en /datos/validar y /datos/upload)
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuración de logging (dictConfig)
from neurocampus.app.logging_config import setup_logging

# Middleware de trazabilidad y LogRecordFactory contextual
from neurocampus.observability.middleware_correlation import CorrelationIdMiddleware
from neurocampus.observability.logging_context import install_logrecord_factory
from neurocampus.app.routers import predicciones


# Routers del dominio
from .routers import datos, jobs, modelos, prediccion, admin_cleanup

# ---------------------------------------------------------------------------
# CORS (necesario para que el navegador permita las peticiones desde Vite)
#   - NC_ALLOWED_ORIGINS tiene prioridad (coma-separados)
#   - CORS_ALLOW_ORIGINS se acepta como respaldo (compat)
#   - Si ninguno está definido, se usan los defaults locales
# ---------------------------------------------------------------------------
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_env_origins_nc = os.getenv("NC_ALLOWED_ORIGINS")  # p. ej.: "http://localhost:5173,http://127.0.0.1:5173"
_env_origins_old = os.getenv("CORS_ALLOW_ORIGINS")  # compat retro

_raw_origins = _env_origins_nc if _env_origins_nc else _env_origins_old
ALLOWED_ORIGINS = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else _default_origins
)

# ---------------------------------------------------------------------------
# Límite de subida por Content-Length → 413 Payload Too Large (solo datos.*)
#   - Controlado por NC_MAX_UPLOAD_MB (entero, por defecto 10)
#   - Implementado vía middleware de FastAPI (no por flag de Uvicorn)
# ---------------------------------------------------------------------------
MAX_MB = int(os.getenv("NC_MAX_UPLOAD_MB", "10"))
MAX_BYTES = MAX_MB * 1024 * 1024
_UPLOAD_PATHS = ("/datos/upload", "/datos/validar")

async def limit_upload_size(request: Request, call_next):
    """Middleware para limitar el tamaño de carga en endpoints de datos."""
    path = request.url.path
    if path.startswith(_UPLOAD_PATHS):
        cl = request.headers.get("content-length")
        try:
            if cl is not None and int(cl) > MAX_BYTES:
                return JSONResponse(
                    {"detail": f"Archivo supera el límite de {MAX_MB} MB"},
                    status_code=413,
                )
        except ValueError:
            pass
    return await call_next(request)


def _wire_observability_safe() -> None:
    """Conecta el handler de logging a los eventos training.* y prediction.*."""
    log = logging.getLogger("neurocampus")
    try:
        from neurocampus.observability.destinos.log_handler import wire_logging_destination
        wire_logging_destination()
        log.info("Observability wiring OK: training.* & prediction.* -> logging.INFO")
    except ModuleNotFoundError as e:
        log.warning(
            "Observability module not found: %s. La API arrancará sin logging de training.* / prediction.*.",
            e,
        )
    except Exception as e:
        log.warning("Fallo conectando observabilidad (se ignora para no bloquear): %s", e)


# ---------------------------------------------------------------------------
# Lifespan moderno (reemplaza @app.on_event("startup"))
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bloque lifespan para configurar logging y observabilidad."""
    setup_logging()
    install_logrecord_factory()
    _wire_observability_safe()
    yield  # Aquí se podría agregar lógica de shutdown si fuera necesario.


# ---------------------------------------------------------------------------
# Instanciación de la aplicación
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NeuroCampus API",
    version=os.getenv("API_VERSION", "0.6.0"),
    lifespan=lifespan,
)

# --- Middlewares ---
app.add_middleware(CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)
app.middleware("http")(limit_upload_size)
app.add_middleware(CorrelationIdMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Endpoint de salud: permite saber si la API está arriba."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Registro de routers
# ---------------------------------------------------------------------------
app.include_router(datos.router,      prefix="/datos",       tags=["datos"])
app.include_router(jobs.router,       prefix="/jobs",        tags=["jobs"])
app.include_router(modelos.router,    prefix="/modelos",     tags=["modelos"])
app.include_router(prediccion.router, prefix="/prediccion",  tags=["prediccion"])
app.include_router(predicciones.router)
app.include_router(admin_cleanup.router,                     tags=["admin"])
