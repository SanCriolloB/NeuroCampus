"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI
- Registro de routers (rutas agrupadas por dominio)
- Endpoints globales mínimos (/health)
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173)
- Conectar el destino de observabilidad (logging) para eventos training.* y prediction.*
- Inyectar middleware de Correlation-Id (X-Correlation-Id) para trazabilidad
- Aplicar límite de tamaño de subida (413) según NC_MAX_UPLOAD_MB
"""

from __future__ import annotations

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configuración de logging (dictConfig)
from neurocampus.app.logging_config import setup_logging

# Middleware de trazabilidad y LogRecordFactory contextual
from neurocampus.observability.middleware_correlation import CorrelationIdMiddleware
from neurocampus.observability.logging_context import install_logrecord_factory

# Routers del dominio
from .routers import datos, jobs, modelos, prediccion, admin_cleanup

# Instancia de la aplicación (título visible en /docs y /openapi.json)
app = FastAPI(title="NeuroCampus API", version=os.getenv("API_VERSION", "0.6.0"))

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # en desarrollo podrías usar ["*"] si lo prefieres
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---------------------------------------------------------------------------
# Límite de subida por Content-Length → 413 Payload Too Large
#   - Controlado por NC_MAX_UPLOAD_MB (entero, por defecto 10)
#   - Defensa adicional a la de uvicorn (que puede configurarse con --limit-max-request-size)
# ---------------------------------------------------------------------------
MAX_MB = int(os.getenv("NC_MAX_UPLOAD_MB", "10"))
MAX_BYTES = MAX_MB * 1024 * 1024

class MaxSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_len = request.headers.get("content-length")
        try:
            if content_len is not None and int(content_len) > MAX_BYTES:
                return JSONResponse(
                    {"detail": f"Archivo demasiado grande (> {MAX_MB}MB)"},
                    status_code=413,
                )
        except ValueError:
            # Si el header no es un entero válido, dejamos que continúe el flujo normal.
            pass
        return await call_next(request)

app.add_middleware(MaxSizeMiddleware)

# --- Correlation-Id Middleware (trazabilidad end-to-end) ---
# Agrega/propaga X-Correlation-Id y lo expone en request.state.correlation_id
app.add_middleware(CorrelationIdMiddleware)


def _wire_observability_safe() -> None:
    """
    Conecta el handler de logging a los eventos training.* y prediction.* si el módulo existe.
    Si no existe (aún no creado), la app sigue funcionando sin observabilidad.
    """
    log = logging.getLogger("neurocampus")
    try:
        # Preferimos import absoluto con --app-dir backend/src
        from neurocampus.observability.destinos.log_handler import wire_logging_destination

        wire_logging_destination()
        log.info("Observability wiring OK: training.* & prediction.* -> logging.INFO")
    except ModuleNotFoundError as e:
        log.warning(
            "Observability module not found: %s. La API arrancará sin logging de training.* / prediction.* "
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
    - Conecta el destino de logging para eventos training.* y prediction.*
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
app.include_router(datos.router,         prefix="/datos",       tags=["datos"])
app.include_router(jobs.router,          prefix="/jobs",        tags=["jobs"])
app.include_router(modelos.router,       prefix="/modelos",     tags=["modelos"])
app.include_router(prediccion.router,    prefix="/prediccion",  tags=["prediccion"])
app.include_router(admin_cleanup.router,                     tags=["admin"])
