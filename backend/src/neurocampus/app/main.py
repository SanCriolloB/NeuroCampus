"""
Módulo principal de la API de NeuroCampus.

Responsabilidades:
- Instanciación de FastAPI.
- Registro de routers (rutas agrupadas por dominio).
- Endpoints globales mínimos (/health).
- Habilitar CORS para permitir acceso desde el frontend (Vite, puerto 5173).
- Conectar el destino de observabilidad (logging) para eventos training.* y prediction.*.
- Inyectar middleware de Correlation-Id (X-Correlation-Id) para trazabilidad.
- Aplicar límite de tamaño de subida (413) según NC_MAX_UPLOAD_MB (solo en /datos/validar y /datos/upload).

.. important::
   Los routers deben declarar rutas **relativas** al prefijo con el que se montan.
   Por ejemplo, el router de ``jobs`` se monta en ``/jobs``; por tanto, sus endpoints
   deben ser ``/preproc/...`` o ``/data/...`` (NO ``/jobs/...``), de lo contrario se
   produce el bug de prefijo duplicado ``/jobs/jobs/...``.
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
    """Middleware para limitar el tamaño de carga en endpoints de datos.

    :param request: Request entrante.
    :param call_next: Handler del siguiente middleware/endpoint.
    :return: Respuesta HTTP, potencialmente 413 si se excede el límite.
    """
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
            # Content-Length inválido: no bloqueamos, dejamos pasar.
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


def _warn_double_jobs_prefix(app: FastAPI) -> None:
    """Detecta rutas accidentales con prefijo duplicado ``/jobs/jobs``.

    Esto ayuda a evitar que regrese el bug donde:
    - main monta router jobs con ``prefix='/jobs'`` y
    - el router declara rutas que ya empiezan por ``/jobs/...``.

    :param app: Instancia FastAPI ya con routers registrados.
    """
    log = logging.getLogger("neurocampus")
    try:
        paths = [getattr(r, "path", "") for r in app.routes]
        if any(p.startswith("/jobs/jobs/") for p in paths):
            log.warning(
                "Detectadas rutas con prefijo duplicado '/jobs/jobs/'. "
                "Revisa que el router de jobs NO declare rutas que empiecen por '/jobs'."
            )
    except Exception:
        # Nunca bloquear el arranque por esto.
        pass


# ---------------------------------------------------------------------------
# Lifespan moderno (reemplaza @app.on_event("startup"))
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bloque lifespan para configurar logging y observabilidad."""
    setup_logging()
    install_logrecord_factory()
    _wire_observability_safe()
    _warn_double_jobs_prefix(app)
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
app.add_middleware(
    CORSMiddleware,
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
app.include_router(datos.router, prefix="/datos", tags=["datos"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(modelos.router, prefix="/modelos", tags=["modelos"])
app.include_router(prediccion.router, prefix="/prediccion", tags=["prediccion"])
app.include_router(admin_cleanup.router, tags=["admin"])
