# backend/src/neurocampus/observability/middleware_correlation.py
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response
from .logging_context import correlation_id_var

HEADER = "X-Correlation-Id"

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    - Lee o genera un X-Correlation-Id por request
    - Lo expone en request.state.correlation_id
    - Lo agrega a la respuesta
    - Lo guarda en ContextVar para que el logger lo incluya automáticamente
    """
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get(HEADER) or str(uuid.uuid4())
        request.state.correlation_id = cid

        # ContextVar: set -> call_next -> reset
        token = correlation_id_var.set(cid)
        try:
            response: Response = await call_next(request)
        finally:
            correlation_id_var.reset(token)

        response.headers[HEADER] = cid
        return response
