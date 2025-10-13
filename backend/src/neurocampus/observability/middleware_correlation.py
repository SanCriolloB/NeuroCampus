# backend/src/neurocampus/observability/middleware_correlation.py
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response

HEADER = "X-Correlation-Id"

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get(HEADER) or str(uuid.uuid4())
        # Disponibilizar para handlers/routers
        request.state.correlation_id = cid
        response: Response = await call_next(request)
        # Eco para depuración cliente-servidor
        response.headers[HEADER] = cid
        return response
