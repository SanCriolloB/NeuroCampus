# NeuroCampus-main/backend/src/neurocampus/app/routers/admin_cleanup.py

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

# --- Habilitar import del módulo tools.cleanup.py (que está en la raíz del repo) ---
# Estructura: backend/src/neurocampus/app/routers/admin_cleanup.py
# parents[0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=<repo-root>
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.cleanup import run_cleanup, LOG_FILE  # noqa: E402

router = APIRouter()

ADMIN_TOKEN = os.getenv("NC_ADMIN_TOKEN", "")


def require_token(authorization: Optional[str] = Header(default=None)):
    """Valida Authorization: Bearer <token> con NC_ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Admin token not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return True


class CleanupRequest(BaseModel):
    retention_days: int = Field(default=90, ge=0)
    keep_last: int = Field(default=3, ge=0)
    exclude_globs: Optional[str] = Field(default=None, description="Globs separados por coma")
    dry_run: bool = True
    force: bool = False
    trash_dir: str = ".trash"
    trash_retention_days: int = Field(default=14, ge=0)


@router.get("/admin/cleanup/inventory")
def get_inventory(
    retention_days: int = Query(90, ge=0),
    keep_last: int = Query(3, ge=0),
    exclude_globs: Optional[str] = Query(None),
    auth_ok: bool = Depends(require_token),
):
    """Inventario + candidatos (siempre dry_run)."""
    return run_cleanup(
        retention_days=retention_days,
        keep_last=keep_last,
        exclude_globs_str=exclude_globs,
        dry_run=True,
        force=False,
    )


@router.post("/admin/cleanup")
def post_cleanup(payload: CleanupRequest, auth_ok: bool = Depends(require_token)):
    """Ejecuta dry-run o mover a papelera (force)."""
    return run_cleanup(
        retention_days=payload.retention_days,
        keep_last=payload.keep_last,
        exclude_globs_str=payload.exclude_globs,
        dry_run=payload.dry_run,
        force=payload.force,
        trash_dir=payload.trash_dir,
        trash_retention_days=payload.trash_retention_days,
    )


@router.get("/admin/cleanup/logs")
def get_cleanup_logs(limit: int = Query(200, ge=1, le=5000), auth_ok: bool = Depends(require_token)):
    """Devuelve las últimas N líneas del log CSV como texto."""
    lf = Path(LOG_FILE)
    if not lf.exists():
        return {"lines": []}
    lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {"lines": lines[-limit:]}
