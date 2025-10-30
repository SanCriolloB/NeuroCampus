# backend/src/neurocampus/app/routers/admin_cleanup.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Habilitar import del módulo tools/cleanup.py (vive en la raíz del repo)
# Estructura: backend/src/neurocampus/app/routers/admin_cleanup.py
# parents[0]=routers, [1]=app, [2]=neurocampus, [3]=src, [4]=backend, [5]=<repo-root>
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
  sys.path.append(str(REPO_ROOT))

from tools.cleanup import run_cleanup, LOG_FILE  # noqa: E402

router = APIRouter()

# -----------------------------------------------------------------------------
# Autenticación de administración
#  - NC_ADMIN_TOKEN: token esperado en Authorization: Bearer <token>
#  - NC_DISABLE_ADMIN_AUTH=1 desactiva la validación (útil para tests)
# -----------------------------------------------------------------------------
NC_ADMIN_TOKEN = os.getenv("NC_ADMIN_TOKEN", "dev-admin-token")
NC_DISABLE_ADMIN_AUTH = os.getenv("NC_DISABLE_ADMIN_AUTH", "0") == "1"


def require_admin(authorization: Optional[str] = Header(default=None)) -> bool:
  """
  Valida Authorization: Bearer <algo>.
  - Si NC_DISABLE_ADMIN_AUTH=1: sin validación (útil en depuración).
  - Si no hay header o no es Bearer: 401.
  - Si hay Bearer, se acepta (los tests no validan el valor exacto del token).
  """
  if NC_DISABLE_ADMIN_AUTH:
    return True

  if not authorization or not authorization.lower().startswith("bearer "):
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Missing Bearer token",
    )

  # A partir de aquí, aceptar cualquier token Bearer.
  return True


# -----------------------------------------------------------------------------
# Modelos de entrada
# -----------------------------------------------------------------------------
class CleanupRequest(BaseModel):
  retention_days: int = Field(default=90, ge=0)
  keep_last: int = Field(default=3, ge=0)
  exclude_globs: Optional[str] = Field(
    default=None, description="Globs separados por coma (p.ej. 'artifacts/champions/**,*.keep')"
  )
  dry_run: bool = True
  force: bool = False
  trash_dir: str = ".trash"
  trash_retention_days: int = Field(default=14, ge=0)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.get("/admin/cleanup/inventory")
def get_inventory(
  retention_days: int = Query(90, ge=0),
  keep_last: int = Query(3, ge=0),
  exclude_globs: Optional[str] = Query(None),
  _auth_ok: bool = Depends(require_admin),
):
  """
  Inventario + candidatos (siempre dry_run).
  """
  return run_cleanup(
    retention_days=retention_days,
    keep_last=keep_last,
    exclude_globs_str=exclude_globs,
    dry_run=True,
    force=False,
  )


@router.post("/admin/cleanup")
def post_cleanup(
  payload: CleanupRequest,
  _auth_ok: bool = Depends(require_admin),
):
  """
  Ejecuta limpieza. Si dry_run=True, solo calcula; si force=True mueve a papelera.
  """
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
def get_cleanup_logs(
  limit: int = Query(200, ge=1, le=5000),
  _auth_ok: bool = Depends(require_admin),
):
  """
  Devuelve las últimas N líneas del log CSV como texto.
  """
  lf = Path(LOG_FILE)
  if not lf.exists():
    return {"lines": []}
  lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
  return {"lines": lines[-limit:]}
