import os
import sys
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Test bootstrap
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()

# Asegura imports sin depender del entorno del usuario (Makefile/CI)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

# IMPORTANT: setear NC_ARTIFACTS_DIR ANTES de importar el app/router/runs_io.
_ARTIFACTS_TMP = Path(tempfile.mkdtemp(prefix="neurocampus_artifacts_")).resolve()
os.environ["NC_ARTIFACTS_DIR"] = str(_ARTIFACTS_TMP)  # evitar setdefault (NC_ARTIFACTS_DIR preexistente)


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    """Directorio temporal de artifacts (session-scoped)."""
    return Path(os.environ["NC_ARTIFACTS_DIR"]).resolve()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient (session-scoped)."""
    from neurocampus.app.main import app  # import lazy para respetar env var
    return TestClient(app)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Limpieza best-effort del artifacts temp."""
    shutil.rmtree(_ARTIFACTS_TMP, ignore_errors=True)
