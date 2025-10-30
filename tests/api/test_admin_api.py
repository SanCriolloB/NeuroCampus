import os
import sys
from pathlib import Path

# Añadir backend/src al sys.path para importar neurocampus.app.main
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_SRC = TESTS_DIR.parent / "backend" / "src"
sys.path.append(str(BACKEND_SRC))

os.environ["NC_ADMIN_TOKEN"] = "testing-token"

from fastapi.testclient import TestClient
from neurocampus.app.main import app


def auth_headers():
    return {"Authorization": "Bearer testing-token"}


def test_inventory_requires_auth():
    client = TestClient(app)
    r = client.get("/admin/cleanup/inventory")
    assert r.status_code in (401, 503)

def test_inventory_ok(tmp_path):
    # Crear un archivo dummy para que haya al menos 1 candidato si retention_days=0
    p = Path("artifacts/modelZ/run_0")
    p.mkdir(parents=True, exist_ok=True)
    f = p / "dummy.bin"
    f.write_bytes(b"x" * 128)

    client = TestClient(app)
    r = client.get("/admin/cleanup/inventory?retention_days=0&keep_last=0", headers=auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data and "candidates" in data
    assert data["dry_run"] is True
    assert data["force"] is False
    assert data["summary"]["candidates_count"] >= 1

def test_post_cleanup_force_moves_to_trash():
    client = TestClient(app)
    payload = {
        "retention_days": 0,
        "keep_last": 0,
        "dry_run": False,
        "force": True,
        "trash_dir": ".trash",
        "trash_retention_days": 1
    }
    r = client.post("/admin/cleanup", headers=auth_headers(), json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["dry_run"] is False and data["force"] is True
    assert "moved_bytes" in data
