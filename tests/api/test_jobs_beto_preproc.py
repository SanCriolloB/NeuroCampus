# backend/tests/api/test_jobs_beto_preproc.py
from fastapi.testclient import TestClient
from neurocampus.app.main import app
from pathlib import Path

client = TestClient(app)

def test_launch_beto_preproc_missing_dataset(tmp_path, monkeypatch):
    # Forzamos DATA_PROCESSED_DIR a un tmp vacío mediante monkeypatch
    from neurocampus.app.routers import jobs as jobs_router

    monkeypatch.setattr(jobs_router, "DATA_PROCESSED_DIR", tmp_path)

    resp = client.post("/jobs/preproc/beto/run", json={"dataset": "no_existe"})
    assert resp.status_code == 400
    body = resp.json()
    assert "No existe dataset procesado" in body["detail"]
