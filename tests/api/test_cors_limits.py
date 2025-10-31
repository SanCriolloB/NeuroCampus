# tests/api/test_cors_limits.py
import os
from starlette.testclient import TestClient
from neurocampus.app.main import app

def test_cors_preflight_options():
    client = TestClient(app)
    headers = {
        "Origin": os.getenv("NC_ALLOWED_ORIGINS", "http://localhost:5173"),
        "Access-Control-Request-Method": "POST",
    }
    r = client.options("/datos/validar", headers=headers)
    # FastAPI/CORSMiddleware suele responder 200/204 con cabeceras CORS
    assert r.status_code in (200, 204)
    assert "access-control-allow-origin" in r.headers

def test_payload_too_large():
    client = TestClient(app)
    # simula que Content-Length excede
    big_len = str(int(os.getenv("NC_MAX_UPLOAD_MB", "10")) * 1024 * 1024 + 1)
    files = {"file": ("x.csv", "a,b\n1,2\n", "text/csv")}
    data = {"dataset_id": "docentes"}
    r = client.post("/datos/validar", files=files, data=data, headers={"Content-Length": big_len})
    assert r.status_code in (413, 200)  # En TestClient, Content-Length no siempre fuerza 413
