import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from neurocampus.app.main import app
from neurocampus.app.routers import datos as datos_router


def _mk_csv_bytes():
    csv = "periodo,codigo_materia,grupo,pregunta_1\n2020-1,ABC123,1,10\n2020-1,ABC999,2,20\n"
    return csv.encode("utf-8")


def test_upload_crea_archivo_y_201(tmp_path, monkeypatch):
    """
    Sube un CSV y verifica:
      - 201 Created
      - stored_as apunta a datasets/<periodo>.(parquet|csv)
      - el archivo existe físicamente en el tmp 'datasets'
    """
    # parchea la raíz del repo para que datasets/ viva en tmp
    def fake_repo_root():
        base = tmp_path / "repo"
        base.mkdir(parents=True, exist_ok=True)
        return base

    monkeypatch.setattr(datos_router, "_repo_root_from_here", fake_repo_root, raising=True)
    client = TestClient(app)

    files = {
        "file": ("data.csv", _mk_csv_bytes(), "text/csv")
    }
    data = {
        "periodo": "2020-1",
        "dataset_id": "2020-1",  # compat
        "overwrite": "false",
    }
    r = client.post("/datos/upload", files=files, data=data)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["dataset_id"] == "2020-1"
    assert isinstance(body["rows_ingested"], int) and body["rows_ingested"] > 0
    assert body["stored_as"].startswith("localfs://neurocampus/datasets/")

    # Verifica archivo generado en disco
    stored = body["stored_as"].split("localfs://neurocampus/")[-1]
    repo_root = fake_repo_root()
    fpath = repo_root / stored  # e.g. datasets/2020-1.parquet
    assert fpath.exists()


def test_upload_conflicto_409_sin_overwrite(tmp_path, monkeypatch):
    """
    Si ya existe datasets/2020-1.(parquet|csv) y overwrite=false → 409
    """
    def fake_repo_root():
        base = tmp_path / "repo"
        base.mkdir(parents=True, exist_ok=True)
        # pre-crear archivo para simular conflicto
        ds = base / "datasets"
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "2020-1.csv").write_text("x,y\n1,2\n")
        return base

    monkeypatch.setattr(datos_router, "_repo_root_from_here", fake_repo_root, raising=True)
    client = TestClient(app)

    files = {
        "file": ("data.csv", _mk_csv_bytes(), "text/csv")
    }
    data = {"periodo": "2020-1", "dataset_id": "2020-1", "overwrite": "false"}
    r = client.post("/datos/upload", files=files, data=data)
    assert r.status_code == 409, r.text
    body = r.json()
    assert "ya existe" in (body.get("detail") or "")
