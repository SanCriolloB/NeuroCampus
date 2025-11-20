# backend/tests/api/test_datos_dashboard_api.py
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from neurocampus.app.main import app
from neurocampus.data import datos_dashboard as dd


def _prep_tmp_dataset(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "processed").mkdir(parents=True)
    (data_root / "labeled").mkdir(parents=True)

    df_proc = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-2"],
            "docente": ["Alice", "Bob", "Alice"],
            "asignatura": ["MAT101", "MAT101", "FIS201"],
            "nota": [4.5, 3.8, 4.0],
            "comentario": ["x", "y", "z"],
        }
    )
    df_lab = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-2", "2024-2"],
            "docente": ["Alice", "Alice", "Bob", "Bob"],
            "asignatura": ["MAT101", "MAT101", "MAT101", "FIS201"],
            "comentario": ["x", "y", "z", "w"],
            "sentiment_label_teacher": ["pos", "neu", "neg", "pos"],
        }
    )

    ds_id = "demo_ds"
    df_proc.to_parquet(data_root / "processed" / f"{ds_id}.parquet")
    df_lab.to_parquet(data_root / "labeled" / f"{ds_id}_teacher.parquet")
    return ds_id


def test_resumen_y_sentimientos_ok(tmp_path, monkeypatch):
    # redirigir repo_root a tmp_path para este test
    def fake_repo_root():
        return tmp_path

    monkeypatch.setattr(dd, "_repo_root_from_here", fake_repo_root, raising=True)

    dataset_id = _prep_tmp_dataset(tmp_path)

    client = TestClient(app)

    # /datos/resumen
    r = client.get("/datos/resumen", params={"dataset": dataset_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset_id"] == dataset_id
    assert body["n_rows"] == 3
    assert body["n_cols"] >= 4

    # /datos/sentimientos
    r2 = client.get("/datos/sentimientos", params={"dataset": dataset_id})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["dataset_id"] == dataset_id
    assert body2["total_comentarios"] == 4

    labels = {item["label"]: item["count"] for item in body2["global_counts"]}
    assert labels["pos"] == 2
    assert labels["neu"] == 1
    assert labels["neg"] == 1


def test_resumen_not_found(tmp_path, monkeypatch):
    def fake_repo_root():
        return tmp_path

    monkeypatch.setattr(dd, "_repo_root_from_here", fake_repo_root, raising=True)
    client = TestClient(app)

    r = client.get("/datos/resumen", params={"dataset": "no_existe"})
    assert r.status_code == 404


def test_sentimientos_not_found(tmp_path, monkeypatch):
    def fake_repo_root():
        return tmp_path

    monkeypatch.setattr(dd, "_repo_root_from_here", fake_repo_root, raising=True)
    client = TestClient(app)

    r = client.get("/datos/sentimientos", params={"dataset": "no_existe"})
    assert r.status_code == 404
