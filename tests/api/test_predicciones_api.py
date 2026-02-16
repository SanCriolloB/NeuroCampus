from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurocampus.predictions.bundle import build_predictor_manifest, bundle_paths, write_json
from neurocampus.utils.paths import artifacts_dir


def _write_real_run_bundle(base: Path, *, run_id: str, dataset_id: str = "ds", family: str = "sentiment_desempeno") -> Path:
    """Crea un run_dir con predictor.json + model.bin 'real' (no placeholder)."""
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bp = bundle_paths(run_dir)

    manifest = build_predictor_manifest(
        run_id=run_id,
        dataset_id=dataset_id,
        model_name="rbm_general",
        task_type="classification",
        input_level="row",
        target_col="y_sentimiento",
        extra={"family": family},
    )
    write_json(bp.predictor_json, manifest)
    write_json(bp.preprocess_json, {"schema_version": 1, "notes": "test"})
    bp.model_bin.write_bytes(b"REAL_MODEL_BYTES_v1")

    return run_dir


def _write_placeholder_run_bundle(base: Path, *, run_id: str) -> Path:
    """Crea un run_dir con model.bin placeholder (P2.1)."""
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bp = bundle_paths(run_dir)

    manifest = build_predictor_manifest(
        run_id=run_id,
        dataset_id="ds",
        model_name="rbm_general",
        task_type="classification",
        input_level="row",
        target_col="y",
    )
    write_json(bp.predictor_json, manifest)
    bp.model_bin.write_bytes(b"PLACEHOLDER_MODEL_BIN_P2_1")
    return run_dir


def _write_champion(base: Path, *, family: str, dataset_id: str, run_id: str) -> Path:
    champ = base / "champions" / family / dataset_id / "champion.json"
    champ.parent.mkdir(parents=True, exist_ok=True)
    champ.write_text(json.dumps({"source_run_id": run_id}, indent=2), encoding="utf-8")
    return champ


def test_predicciones_health_ok(client):
    r = client.get("/predicciones/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert "artifacts_dir" in data


def test_predicciones_predict_by_run_id_ok(client, artifacts_dir: Path, monkeypatch):
    # Asegura que el router use este artifacts_dir
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_real"
    _write_real_run_bundle(base, run_id=run_id, dataset_id="ds_api")

    r = client.post("/predicciones/predict", json={"run_id": run_id})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["resolved_from"] == "run_id"
    assert body["resolved_run_id"] == run_id
    assert body["predictor"]["run_id"] == run_id


def test_predicciones_predict_by_champion_ok(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_real_champ"
    dataset_id = "ds_champ"
    family = "sentiment_desempeno"

    _write_real_run_bundle(base, run_id=run_id, dataset_id=dataset_id, family=family)
    _write_champion(base, family=family, dataset_id=dataset_id, run_id=run_id)

    r = client.post(
        "/predicciones/predict",
        json={"use_champion": True, "dataset_id": dataset_id, "family": family},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["resolved_from"] == "champion"
    assert body["resolved_run_id"] == run_id


def test_predicciones_predict_champion_not_found_404(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))

    r = client.post(
        "/predicciones/predict",
        json={"use_champion": True, "dataset_id": "missing_ds", "family": "sentiment_desempeno"},
    )
    assert r.status_code == 404, r.text


def test_predicciones_predict_placeholder_422(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_placeholder"
    _write_placeholder_run_bundle(base, run_id=run_id)

    r = client.post("/predicciones/predict", json={"run_id": run_id})
    assert r.status_code == 422, r.text


def test_predicciones_predict_run_not_found_404(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))

    run_id = "run_missing_bundle"
    r = client.post("/predicciones/predict", json={"run_id": run_id})

    assert r.status_code == 404, r.text
    detail = r.json().get("detail", "")
    assert run_id in detail


def test_predicciones_predict_champion_points_to_missing_run_404(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_missing_bundle_champ"
    dataset_id = "ds_champ_missing_bundle"
    family = "sentiment_desempeno"

    _write_champion(base, family=family, dataset_id=dataset_id, run_id=run_id)

    r = client.post(
        "/predicciones/predict",
        json={"use_champion": True, "dataset_id": dataset_id, "family": family},
    )

    assert r.status_code == 404, r.text
    detail = r.json().get("detail", "")
    assert run_id in detail


def test_predicciones_predict_champion_missing_source_run_id_422(client, artifacts_dir: Path, monkeypatch):
    """Si champion.json existe pero no incluye source_run_id, debe ser 422 (PredictorNotReadyError)."""
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    dataset_id = "ds_champ_missing_source_run_id"
    family = "sentiment_desempeno"

    champ = base / "champions" / family / dataset_id / "champion.json"
    champ.parent.mkdir(parents=True, exist_ok=True)
    champ.write_text(json.dumps({"note": "missing source_run_id"}, indent=2), encoding="utf-8")

    r = client.post(
        "/predicciones/predict",
        json={"use_champion": True, "dataset_id": dataset_id, "family": family},
    )

    assert r.status_code == 422, r.text
