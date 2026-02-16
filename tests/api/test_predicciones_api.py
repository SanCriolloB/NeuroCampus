from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurocampus.predictions.bundle import build_predictor_manifest, bundle_paths, write_json
from neurocampus.utils.paths import artifacts_dir


class DummyPickleModel:
    """Modelo mínimo (picklable) para tests de inferencia P2.4.

    Implementa el subset de API que espera `predictions_service`:
    - predict_proba_df(df) -> np.ndarray (n, 3)
    - predict_df(df) -> List[str]

    Nota: se define a nivel de módulo para que pickle funcione en Windows.
    """

    labels = ["neg", "neu", "pos"]

    def predict_proba_df(self, df):
        import numpy as np

        n = int(len(df))
        proba = np.zeros((n, 3), dtype=float)
        proba[:, 1] = 1.0  # siempre 'neu'
        return proba

    def predict_df(self, df):
        return ["neu"] * int(len(df))


def _write_pickled_run_bundle(base: Path, *, run_id: str, dataset_id: str = "ds") -> Path:
    """Crea un run_dir con predictor.json + model.bin pickled (listo para inferencia)."""
    import pickle

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
        extra={"family": "sentiment_desempeno"},
    )
    write_json(bp.predictor_json, manifest)
    write_json(bp.preprocess_json, {"schema_version": 1, "notes": "test"})

    with open(bp.model_bin, "wb") as fh:
        pickle.dump(DummyPickleModel(), fh)

    return run_dir


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


def test_predicciones_model_info_by_run_id_ok(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_model_info_ok"
    _write_real_run_bundle(base, run_id=run_id, dataset_id="ds_model_info")

    r = client.get("/predicciones/model-info", params={"run_id": run_id})
    assert r.status_code == 200, r.text

    payload = r.json()
    assert payload["resolved_run_id"] == run_id
    assert payload["resolved_from"] == "run_id"
    assert "predictor" in payload
    assert "preprocess" in payload


def test_predicciones_model_info_by_champion_ok(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_model_info_champion_ok"
    dataset_id = "ds_model_info_champ"
    family = "sentiment_desempeno"

    _write_real_run_bundle(base, run_id=run_id, dataset_id=dataset_id, family=family)
    _write_champion(base, family=family, dataset_id=dataset_id, run_id=run_id)

    r = client.get(
        "/predicciones/model-info",
        params={"use_champion": True, "dataset_id": dataset_id, "family": family},
    )
    assert r.status_code == 200, r.text

    payload = r.json()
    assert payload["resolved_run_id"] == run_id
    assert payload["resolved_from"] == "champion"


def test_predicciones_model_info_champion_missing_source_run_id_422(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    dataset_id = "ds_model_info_missing_source"
    family = "sentiment_desempeno"

    champ = base / "champions" / family / dataset_id / "champion.json"
    champ.parent.mkdir(parents=True, exist_ok=True)
    champ.write_text(json.dumps({"note": "missing source_run_id"}, indent=2), encoding="utf-8")

    r = client.get(
        "/predicciones/model-info",
        params={"use_champion": True, "dataset_id": dataset_id, "family": family},
    )
    assert r.status_code == 422, r.text


def test_predicciones_model_info_run_not_found_404(client, artifacts_dir: Path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))

    run_id = "run_model_info_missing_bundle"
    r = client.get("/predicciones/model-info", params={"run_id": run_id})

    assert r.status_code == 404, r.text


def test_predicciones_cache_isolated_by_artifacts_dir(client, artifacts_dir: Path, monkeypatch):
    """El cache LRU debe aislarse por NC_ARTIFACTS_DIR.

    Si cambiamos NC_ARTIFACTS_DIR entre requests, no debe re-usar el bundle
    cacheado del directorio anterior.
    """

    # artifacts A
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base_a = artifacts_dir

    run_id = "run_cache_isolation"
    _write_real_run_bundle(base_a, run_id=run_id, dataset_id="ds_cache")

    r1 = client.post("/predicciones/predict", json={"run_id": run_id})
    assert r1.status_code == 200, r1.text

    # artifacts B (vacío)
    base_b = artifacts_dir.parent / "artifacts_b"
    base_b.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(base_b))

    r2 = client.post("/predicciones/predict", json={"run_id": run_id})
    assert r2.status_code == 404, r2.text


def test_predicciones_predict_feature_pack_inference_ok(client, artifacts_dir: Path, monkeypatch):
    """Smoke test P2.4: inferencia opt-in desde feature_pack (train_matrix.parquet)."""

    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_pickle_infer"
    dataset_id = "ds_infer"

    _write_pickled_run_bundle(base, run_id=run_id, dataset_id=dataset_id)

    # Crear feature_pack mínimo
    feat_dir = base / "features" / dataset_id
    feat_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(
        {
            "teacher_id": [1, 2, 3],
            "materia_id": [10, 20, 30],
            **{f"calif_{i+1}": [1, 2, 3] for i in range(10)},
        }
    )
    df.to_parquet(feat_dir / "train_matrix.parquet", index=False)
    (feat_dir / "meta.json").write_text(json.dumps({"dataset_id": dataset_id}, indent=2), encoding="utf-8")

    r = client.post(
        "/predicciones/predict",
        json={
            "run_id": run_id,
            "do_inference": True,
            "input_uri": "feature_pack",
            "limit": 2,
            "offset": 1,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["resolved_run_id"] == run_id
    assert body.get("predictions") is not None
    assert len(body["predictions"]) == 2

    # DummyPickleModel siempre retorna 'neu'
    assert body["predictions"][0]["label"] == "neu"
    assert "proba" in body["predictions"][0]


def test_predicciones_predict_feature_pack_inference_persist_ok(client, artifacts_dir: Path, monkeypatch):
    """P2.4-C: persistencia opt-in de predictions.parquet."""

    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_pickle_infer_persist"
    dataset_id = "ds_infer_persist"
    family = "sentiment_desempeno"

    _write_pickled_run_bundle(base, run_id=run_id, dataset_id=dataset_id)

    # Crear feature_pack mínimo
    feat_dir = base / "features" / dataset_id
    feat_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(
        {
            "teacher_id": [1, 2, 3],
            "materia_id": [10, 20, 30],
            **{f"calif_{i+1}": [1, 2, 3] for i in range(10)},
        }
    )
    df.to_parquet(feat_dir / "train_matrix.parquet", index=False)
    (feat_dir / "meta.json").write_text(json.dumps({"dataset_id": dataset_id}, indent=2), encoding="utf-8")

    r = client.post(
        "/predicciones/predict",
        json={
            "run_id": run_id,
            "do_inference": True,
            "input_uri": "feature_pack",
            "persist": True,
            "family": family,
            "limit": 1,
            "offset": 0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body.get("predictions") is not None
    assert len(body["predictions"]) == 1

    uri = body.get("predictions_uri")
    assert uri, body

    # Verificar en disco que se escribió el parquet
    from neurocampus.utils.paths import abs_artifact_path

    parquet_path = abs_artifact_path(uri)
    assert parquet_path.exists(), f"No existe: {parquet_path} (uri={uri})"

    out_df = pd.read_parquet(parquet_path)
    assert len(out_df) == 1
    assert "label" in out_df.columns

    # schema.json opcional (pero recomendado) debe existir
    schema_path = parquet_path.parent / "schema.json"
    assert schema_path.exists(), f"No existe schema.json: {schema_path}"



def test_predicciones_predict_persist_requires_inference_422(client, artifacts_dir: Path, monkeypatch):
    """Si persist=true pero do_inference=false, debe ser 422."""

    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(artifacts_dir))
    base = artifacts_dir

    run_id = "run_test_pickle_persist_requires_infer"
    dataset_id = "ds_persist_requires_infer"

    _write_pickled_run_bundle(base, run_id=run_id, dataset_id=dataset_id)

    r = client.post(
        "/predicciones/predict",
        json={
            "run_id": run_id,
            "persist": True,
        },
    )
    assert r.status_code == 422, r.text
