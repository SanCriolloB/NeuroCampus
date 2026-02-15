from __future__ import annotations

from pathlib import Path
import time
import uuid

import pandas as pd
import pytest


def _make_minimal_dataset(tmp_path: Path, dataset_id: str) -> Path:
    """Crea un parquet mínimo compatible con feature-pack.

    Incluye:
    - docente/materia para mapping
    - rating para score_base
    - p_neg/p_neu/p_pos + has_text para score_total (y y_sentimiento)
    - periodo para coherencia con histórico
    """
    df = pd.DataFrame(
        {
            "periodo": [dataset_id] * 6,
            "docente": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "materia": ["M1", "M2", "M1", "M2", "M1", "M2"],
            "rating": [3, 4, 5, 2, 1, 4],
            "has_text": [1, 1, 1, 0, 1, 0],
            "p_neg": [0.1, 0.2, 0.05, 0.7, 0.8, 0.6],
            "p_neu": [0.2, 0.2, 0.10, 0.2, 0.1, 0.2],
            "p_pos": [0.7, 0.6, 0.85, 0.1, 0.1, 0.2],
        }
    )
    p = tmp_path / f"{dataset_id}.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def prepared_feature_pack(client, artifacts_dir: Path, tmp_path: Path) -> str:
    """Prepara feature-pack para un dataset_id único."""
    dataset_id = f"ds_test_{uuid.uuid4().hex[:6]}"
    src = _make_minimal_dataset(tmp_path, dataset_id)

    r = client.post(
        "/modelos/feature-pack/prepare",
        params={"dataset_id": dataset_id, "input_uri": str(src), "force": True},
    )
    assert r.status_code == 200, r.text

    # Validar artifacts en NC_ARTIFACTS_DIR
    fp_dir = artifacts_dir / "features" / dataset_id
    assert (fp_dir / "train_matrix.parquet").exists()
    assert (fp_dir / "meta.json").exists()
    assert (fp_dir / "pair_matrix.parquet").exists()
    assert (fp_dir / "pair_meta.json").exists()

    return dataset_id


def test_readiness_reports_feature_pack_and_score_col(client, prepared_feature_pack: str):
    dataset_id = prepared_feature_pack
    r = client.get("/modelos/readiness", params={"dataset_id": dataset_id})
    assert r.status_code == 200, r.text
    payload = r.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["feature_pack_exists"] is True
    assert payload["pair_matrix_exists"] is True
    assert payload["score_col"] is not None


def test_entrenar_estado_and_promote_contract(client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch):
    """Cubre:
    - POST /modelos/entrenar
    - GET /modelos/estado/{job_id}
    - POST /modelos/champion/promote (422/404/200)
    """
    from neurocampus.app.routers import modelos as m

    # Evitar entrenamiento pesado: parchea la plantilla para devolver métricas deterministas.
    class DummyStrategy:
        pass

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        return DummyStrategy()

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.1, "val_accuracy": 0.9},
                "history": [{"epoch": 1, "loss": 0.1}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)

    # (Opcional) evita side-effects de auto champion durante training
    monkeypatch.setattr(
        m,
        "maybe_update_champion",
        lambda **kwargs: {"promoted": False},
    )

    dataset_id = prepared_feature_pack

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert job_id

    # Compatibilidad de payload: aceptar model_name como alias de modelo
    r_alias = client.post(
        "/modelos/entrenar",
        json={
            "model_name": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
        },
    )
    assert r_alias.status_code == 200, r_alias.text
    assert r_alias.json().get("job_id")


    # Poll estado (debería completar casi inmediato con DummyTemplate)
    st = None
    for _ in range(50):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200, s.text
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st

    run_id = st["run_id"]
    assert run_id

    run_dir = artifacts_dir / "runs" / run_id
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.json").exists()

    # 422 si run_id inválido
    r422 = client.post(
        "/modelos/champion/promote",
        json={"dataset_id": dataset_id, "run_id": "null", "model_name": "rbm_general", "family": "sentiment_desempeno"},
    )
    assert r422.status_code == 422

    # 404 si no existe metrics.json para ese run_id
    r404 = client.post(
        "/modelos/champion/promote",
        json={"dataset_id": dataset_id, "run_id": "does_not_exist_123", "model_name": "rbm_general", "family": "sentiment_desempeno"},
    )
    assert r404.status_code == 404

    # 200 happy path
    r200 = client.post(
        "/modelos/champion/promote",
        json={"dataset_id": dataset_id, "run_id": run_id, "model_name": "rbm_general", "family": "sentiment_desempeno"},
    )
    assert r200.status_code == 200, r200.text

    champ_ds_dir = artifacts_dir / "champions" / "sentiment_desempeno" / dataset_id
    assert (champ_ds_dir / "champion.json").exists()
