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

    # --- P2.1: el API debe devolver contexto completo (sin null/unknown) ---
    # Champion
    r_ch = client.get(
        "/modelos/champion",
        params={"dataset_id": dataset_id, "family": "sentiment_desempeno", "model_name": "rbm_general"},
    )
    assert r_ch.status_code == 200, r_ch.text
    ch = r_ch.json()
    assert ch.get("dataset_id") == dataset_id
    assert str(ch.get("family") or "").lower() == "sentiment_desempeno"
    assert ch.get("model_name") in ("rbm_general", "rbm_restringida")
    assert ch.get("task_type") in ("classification", "regression")
    assert ch.get("input_level") in ("row", "pair")
    assert ch.get("target_col") not in (None, "unknown", "null", "")

    # Runs (listado)
    r_runs = client.get(
        "/modelos/runs",
        params={"dataset_id": dataset_id, "family": "sentiment_desempeno"},
    )
    assert r_runs.status_code == 200, r_runs.text
    runs = r_runs.json()
    assert isinstance(runs, list)
    row = next((x for x in runs if x.get("run_id") == run_id), None)
    assert row is not None, f"run_id {run_id} no encontrado en /modelos/runs"
    assert str(row.get("family") or "").lower() == "sentiment_desempeno"
    assert row.get("task_type") in ("classification", "regression")
    assert row.get("input_level") in ("row", "pair")
    assert row.get("target_col") not in (None, "unknown", "null", "")

    # Run details
    r_det = client.get(f"/modelos/runs/{run_id}")
    assert r_det.status_code == 200, r_det.text
    det = r_det.json()
    assert det.get("run_id") == run_id
    assert det.get("dataset_id") == dataset_id
    assert str(det.get("family") or "").lower() == "sentiment_desempeno"
    assert det.get("task_type") in ("classification", "regression")
    assert det.get("input_level") in ("row", "pair")
    assert det.get("target_col") not in (None, "unknown", "null", "")


# ============================================================================
# P2 – Parte 2: Tests de warm start RBM desde API
# ============================================================================

def _make_fake_model_dir(run_dir: Path) -> Path:
    """Crea un model/ mínimo válido para warm start en un run dir."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text('{"task_type":"classification"}', encoding="utf-8")
    (model_dir / "rbm.pt").write_bytes(b"\x00" * 8)  # peso falso
    return model_dir


def _make_base_run(
    client,
    artifacts_dir: Path,
    dataset_id: str,
    monkeypatch,
) -> str:
    """
    Entrena un run base (con DummyTemplate) y crea model/ mínimo.
    Devuelve run_id.
    """
    from neurocampus.app.routers import modelos as m

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
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "none",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # Poll
    st = None
    for _ in range(80):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        import time as _t; _t.sleep(0.01)

    assert st["status"] == "completed", st
    run_id = st["run_id"]
    assert run_id

    # Crear model/ mínimo para warm start
    run_dir = artifacts_dir / "runs" / run_id
    _make_fake_model_dir(run_dir)

    return run_id


def test_warm_start_errors_404_422(client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch):
    """
    Casos de error de warm start:
    - run_id inexistente → 404
    - champion inexistente → 404
    - warm_start_from=run_id sin warm_start_run_id → 422 (schema)
    - run existe pero sin model/ → 422
    """
    from neurocampus.utils.warm_start import resolve_warm_start_path

    # run inexistente → 404
    try:
        resolve_warm_start_path(
            artifacts_dir=artifacts_dir,
            dataset_id="ds_nonexistent",
            family="sentiment_desempeno",
            model_name="rbm_general",
            warm_start_from="run_id",
            warm_start_run_id="run_does_not_exist_xyz",
        )
        assert False, "Debería haber lanzado HTTPException 404"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404, f"Esperaba 404, got {exc}"

    # champion inexistente → 404
    try:
        resolve_warm_start_path(
            artifacts_dir=artifacts_dir,
            dataset_id="ds_no_champion_xyz",
            family="sentiment_desempeno",
            model_name="rbm_general",
            warm_start_from="champion",
        )
        assert False, "Debería haber lanzado HTTPException 404"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404, f"Esperaba 404, got {exc}"

    # run existe pero sin model/ → 422
    fake_run_id = f"run_nomodel_{uuid.uuid4().hex[:6]}"
    fake_run_dir = artifacts_dir / "runs" / fake_run_id
    fake_run_dir.mkdir(parents=True, exist_ok=True)
    (fake_run_dir / "metrics.json").write_text('{}', encoding="utf-8")

    try:
        resolve_warm_start_path(
            artifacts_dir=artifacts_dir,
            dataset_id="ds_any",
            family="sentiment_desempeno",
            model_name="rbm_general",
            warm_start_from="run_id",
            warm_start_run_id=fake_run_id,
        )
        assert False, "Debería haber lanzado HTTPException 422"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422, f"Esperaba 422, got {exc}"

    # warm_start_from=run_id sin warm_start_run_id → 422
    try:
        resolve_warm_start_path(
            artifacts_dir=artifacts_dir,
            dataset_id="ds_any",
            family="sentiment_desempeno",
            model_name="rbm_general",
            warm_start_from="run_id",
            warm_start_run_id=None,
        )
        assert False, "Debería haber lanzado HTTPException 422"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422, f"Esperaba 422, got {exc}"


def test_warm_start_run_id_ok_and_trace(
    client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch
):
    """
    Warm start por run_id:
    1. Entrena run base (cold start) + crea model/ mínimo.
    2. Entrena nuevo run con warm_start_from=run_id.
    3. Verifica trazabilidad en metrics.json del nuevo run.
    """
    dataset_id = prepared_feature_pack

    base_run_id = _make_base_run(client, artifacts_dir, dataset_id, monkeypatch)

    # Ahora entrenar con warm start por run_id
    from neurocampus.app.routers import modelos as m

    class DummyStrategy:
        pass

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        # Verificar que warm_start_path llegó a los hparams
        assert "warm_start_path" in hparams, (
            f"warm_start_path no llegó a hparams: {list(hparams)}"
        )
        return DummyStrategy()

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.05, "val_accuracy": 0.92},
                "history": [{"epoch": 1, "loss": 0.05}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "run_id",
            "warm_start_run_id": base_run_id,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    import time as _t
    st = None
    for _ in range(80):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        _t.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st

    new_run_id = st["run_id"]
    assert new_run_id
    assert new_run_id != base_run_id

    # Verificar trazabilidad en metrics.json
    import json as _json
    metrics_path = artifacts_dir / "runs" / new_run_id / "metrics.json"
    assert metrics_path.exists(), "metrics.json debe existir"
    metrics = _json.loads(metrics_path.read_text())

    assert metrics.get("warm_started") is True, f"warm_started no es True: {metrics}"
    assert metrics.get("warm_start_from") == "run_id", metrics
    assert metrics.get("warm_start_source_run_id") == base_run_id, metrics
    assert "warm_start_path" in metrics, metrics

    # Verificar trazabilidad en el estado del job
    trace = st.get("warm_start_trace", {})
    assert trace.get("warm_started") is True, f"warm_start_trace incorrecto: {trace}"
    assert trace.get("warm_start_source_run_id") == base_run_id


def test_warm_start_champion_ok_and_trace(
    client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch
):
    """
    Warm start por champion:
    1. Entrena run base + crea model/ + promueve a champion.
    2. Entrena nuevo run con warm_start_from=champion.
    3. Verifica trazabilidad en metrics.json.
    """
    dataset_id = prepared_feature_pack

    base_run_id = _make_base_run(client, artifacts_dir, dataset_id, monkeypatch)

    # Promover base_run a champion
    r_prom = client.post(
        "/modelos/champion/promote",
        json={
            "dataset_id": dataset_id,
            "run_id": base_run_id,
            "model_name": "rbm_general",
            "family": "sentiment_desempeno",
        },
    )
    assert r_prom.status_code == 200, r_prom.text

    # Verificar que champion.json existe y tiene source_run_id
    champ_path = artifacts_dir / "champions" / "sentiment_desempeno" / dataset_id / "champion.json"
    assert champ_path.exists(), "champion.json debe existir tras promote"
    import json as _json
    champ_data = _json.loads(champ_path.read_text())
    assert champ_data.get("source_run_id") == base_run_id, champ_data

    # Ahora entrenar con warm start por champion
    from neurocampus.app.routers import modelos as m

    class DummyStrategy:
        pass

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        assert "warm_start_path" in hparams, (
            f"warm_start_path no llegó a hparams en warm-start champion: {list(hparams)}"
        )
        return DummyStrategy()

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.04, "val_accuracy": 0.95},
                "history": [{"epoch": 1, "loss": 0.04}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "champion",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    import time as _t
    st = None
    for _ in range(80):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        _t.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st

    new_run_id = st["run_id"]
    assert new_run_id

    metrics_path = artifacts_dir / "runs" / new_run_id / "metrics.json"
    assert metrics_path.exists()
    metrics = _json.loads(metrics_path.read_text())

    assert metrics.get("warm_started") is True, metrics
    assert metrics.get("warm_start_from") == "champion", metrics
    assert metrics.get("warm_start_source_run_id") == base_run_id, metrics
    assert "warm_start_path" in metrics, metrics


def test_warm_start_none_leaves_no_trace(
    client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch
):
    """
    warm_start_from=none → warm_started=False en metrics, sin warm_start_path.
    """
    dataset_id = prepared_feature_pack

    from neurocampus.app.routers import modelos as m

    class DummyStrategy:
        pass

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        assert "warm_start_path" not in hparams, (
            f"warm_start_path NO debería estar en hparams con warm_start_from=none: {hparams}"
        )
        return DummyStrategy()

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.1},
                "history": [],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "rbm_general",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "none",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    import time as _t
    st = None
    for _ in range(80):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        _t.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st

    new_run_id = st["run_id"]
    import json as _json
    metrics = _json.loads((artifacts_dir / "runs" / new_run_id / "metrics.json").read_text())
    assert metrics.get("warm_started") is False, metrics
    assert "warm_start_path" not in metrics


# ============================================================================
# P2 – Parte 3: Tests de warm start y persistencia DBM
# ============================================================================

def _make_base_dbm_run(
    client,
    artifacts_dir: Path,
    dataset_id: str,
    monkeypatch,
) -> str:
    """
    Entrena un run DBM real (DBMManualPlantillaStrategy sin mock de estrategia),
    verifica que model/ quede con dbm_state.npz + meta.json,
    y devuelve el run_id.
    """
    from neurocampus.app.routers import modelos as m

    # Solo mockear PlantillaEntrenamiento para no depender de pair_matrix real,
    # pero dejar que la estrategia real se cree para que save() funcione.
    import numpy as np
    from neurocampus.models.strategies.dbm_manual_strategy import DBMManualPlantillaStrategy

    _captured_strategy = {}

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        s = DBMManualPlantillaStrategy()
        _captured_strategy["s"] = s
        return s

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            s = self.estrategia
            # Inicializar manualmente el modelo DBM con dims mínimas
            import numpy as _np
            s.feat_cols_ = [f"f{i}" for i in range(4)]
            s.task_type_ = "unsupervised"
            from neurocampus.models.dbm_manual import DBMManual
            s.model = DBMManual(n_visible=4, n_hidden1=3, n_hidden2=2, seed=7)
            s.X = _np.random.rand(10, 4).astype("f4")
            s._warm_start_info_ = {"warm_start": "skipped"}
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.3, "recon_error": 0.3},
                "history": [{"epoch": 1, "loss": 0.3}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "dbm_manual",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "none",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    import time as _t
    st = None
    for _ in range(80):
        s = client.get(f"/modelos/estado/{job_id}")
        assert s.status_code == 200
        st = s.json()
        if st["status"] in ("completed", "failed"):
            break
        _t.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st
    run_id = st["run_id"]
    assert run_id

    # Verificar que model/ tiene los archivos esperados de DBM
    model_dir = artifacts_dir / "runs" / run_id / "model"
    assert model_dir.exists(), f"model/ no creado en {model_dir}"
    assert (model_dir / "dbm_state.npz").exists(), "dbm_state.npz falta"
    assert (model_dir / "meta.json").exists(), "meta.json falta"

    return run_id


def test_dbm_model_dir_created_after_training(
    client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch
):
    """
    Al completar un run DBM, model/ debe contener dbm_state.npz y meta.json.
    """
    dataset_id = prepared_feature_pack
    run_id = _make_base_dbm_run(client, artifacts_dir, dataset_id, monkeypatch)
    # La verificación ya ocurre dentro de _make_base_dbm_run.
    # Adicional: verificar contenido de meta.json
    import json as _json
    meta = _json.loads(
        (artifacts_dir / "runs" / run_id / "model" / "meta.json").read_text()
    )
    assert meta.get("n_visible") == 4
    assert meta.get("n_hidden1") == 3
    assert meta.get("n_hidden2") == 2
    assert "feat_cols_" in meta


def test_dbm_warm_start_run_id_ok_and_trace(
    client, artifacts_dir: Path, prepared_feature_pack: str, monkeypatch
):
    """
    Warm start DBM por run_id:
    1. Entrenar run base (DBM) → genera model/dbm_state.npz.
    2. Entrenar nuevo run con warm_start_from=run_id.
    3. Verificar trazabilidad en metrics.json.
    """
    dataset_id = prepared_feature_pack
    base_run_id = _make_base_dbm_run(client, artifacts_dir, dataset_id, monkeypatch)

    # Segundo entrenamiento con warm start
    from neurocampus.app.routers import modelos as m
    from neurocampus.models.strategies.dbm_manual_strategy import DBMManualPlantillaStrategy

    _ws_path_received = {}

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        _ws_path_received["path"] = hparams.get("warm_start_path")
        s = DBMManualPlantillaStrategy()
        return s

    class DummyTemplate2:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            s = self.estrategia
            import numpy as _np
            from neurocampus.models.dbm_manual import DBMManual
            # Inicializar modelo compatible (mismas dims que el run base)
            s.feat_cols_ = [f"f{i}" for i in range(4)]
            s.task_type_ = "unsupervised"
            s.model = DBMManual(n_visible=4, n_hidden1=3, n_hidden2=2, seed=99)
            s.X = _np.random.rand(10, 4).astype("f4")

            # Simular que warm start fue ok
            ws_path = hparams.get("warm_start_path", "")
            if ws_path:
                try:
                    prev = DBMManual.load(ws_path)
                    s.model.copy_weights_from(prev)
                    s._warm_start_info_ = {"warm_start": "ok", "warm_start_dir": ws_path}
                except Exception as exc:
                    s._warm_start_info_ = {"warm_start": "error", "error": str(exc)}
            else:
                s._warm_start_info_ = {"warm_start": "skipped"}

            return {
                "status": "completed",
                "model": model_name,
                "metrics": {
                    "loss": 0.2,
                    "recon_error": 0.2,
                    "warm_start": dict(s._warm_start_info_),
                },
                "history": [{"epoch": 1, "loss": 0.2}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate2)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kwargs: {"promoted": False})

    r = client.post(
        "/modelos/entrenar",
        json={
            "modelo": "dbm_manual",
            "dataset_id": dataset_id,
            "family": "sentiment_desempeno",
            "epochs": 1,
            "data_source": "feature_pack",
            "auto_prepare": False,
            "warm_start_from": "run_id",
            "warm_start_run_id": base_run_id,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    import time as _t
    st = None
    for _ in range(80):
        s2 = client.get(f"/modelos/estado/{job_id}")
        assert s2.status_code == 200
        st = s2.json()
        if st["status"] in ("completed", "failed"):
            break
        _t.sleep(0.01)

    assert st is not None
    assert st["status"] == "completed", st

    new_run_id = st["run_id"]
    assert new_run_id
    assert new_run_id != base_run_id

    # warm_start_path llegó al strategy
    assert _ws_path_received.get("path"), (
        f"warm_start_path no llegó al strategy: {_ws_path_received}"
    )

    # Trazabilidad en metrics.json
    import json as _json
    metrics = _json.loads(
        (artifacts_dir / "runs" / new_run_id / "metrics.json").read_text()
    )
    assert metrics.get("warm_started") is True, metrics
    assert metrics.get("warm_start_from") == "run_id", metrics
    assert metrics.get("warm_start_source_run_id") == base_run_id, metrics
    assert "warm_start_path" in metrics, metrics

    # warm_start_trace en el estado del job
    trace = st.get("warm_start_trace", {})
    assert trace.get("warm_started") is True, f"warm_start_trace: {trace}"
    assert trace.get("warm_start_source_run_id") == base_run_id


def test_dbm_warm_start_validates_dbm_files(artifacts_dir: Path):
    """
    resolve_warm_start_path acepta model/ con dbm_state.npz (DBM)
    y rechaza con 422 si falta dbm_state.npz o meta.json.
    """
    from neurocampus.utils.warm_start import resolve_warm_start_path
    import numpy as _np

    # Crear run con model/ válido para DBM
    valid_run_id = f"run_dbm_valid_{uuid.uuid4().hex[:6]}"
    valid_model_dir = artifacts_dir / "runs" / valid_run_id / "model"
    valid_model_dir.mkdir(parents=True, exist_ok=True)
    (valid_model_dir / "meta.json").write_text('{"n_visible":4}', encoding="utf-8")
    _np.savez(valid_model_dir / "dbm_state.npz", W1=_np.zeros((4, 3), dtype="f4"))

    path, trace = resolve_warm_start_path(
        artifacts_dir=artifacts_dir,
        dataset_id="ds_any",
        family="sentiment_desempeno",
        model_name="dbm_manual",
        warm_start_from="run_id",
        warm_start_run_id=valid_run_id,
    )
    assert path is not None
    assert trace["warm_started"] is True

    # Crear run con model/ que solo tiene meta.json (sin dbm_state.npz ni pesos RBM)
    invalid_run_id = f"run_dbm_invalid_{uuid.uuid4().hex[:6]}"
    invalid_model_dir = artifacts_dir / "runs" / invalid_run_id / "model"
    invalid_model_dir.mkdir(parents=True, exist_ok=True)
    (invalid_model_dir / "meta.json").write_text('{"n_visible":4}', encoding="utf-8")
    # Sin dbm_state.npz → debe fallar con 422

    try:
        resolve_warm_start_path(
            artifacts_dir=artifacts_dir,
            dataset_id="ds_any",
            family="sentiment_desempeno",
            model_name="dbm_manual",
            warm_start_from="run_id",
            warm_start_run_id=invalid_run_id,
        )
        assert False, "Debería haber lanzado HTTPException 422"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422, f"Esperaba 422, got: {exc}"


# ===========================================================================
# P2 Parte 5 — Tests del sweep determinístico (POST /modelos/sweep)
# ===========================================================================

def _make_sweep_request(dataset_id: str, **overrides) -> dict:
    """Payload base para el sweep."""
    base = {
        "dataset_id": dataset_id,
        "family": "score_docente",
        "data_source": "pair_matrix",
        "seed": 42,
        "epochs": 1,
        "auto_prepare": False,
        "models": ["rbm_general", "rbm_restringida", "dbm_manual"],
        "base_hparams": {"n_hidden": 4, "n_hidden1": 3, "n_hidden2": 2},
        "auto_promote_champion": False,
        "warm_start_from": "none",
    }
    base.update(overrides)
    return base


def _make_sweep_mocks(monkeypatch, fake_metrics_by_model: dict):
    """
    Monkeypatches para que el sweep no necesite datos reales.

    fake_metrics_by_model: {model_name: {primary_metric_value: X, ...}}
    """
    import time as _time
    from neurocampus.app.routers import modelos as m
    from neurocampus.models.utils.metrics_contract import standardize_run_metrics

    call_count = {"n": 0}

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family, **kw):
        from neurocampus.models.strategies.dbm_manual_strategy import DBMManualPlantillaStrategy
        s = DBMManualPlantillaStrategy()
        return s

    class DummyTemplateSweep:
        def __init__(self, estrategia):
            self.estrategia = estrategia
            self._model_name = None  # se rellena en run()

        def run(self, *, data_ref, epochs, hparams, model_name):
            call_count["n"] += 1
            raw = fake_metrics_by_model.get(str(model_name), {"loss": 0.5})
            return {
                "status": "completed",
                "model": model_name,
                "metrics": raw,
                "history": [{"epoch": 1, "loss": raw.get("loss", 0.5)}],
            }

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)
    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplateSweep)
    monkeypatch.setattr(m, "maybe_update_champion", lambda **kw: {"promoted": True})

    return call_count


# ---------------------------------------------------------------------------
# Test 1: sweep score_docente → 3 candidatos + best por val_rmse
# ---------------------------------------------------------------------------

def test_sweep_score_docente_3_candidates(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """
    Sweep score_docente con 3 modelos: verifica estructura de respuesta
    y que best tiene el menor val_rmse.
    """
    dataset_id = prepared_feature_pack

    fake_metrics = {
        "rbm_general":    {"val_rmse": 6.5,  "val_mae": 4.0, "task_type": "regression"},
        "rbm_restringida":{"val_rmse": 7.2,  "val_mae": 4.5, "task_type": "regression"},
        "dbm_manual":     {"val_rmse": 8.0,  "val_mae": 5.0, "task_type": "regression"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(dataset_id),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    # Estructura mínima
    assert data["sweep_id"]
    assert data["status"] == "completed"
    assert data["family"] == "score_docente"
    assert data["primary_metric"] == "val_rmse"
    assert data["primary_metric_mode"] == "min"

    # 3 candidatos
    assert len(data["candidates"]) == 3
    models_found = {c["model_name"] for c in data["candidates"]}
    assert models_found == {"rbm_general", "rbm_restringida", "dbm_manual"}

    # Best = rbm_general (menor val_rmse = 6.5)
    assert data["best"] is not None
    assert data["best"]["model_name"] == "rbm_general"

    # n_completed
    assert data["n_completed"] == 3
    assert data["n_failed"] == 0


# ---------------------------------------------------------------------------
# Test 2: sweep sentiment_desempeno → best por val_f1_macro
# ---------------------------------------------------------------------------

def test_sweep_sentiment_3_candidates_best_by_f1(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """
    Sweep sentiment_desempeno: best es el que tiene mayor val_f1_macro.
    """
    dataset_id = prepared_feature_pack

    fake_metrics = {
        "rbm_general":    {"val_f1_macro": 0.72, "val_accuracy": 0.75, "task_type": "classification"},
        "rbm_restringida":{"val_f1_macro": 0.80, "val_accuracy": 0.82, "task_type": "classification"},
        "dbm_manual":     {"val_f1_macro": 0.65, "val_accuracy": 0.70, "task_type": "classification"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(
            dataset_id,
            family="sentiment_desempeno",
            data_source="feature_pack",
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["primary_metric"] == "val_f1_macro"
    assert data["primary_metric_mode"] == "max"
    assert data["best"]["model_name"] == "rbm_restringida"  # mayor f1 = 0.80


# ---------------------------------------------------------------------------
# Test 3: determinismo — empate en primary_metric_value
# ---------------------------------------------------------------------------

def test_sweep_deterministic_tiebreaker(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """
    Cuando 2 modelos tienen el mismo primary_metric_value,
    el tie-breaker es el orden canónico de _SWEEP_MODEL_ORDER.
    rbm_general < rbm_restringida < dbm_manual → rbm_general gana el empate.
    """
    dataset_id = prepared_feature_pack

    # rbm_general y rbm_restringida empatan en val_rmse
    fake_metrics = {
        "rbm_general":    {"val_rmse": 5.0, "task_type": "regression"},
        "rbm_restringida":{"val_rmse": 5.0, "task_type": "regression"},
        "dbm_manual":     {"val_rmse": 9.0, "task_type": "regression"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(dataset_id),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    # En empate de métrica, rbm_general gana (primero en _SWEEP_MODEL_ORDER)
    assert data["best"]["model_name"] == "rbm_general"


# ---------------------------------------------------------------------------
# Test 4: auto_promote_champion=True → champion_promoted=True
# ---------------------------------------------------------------------------

def test_sweep_auto_promote_champion(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """
    Con auto_promote_champion=True, el sweep promueve el best a champion
    y devuelve champion_promoted=True + champion_run_id.
    """
    dataset_id = prepared_feature_pack

    fake_metrics = {
        "rbm_general":    {"val_rmse": 5.0, "task_type": "regression"},
        "rbm_restringida":{"val_rmse": 7.0, "task_type": "regression"},
        "dbm_manual":     {"val_rmse": 8.0, "task_type": "regression"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(dataset_id, auto_promote_champion=True),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["champion_promoted"] is True
    assert data["champion_run_id"] is not None
    assert data["best"]["model_name"] == "rbm_general"


# ---------------------------------------------------------------------------
# Test 5: modelo no permitido → 422
# ---------------------------------------------------------------------------

def test_sweep_invalid_model_422(client, prepared_feature_pack: str):
    """Modelo no soportado devuelve 422."""
    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(
            prepared_feature_pack,
            models=["rbm_general", "modelo_inexistente"],
        ),
    )
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Test 6: falta dataset_id → 422
# ---------------------------------------------------------------------------

def test_sweep_missing_dataset_id_422(client):
    """dataset_id vacío devuelve 422."""
    r = client.post(
        "/modelos/sweep",
        json={
            "dataset_id": "",
            "family": "score_docente",
            "models": ["rbm_general"],
        },
    )
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Test 7: subset de modelos → solo esos candidatos
# ---------------------------------------------------------------------------

def test_sweep_subset_models(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """Sweep con solo 2 modelos devuelve exactamente 2 candidatos."""
    dataset_id = prepared_feature_pack

    fake_metrics = {
        "rbm_general":    {"val_rmse": 6.0, "task_type": "regression"},
        "rbm_restringida":{"val_rmse": 7.0, "task_type": "regression"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(
            dataset_id,
            models=["rbm_general", "rbm_restringida"],
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    assert len(data["candidates"]) == 2
    assert data["best"]["model_name"] == "rbm_general"


# ---------------------------------------------------------------------------
# Test 8: primary_metric en cada candidato del sweep viene del contrato P4
# ---------------------------------------------------------------------------

def test_sweep_candidates_have_primary_metric_value(
    client, artifacts_dir, prepared_feature_pack: str, monkeypatch
):
    """
    Cada candidato completado tiene primary_metric_value no None
    cuando sus métricas incluyen val_rmse.
    """
    dataset_id = prepared_feature_pack

    fake_metrics = {
        "rbm_general":    {"val_rmse": 6.1, "task_type": "regression"},
        "rbm_restringida":{"val_rmse": 7.3, "task_type": "regression"},
        "dbm_manual":     {"val_rmse": 8.9, "task_type": "regression"},
    }

    _make_sweep_mocks(monkeypatch, fake_metrics)

    r = client.post(
        "/modelos/sweep",
        json=_make_sweep_request(dataset_id),
    )
    assert r.status_code == 200, r.text
    data = r.json()

    completed = [c for c in data["candidates"] if c["status"] == "completed"]
    assert len(completed) == 3
    for c in completed:
        assert c["primary_metric_value"] is not None, f"Falta primary_metric_value en {c['model_name']}"
