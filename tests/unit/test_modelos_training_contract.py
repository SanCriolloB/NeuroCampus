def test_run_training_uses_template_contract(monkeypatch, tmp_path):
    """Regression test: valida contrato entre router y PlantillaEntrenamiento + runs_io."""

    from neurocampus.app.schemas.modelos import EntrenarRequest
    from neurocampus.app.routers import modelos as m

    job_id = "job12345-6789"

    m._ESTADOS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.0,
        "model": "rbm_general",
        "params": {"epochs": 2, "dataset_id": "ds1", "family": "sentiment_desempeno"},
        "metrics": {},
        "history": [],
        "run_id": None,
        "artifact_path": None,
        "job_type": "train",
    }

    req = EntrenarRequest(
        modelo="rbm_general",
        dataset_id="ds1",
        family="sentiment_desempeno",
        epochs=2,
    )

    calls = {}

    monkeypatch.setattr(m, "_ensure_feature_pack", lambda *a, **k: None)
    monkeypatch.setattr(m, "_infer_target_col", lambda *a, **k: None)
    monkeypatch.setattr(m, "_build_run_hparams", lambda *_a, **_k: {"job_id": job_id, "seed": 42})

    def fake_prepare_selected_data(_req_norm, _job_id=None):
        p = tmp_path / "dummy.parquet"
        p.write_bytes(b"PAR1")
        return str(p)

    monkeypatch.setattr(m, "_prepare_selected_data", fake_prepare_selected_data)

    class DummyStrategy:
        pass

    def fake_create_strategy(*, model_name, hparams, job_id, dataset_id, family):
        calls["create_strategy"] = {"model_name": model_name, "job_id": job_id}
        return DummyStrategy()

    monkeypatch.setattr(m, "_create_strategy", fake_create_strategy)

    class DummyTemplate:
        def __init__(self, estrategia):
            calls["tpl_init"] = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            calls["tpl_run"] = {"data_ref": data_ref, "epochs": epochs}
            return {"status": "completed", "model": model_name, "metrics": {"loss": 0.1}, "history": [{"epoch": 1}]}

    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)

    monkeypatch.setattr(
        m,
        "build_run_id",
        lambda *, dataset_id, model_name, job_id: f"{dataset_id}__{model_name}__0000__{job_id[:8]}",
    )

    def fake_save_run(*, run_id, job_id, dataset_id, model_name, data_ref, params, final_metrics, history):
        calls["save_run"] = {"run_id": run_id, "job_id": job_id}
        out = tmp_path / run_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(m, "save_run", fake_save_run)

    # Este test valida el contrato router <-> template <-> runs_io.
    # No valida export/serialización del modelo (eso se cubre en tests específicos),
    # por lo que neutralizamos la verificación de archivos exportados.
    monkeypatch.setattr(m, "_require_exported_model", lambda *_a, **_k: None)

    monkeypatch.setattr(
        m,
        "maybe_update_champion",
        lambda *, dataset_id, model_name, metrics, source_run_id=None, family=None: {"source_run_id": source_run_id},
    )

    m._run_training(job_id, req)

    assert isinstance(calls["tpl_init"], DummyStrategy)
    assert calls["tpl_run"]["data_ref"].endswith("dummy.parquet")
    assert calls["tpl_run"]["epochs"] == 2
    assert calls["save_run"]["job_id"] == job_id
    assert m._ESTADOS[job_id]["status"] == "completed"
