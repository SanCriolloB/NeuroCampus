def test_job_state_preserves_run_id_and_artifact_path_on_late_failure(monkeypatch, tmp_path):
    """Regression: si el job falla *después* de crear el run, el estado debe
    preservar ``run_id`` y ``artifact_path``.

    Contexto
    --------
    La UI puede consultar ``GET /modelos/estado/{job_id}`` tanto en *completed*
    como en *failed*.  Si el run llegó a persistirse (``save_run``), estos campos
    deben estar disponibles para permitir diagnóstico post-mortem.
    """

    from neurocampus.app.schemas.modelos import EntrenarRequest
    from neurocampus.app.routers import modelos as m

    # Usar un artifacts_dir controlado por el test para que _relpath devuelva
    # referencias lógicas tipo "artifacts/...".
    monkeypatch.setattr(m, "ARTIFACTS_DIR", tmp_path)

    job_id = "job-late-failure"
    run_id_fixed = "ds1__rbm_general__0000__job-late"

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
        auto_prepare=False,
    )

    # Evitar dependencias de auto_prepare
    monkeypatch.setattr(m, "_prepare_selected_data", lambda *_a, **_k: str(tmp_path / "dummy.parquet"))
    monkeypatch.setattr(m, "_infer_target_col", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_build_run_hparams", lambda *_a, **_k: {"job_id": job_id, "seed": 42})

    class DummyStrategy:
        pass

    monkeypatch.setattr(m, "_create_strategy", lambda **_k: DummyStrategy())

    class DummyTemplate:
        def __init__(self, estrategia):
            self.estrategia = estrategia

        def run(self, *, data_ref, epochs, hparams, model_name):
            return {"status": "completed", "model": model_name, "metrics": {"loss": 0.1}, "history": [{"epoch": 1}]}

    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)

    monkeypatch.setattr(
        m,
        "build_run_id",
        lambda *, dataset_id, model_name, job_id: run_id_fixed,
    )

    def fake_save_run(*, run_id, job_id, dataset_id, model_name, data_ref, params, final_metrics, history):
        out = tmp_path / "runs" / run_id
        out.mkdir(parents=True, exist_ok=True)
        # Emular la existencia del metrics.json (algunos flows lo leen inmediatamente)
        (out / "metrics.json").write_text("{}", encoding="utf-8")
        # Emular layout mínimo del export para que el flujo alcance _require_exported_model.
        (out / "model").mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(m, "save_run", fake_save_run)
    monkeypatch.setattr(m, "_try_write_predictor_bundle", lambda *_a, **_k: None)

    # Forzar fallo *después* de save_run
    monkeypatch.setattr(m, "_require_exported_model", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("export incompleto")))

    m._run_training(job_id, req)

    st = m._ESTADOS[job_id]
    assert st["status"] == "failed"
    assert st.get("run_id") == run_id_fixed
    assert st.get("artifact_path") == f"artifacts/runs/{run_id_fixed}"
    assert st.get("artifact_ready") is True
