def test_training_persisted_event_emitted_and_updates_state(monkeypatch, tmp_path):
    """Regression: el router debe emitir ``training.persisted`` tras ``save_run``.

    Esta señal es útil para:

    - UI / polling: saber que ``artifact_path`` ya apunta a un directorio existente.
    - Debug: permitir inspección temprana de ``metrics.json`` aunque export/bundle falle luego.

    El test evita entrenamientos reales mediante monkeypatching de:

    - PlantillaEntrenamiento
    - save_run
    - _try_write_predictor_bundle / _require_exported_model / maybe_update_champion
    """

    from neurocampus.app.schemas.modelos import EntrenarRequest
    from neurocampus.app.routers import modelos as m

    # Usar un artifacts_dir aislado para que _relpath genere rutas lógicas.
    monkeypatch.setattr(m, "ARTIFACTS_DIR", tmp_path)

    job_id = "job-persisted"
    run_id_fixed = "ds1__rbm_general__0000__job-persisted"

    # Capturar eventos publicados.
    captured = []

    def _capture(evt):
        if (evt.payload or {}).get("correlation_id") == job_id:
            captured.append(evt)

    m.BUS.subscribe("training.persisted", _capture)

    req = EntrenarRequest(
        modelo="rbm_general",
        dataset_id="ds1",
        family="sentiment_desempeno",
        epochs=2,
        auto_prepare=False,
    )

    # Evitar dependencias de datos reales.
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
            return {
                "status": "completed",
                "model": model_name,
                "metrics": {"loss": 0.1},
                "history": [{"epoch": 1}],
            }

    monkeypatch.setattr(m, "PlantillaEntrenamiento", DummyTemplate)
    monkeypatch.setattr(m, "build_run_id", lambda *, dataset_id, model_name, job_id: run_id_fixed)

    def fake_save_run(*, run_id, job_id, dataset_id, model_name, data_ref, params, final_metrics, history):
        out = tmp_path / "runs" / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text("{}", encoding="utf-8")
        (out / "model").mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(m, "save_run", fake_save_run)
    monkeypatch.setattr(m, "_try_write_predictor_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_require_exported_model", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "maybe_update_champion", lambda *_a, **_k: {"promoted": False})

    m._run_training(job_id, req)

    # Verificar que el evento se emitió con un payload navegable.
    assert captured, "Se esperaba al menos un evento training.persisted"
    evt = captured[-1]
    assert evt.payload.get("run_id") == run_id_fixed
    assert evt.payload.get("artifact_path") == f"artifacts/runs/{run_id_fixed}"
    assert evt.payload.get("artifact_ready") is True

    # El estado del job debe reflejar la persistencia.
    st = m._ESTADOS[job_id]
    assert st.get("artifact_ready") is True
    assert st.get("run_id") == run_id_fixed
    assert st.get("artifact_path") == f"artifacts/runs/{run_id_fixed}"
