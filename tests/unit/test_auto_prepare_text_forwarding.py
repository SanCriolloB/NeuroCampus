def test_auto_prepare_passes_text_feature_options(monkeypatch, tmp_path):
    """P2.6 regression: auto_prepare debe propagar parámetros de texto al feature-pack.

    Este test NO entrena modelos ni genera matrices reales; solo valida que el router
    (Modelos) forwardee los parámetros al builder cuando el artefacto no existe.
    """

    from neurocampus.app.schemas.modelos import EntrenarRequest
    from neurocampus.app.routers import modelos as m

    # Aislar filesystem del router a un tmp_dir
    monkeypatch.setattr(m, "BASE_DIR", tmp_path)
    monkeypatch.setattr(m, "ARTIFACTS_DIR", tmp_path / "artifacts")

    # Evitar dependencia de labeled BETO en este test
    def _raise_no_labeled(_ds: str):
        raise RuntimeError("no labeled in unit test")

    monkeypatch.setattr(m, "resolve_labeled_path", _raise_no_labeled)

    # Fuente mínima: datasets/<ds>.parquet debe existir para que el router elija input_uri.
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "ds1.parquet").write_bytes(b"PAR1")

    calls = {}

    def fake_ensure_feature_pack(dataset_id: str, input_uri: str, *, force: bool = False, **kwargs):
        calls["dataset_id"] = dataset_id
        calls["input_uri"] = input_uri
        calls["force"] = force
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(m, "_ensure_feature_pack", fake_ensure_feature_pack)

    req = EntrenarRequest(
        modelo="rbm_general",
        dataset_id="ds1",
        family="sentiment_desempeno",
        data_source="feature_pack",
        auto_prepare=True,
        # Parámetros de texto a forwardear
        text_feats_mode="tfidf_lsa",
        text_col="comentario",
        text_n_components=32,
        text_min_df=3,
        text_max_features=1000,
        text_random_state=7,
    )

    data_ref = m._resolve_by_data_source(req)
    m._auto_prepare_if_needed(req, data_ref)

    assert calls["dataset_id"] == "ds1"
    assert calls["text_feats_mode"] == "tfidf_lsa"
    assert calls["text_col"] == "comentario"
    assert calls["text_n_components"] == 32
    assert calls["text_min_df"] == 3
    assert calls["text_max_features"] == 1000
    assert calls["text_random_state"] == 7
