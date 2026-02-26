
def test_auto_prepare_auto_enables_text_feats_for_sentiment(monkeypatch, tmp_path):
    """P2.6 regression: auto_prepare debe activar text feats en sentiment_desempeno.

    Motivación
    ----------
    En datasets de sentimiento es común disponer de una columna de texto libre
    (por ejemplo ``comentario``) que añade señal adicional. Para reducir fricción
    en el flujo de la pestaña *Modelos*, el backend puede activar por defecto la
    generación de ``feat_t_*`` (TF-IDF + LSA) durante ``auto_prepare``.

    Este test valida que, si el usuario NO especifica ``text_feats_mode`` y
    ``auto_text_feats=True`` (default), el router forwardea
    ``text_feats_mode='tfidf_lsa'`` al builder del feature-pack.

    El test no entrena modelos ni genera matrices reales; solo observa los
    argumentos enviados al helper ``_ensure_feature_pack``.
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

    # NOTA: no pasamos text_feats_mode => usa default 'none'
    req = EntrenarRequest(
        modelo="rbm_general",
        dataset_id="ds1",
        family="sentiment_desempeno",
        data_source="feature_pack",
        auto_prepare=True,
    )

    data_ref = m._resolve_by_data_source(req)
    m._auto_prepare_if_needed(req, data_ref)

    assert calls["dataset_id"] == "ds1"
    assert calls["text_feats_mode"] == "tfidf_lsa"



def test_auto_prepare_can_disable_auto_text_feats(monkeypatch, tmp_path):
    """P2.6: auto_text_feats=false debe conservar el comportamiento legacy (no text feats)."""

    from neurocampus.app.schemas.modelos import EntrenarRequest
    from neurocampus.app.routers import modelos as m

    monkeypatch.setattr(m, "BASE_DIR", tmp_path)
    monkeypatch.setattr(m, "ARTIFACTS_DIR", tmp_path / "artifacts")

    def _raise_no_labeled(_ds: str):
        raise RuntimeError("no labeled in unit test")

    monkeypatch.setattr(m, "resolve_labeled_path", _raise_no_labeled)

    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "ds2.parquet").write_bytes(b"PAR1")

    calls = {}

    def fake_ensure_feature_pack(dataset_id: str, input_uri: str, *, force: bool = False, **kwargs):
        calls["dataset_id"] = dataset_id
        calls["input_uri"] = input_uri
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(m, "_ensure_feature_pack", fake_ensure_feature_pack)

    req = EntrenarRequest(
        modelo="rbm_general",
        dataset_id="ds2",
        family="sentiment_desempeno",
        data_source="feature_pack",
        auto_prepare=True,
        auto_text_feats=False,
    )

    data_ref = m._resolve_by_data_source(req)
    m._auto_prepare_if_needed(req, data_ref)

    assert calls["dataset_id"] == "ds2"
    assert calls["text_feats_mode"] == "none"
