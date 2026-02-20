from __future__ import annotations

"""Tests unitarios para plumb de features de texto en sweep determinístico.

Motivación (P2.6):
- El sweep determinístico (``POST /modelos/sweep``) reutiliza el pipeline de entrenamiento
  y, cuando ``auto_prepare=True``, construye un feature-pack para todos los candidatos.
- Para familias con texto (p.ej. ``sentiment_desempeno``) es fácil olvidar activar
  ``text_feats_mode='tfidf_lsa'``.

Criterio de aceptación:
- ``ModelSweepRequest`` expone campos de texto (auto_text_feats, text_feats_mode, ...).
- ``_run_model_sweep`` los propaga al ``EntrenarRequest`` base usado por ``_prepare_selected_data``.

Nota:
- El test parchea ``_prepare_selected_data`` para detener el flujo temprano y evitar
  entrenamiento real.
"""

import pytest


class _StopSweep(Exception):
    """Excepción de control para detener el sweep en tests."""


def test_model_sweep_plumbs_explicit_text_params(monkeypatch):
    """Si el usuario envía parámetros de texto explícitos, deben forwardearse."""
    from neurocampus.app.routers import modelos as m
    from neurocampus.app.schemas.modelos import ModelSweepRequest

    sweep_id = "sweep_model_test_1"

    def _fake_prepare_selected_data(req, job_id: str):
        assert getattr(req, "text_feats_mode") == "tfidf_lsa"
        assert getattr(req, "text_col") == "comentario"
        assert getattr(req, "text_n_components") == 64
        assert getattr(req, "text_min_df") == 2
        assert getattr(req, "text_max_features") == 20000
        assert getattr(req, "text_random_state") == 123
        assert getattr(req, "auto_text_feats") is True
        raise _StopSweep()

    monkeypatch.setattr(m, "_prepare_selected_data", _fake_prepare_selected_data)

    req = ModelSweepRequest(
        dataset_id="ds_dummy",
        family="sentiment_desempeno",
        models=["rbm_general"],
        auto_promote_champion=False,
        # Parámetros de texto
        text_feats_mode="tfidf_lsa",
        text_col="comentario",
        text_n_components=64,
        text_min_df=2,
        text_max_features=20000,
        text_random_state=123,
        auto_text_feats=True,
    )

    with pytest.raises(_StopSweep):
        m._run_model_sweep(sweep_id, req)


def test_model_sweep_respects_auto_text_feats_flag(monkeypatch):
    """auto_text_feats debe forwardearse para permitir desactivar auto-enable."""
    from neurocampus.app.routers import modelos as m
    from neurocampus.app.schemas.modelos import ModelSweepRequest

    sweep_id = "sweep_model_test_2"

    def _fake_prepare_selected_data(req, job_id: str):
        assert getattr(req, "text_feats_mode") == "none"
        assert getattr(req, "auto_text_feats") is False
        raise _StopSweep()

    monkeypatch.setattr(m, "_prepare_selected_data", _fake_prepare_selected_data)

    req = ModelSweepRequest(
        dataset_id="ds_dummy",
        family="sentiment_desempeno",
        models=["rbm_general"],
        auto_promote_champion=False,
        auto_text_feats=False,
    )

    with pytest.raises(_StopSweep):
        m._run_model_sweep(sweep_id, req)
