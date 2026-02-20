from __future__ import annotations

"""Tests unitarios para plumb de features de texto en sweep asíncrono.

Motivación (P2.6):
- El endpoint ``POST /modelos/entrenar`` ya soporta parámetros opcionales de texto
  (``text_feats_mode``, ``text_col``, etc.) que se usan durante ``auto_prepare``.
- El sweep asíncrono (``POST /modelos/entrenar/sweep``) reutiliza el mismo flujo,
  pero históricamente no exponía dichos parámetros.

Criterio de aceptación:
- ``SweepEntrenarRequest`` acepta campos de texto.
- ``_run_sweep_training`` los propaga al ``EntrenarRequest`` base que prepara los datos.

Nota:
- El test parchea ``_prepare_selected_data`` para cortar el flujo temprano y evitar
  acceso real a artifacts/datasets.
"""

import pytest


class _StopSweep(Exception):
    """Excepción de control para detener el sweep en tests."""


def test_sweep_plumbs_explicit_text_params(monkeypatch):
    """Si el usuario envía parámetros de texto explícitos, deben forwardearse."""
    from neurocampus.app.routers import modelos as m
    from neurocampus.app.schemas.modelos import SweepEntrenarRequest

    sweep_id = "sweep_test_1"
    m._ESTADOS[sweep_id] = {"job_id": sweep_id, "job_type": "sweep"}

    def _fake_prepare_selected_data(req, job_id: str):
        # El request base DEBE contener los parámetros explicitados.
        assert getattr(req, "text_feats_mode") == "tfidf_lsa"
        assert getattr(req, "text_col") == "comentario"
        assert getattr(req, "text_n_components") == 64
        assert getattr(req, "text_min_df") == 2
        assert getattr(req, "text_max_features") == 20000
        assert getattr(req, "text_random_state") == 123
        assert getattr(req, "auto_text_feats") is True
        raise _StopSweep()

    monkeypatch.setattr(m, "_prepare_selected_data", _fake_prepare_selected_data)

    req = SweepEntrenarRequest(
        dataset_id="ds_dummy",
        family="sentiment_desempeno",
        modelos=["rbm_general"],
        max_total_runs=1,
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
        m._run_sweep_training(sweep_id, req)


def test_sweep_respects_auto_text_feats_flag(monkeypatch):
    """El flag auto_text_feats debe forwardearse para permitir desactivar el auto-enable."""
    from neurocampus.app.routers import modelos as m
    from neurocampus.app.schemas.modelos import SweepEntrenarRequest

    sweep_id = "sweep_test_2"
    m._ESTADOS[sweep_id] = {"job_id": sweep_id, "job_type": "sweep"}

    def _fake_prepare_selected_data(req, job_id: str):
        # Si no se especifica text_feats_mode, debe conservar el default ('none').
        assert getattr(req, "text_feats_mode") == "none"
        assert getattr(req, "auto_text_feats") is False
        raise _StopSweep()

    monkeypatch.setattr(m, "_prepare_selected_data", _fake_prepare_selected_data)

    req = SweepEntrenarRequest(
        dataset_id="ds_dummy",
        family="sentiment_desempeno",
        modelos=["rbm_general"],
        max_total_runs=1,
        auto_promote_champion=False,
        auto_text_feats=False,
    )

    with pytest.raises(_StopSweep):
        m._run_sweep_training(sweep_id, req)
