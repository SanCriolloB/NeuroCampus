"""
Unit tests para score_total.

Objetivo:
- Validar contratos/backward-compat de la lógica de score (base y total 0..50).
- Cubrir sidecar meta, derivación desde calif_*, y política NO_TEXT.
"""

import json

import numpy as np
import pandas as pd

from neurocampus.data.score_total import (
    compute_score_base_0_50,
    compute_score_total_0_50,
    ensure_score_columns,
    load_sidecar_score_meta,
)


def test_load_sidecar_score_meta_reads_json_when_present(tmp_path):
    """Lee el sidecar `<parquet>.meta.json` cuando existe y es JSON válido."""
    labeled = tmp_path / "ds_beto.parquet"
    labeled.write_bytes(b"PAR1")

    meta_path = tmp_path / "ds_beto.parquet.meta.json"
    meta_payload = {"score_beta": 1.23, "score_delta_max": 8.0}
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")

    got = load_sidecar_score_meta(labeled)
    assert got == meta_payload


def test_load_sidecar_score_meta_returns_none_on_invalid_json(tmp_path):
    """Si el sidecar existe pero es inválido, retorna None (fallback defensivo)."""
    labeled = tmp_path / "ds_beto.parquet"
    labeled.write_bytes(b"PAR1")

    meta_path = tmp_path / "ds_beto.parquet.meta.json"
    meta_path.write_text("{not-json}", encoding="utf-8")

    assert load_sidecar_score_meta(labeled) is None


def test_compute_score_base_prefers_rating_and_scales_likert_to_0_50():
    """Si existe rating (0..5), se reescala a 0..50 y se reporta la fuente."""
    df = pd.DataFrame({"rating": [1, 4, 5]})
    base, dbg = compute_score_base_0_50(df)

    assert dbg["base_source"] == "explicit_rating"
    assert np.allclose(base.to_numpy(), np.array([10.0, 40.0, 50.0]))


def test_compute_score_base_derives_from_calif_mean_when_needed():
    """Si no hay rating/score, usa mean(calif_*) ordenadas por sufijo numérico."""
    df = pd.DataFrame(
        {
            "calif_2": [2, 3],
            "calif_1": [4, 1],
            "calif_5": [5, 5],
        }
    )

    base, dbg = compute_score_base_0_50(df)

    assert dbg["base_source"] == "derived_from_calif_mean"
    # Debe detectar y ordenar columnas por sufijo numérico para reproducibilidad
    assert dbg["calif_cols_detected"] == ["calif_1", "calif_2", "calif_5"]

    # mean fila0 = (4+2+5)/3=3.666.. => Likert => *10
    assert np.isclose(float(base.iloc[0]), 36.6666666667, atol=1e-6)


def test_ensure_score_columns_prefers_existing_total_and_creates_base_if_missing():
    """Si score_total_0_50 existe y prefer_total=True, no recalcula total; asegura base."""
    df = pd.DataFrame({"score_total_0_50": [33.0], "rating": [2]})
    out, score_col, dbg = ensure_score_columns(df, prefer_total=True, allow_derive=True)

    assert score_col == "score_total_0_50"
    assert dbg["source"] == "explicit_score_total"
    assert "score_base_0_50" in out.columns
    assert "score_base_0_50" in dbg["created_columns"]
    assert float(out.loc[0, "score_total_0_50"]) == 33.0


def test_ensure_score_columns_derives_total_without_sentiment_columns():
    """Si faltan p_pos/p_neg, total=base y beta=0.0 (no rompe)."""
    df = pd.DataFrame({"rating": [3, 2]})
    out, score_col, dbg = ensure_score_columns(df, prefer_total=True, allow_derive=True)

    assert score_col == "score_total_0_50"
    assert dbg["source"] == "derived"
    assert dbg["beta_used"] == 0.0

    assert np.allclose(out["score_total_0_50"].to_numpy(), np.array([30.0, 20.0]))
    assert np.allclose(out["sentiment_delta_points"].to_numpy(), np.array([0.0, 0.0]))


def test_compute_score_total_respects_no_text_policy_and_calibration():
    """has_text==0 => señal 0. has_text==1 => calibra beta y aplica delta."""
    df = pd.DataFrame(
        {
            "rating": [2, 2],
            "p_pos": [0.9, 0.9],
            "p_neg": [0.1, 0.1],
            "sentiment_conf": [1.0, 1.0],
            "has_text": [0, 1],
        }
    )

    meta = compute_score_total_0_50(df, delta_max=8.0, calib_q=1.0)

    # Fila sin texto: delta_points=0 y total=base
    assert float(df.loc[0, "sentiment_delta_points"]) == 0.0
    assert float(df.loc[0, "score_total_0_50"]) == float(df.loc[0, "score_base_0_50"])

    # Fila con texto: delta=0.8, qv=0.8 => beta=10, delta_points=8 (clipped)
    assert np.isclose(float(df.loc[1, "sentiment_delta_points"]), 8.0)
    assert np.isclose(float(df.loc[1, "score_total_0_50"]), 28.0)

    # Meta mínima para auditoría
    assert meta["score_delta_max"] == 8.0
    assert meta["score_calib_q"] == 1.0
    assert np.isclose(float(meta["score_beta"]), 10.0)
