"""
tests/unit/test_dbm_serialization.py

Tests de round-trip save/load para DBMManual (Parte 3).
No requieren torch ni FastAPI — solo numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importar sin activar el módulo router (que depende de torch)
# ---------------------------------------------------------------------------
import sys, os
# El conftest ya añade backend/src al sys.path

from neurocampus.models.dbm_manual import DBMManual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dbm(n_visible=6, n_hidden1=4, n_hidden2=3, seed=42) -> DBMManual:
    m = DBMManual(n_visible=n_visible, n_hidden1=n_hidden1, n_hidden2=n_hidden2, seed=seed)
    # Simular pesos "entrenados" (no random xavier — valores deterministas)
    rng = np.random.default_rng(seed)
    m.rbm_v_h1.W  = rng.standard_normal((n_visible, n_hidden1)).astype(np.float32)
    m.rbm_v_h1.bv = rng.standard_normal(n_visible).astype(np.float32)
    m.rbm_v_h1.bh = rng.standard_normal(n_hidden1).astype(np.float32)
    m.rbm_h1_h2.W  = rng.standard_normal((n_hidden1, n_hidden2)).astype(np.float32)
    m.rbm_h1_h2.bv = rng.standard_normal(n_hidden1).astype(np.float32)
    m.rbm_h1_h2.bh = rng.standard_normal(n_hidden2).astype(np.float32)
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dbm_save_creates_expected_files(tmp_path: Path):
    """save() debe crear dbm_state.npz y meta.json."""
    m = _make_dbm()
    m.save(str(tmp_path))

    assert (tmp_path / "dbm_state.npz").exists(), "dbm_state.npz falta"
    assert (tmp_path / "meta.json").exists(), "meta.json falta"


def test_dbm_meta_json_content(tmp_path: Path):
    """meta.json debe contener schema_version, dimensiones y hparams."""
    m = _make_dbm(n_visible=5, n_hidden1=3, n_hidden2=2)
    m.save(str(tmp_path), extra_meta={"feat_cols_": ["a", "b", "c", "d", "e"]})

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["n_visible"] == 5
    assert meta["n_hidden1"] == 3
    assert meta["n_hidden2"] == 2
    assert "hparams" in meta
    assert meta.get("feat_cols_") == ["a", "b", "c", "d", "e"]


def test_dbm_load_roundtrip_shapes(tmp_path: Path):
    """load() debe reconstruir con las mismas dimensiones."""
    m = _make_dbm(n_visible=6, n_hidden1=4, n_hidden2=3)
    m.save(str(tmp_path))

    m2 = DBMManual.load(str(tmp_path))
    assert m2.rbm_v_h1.W.shape  == (6, 4)
    assert m2.rbm_v_h1.bv.shape == (6,)
    assert m2.rbm_v_h1.bh.shape == (4,)
    assert m2.rbm_h1_h2.W.shape  == (4, 3)
    assert m2.rbm_h1_h2.bv.shape == (4,)
    assert m2.rbm_h1_h2.bh.shape == (3,)


def test_dbm_load_roundtrip_values(tmp_path: Path):
    """load() debe preservar los valores de los arrays exactamente."""
    m = _make_dbm()
    m.save(str(tmp_path))

    m2 = DBMManual.load(str(tmp_path))
    assert np.allclose(m.rbm_v_h1.W,   m2.rbm_v_h1.W),   "W1 no coincide"
    assert np.allclose(m.rbm_v_h1.bv,  m2.rbm_v_h1.bv),  "bv1 no coincide"
    assert np.allclose(m.rbm_v_h1.bh,  m2.rbm_v_h1.bh),  "bh1 no coincide"
    assert np.allclose(m.rbm_h1_h2.W,  m2.rbm_h1_h2.W),  "W2 no coincide"
    assert np.allclose(m.rbm_h1_h2.bv, m2.rbm_h1_h2.bv), "bv2 no coincide"
    assert np.allclose(m.rbm_h1_h2.bh, m2.rbm_h1_h2.bh), "bh2 no coincide"


def test_dbm_load_missing_npz_raises(tmp_path: Path):
    """load() lanza FileNotFoundError si falta dbm_state.npz."""
    # Solo crear meta.json, sin npz
    (tmp_path / "meta.json").write_text(
        '{"schema_version":1,"n_visible":3,"n_hidden1":2,"n_hidden2":2,"hparams":{}}',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="dbm_state.npz"):
        DBMManual.load(str(tmp_path))


def test_dbm_load_missing_meta_raises(tmp_path: Path):
    """load() lanza FileNotFoundError si falta meta.json."""
    # Solo crear npz vacío
    np.savez(tmp_path / "dbm_state.npz", W1=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="meta.json"):
        DBMManual.load(str(tmp_path))


def test_dbm_copy_weights_from_ok(tmp_path: Path):
    """copy_weights_from() copia pesos cuando las dimensiones coinciden."""
    src = _make_dbm(n_visible=4, n_hidden1=3, n_hidden2=2, seed=1)
    dst = _make_dbm(n_visible=4, n_hidden1=3, n_hidden2=2, seed=99)

    # Pesos diferentes antes de copiar
    assert not np.allclose(src.rbm_v_h1.W, dst.rbm_v_h1.W)

    dst.copy_weights_from(src)
    assert np.allclose(src.rbm_v_h1.W, dst.rbm_v_h1.W)
    assert np.allclose(src.rbm_h1_h2.W, dst.rbm_h1_h2.W)


def test_dbm_copy_weights_from_incompatible_raises():
    """copy_weights_from() lanza ValueError si las dimensiones no coinciden."""
    src = _make_dbm(n_visible=4, n_hidden1=3, n_hidden2=2)
    dst = _make_dbm(n_visible=6, n_hidden1=3, n_hidden2=2)  # n_visible diferente

    with pytest.raises(ValueError, match="incompatibles"):
        dst.copy_weights_from(src)


def test_dbm_transform_after_roundtrip(tmp_path: Path):
    """El modelo cargado produce las mismas representaciones que el original."""
    m = _make_dbm(n_visible=6, n_hidden1=4, n_hidden2=3)
    m.save(str(tmp_path))

    m2 = DBMManual.load(str(tmp_path))

    X = np.random.default_rng(0).standard_normal((10, 6)).astype(np.float32)
    h_orig = m.transform(X)
    h_loaded = m2.transform(X)

    assert np.allclose(h_orig, h_loaded, atol=1e-5), \
        "transform() produce resultados distintos tras round-trip"
