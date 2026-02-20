"""
Tests unitarios para :mod:`neurocampus.models.utils.feature_selectors`.

Cubre:
- Selección de columnas calif_* con ordenamiento
- Fallback a numéricas cuando no hay calif_*
- Exclusión de columnas de metadatos
- Inclusión condicional de text_probs (p_neg/p_neu/p_pos)
- Inclusión condicional de text_embeds con autodetección de prefijo
- Trazabilidad: conteos, flags y diccionario para meta.json
- max_calif respetado

.. versionadded:: P2.6
"""

import numpy as np
import pandas as pd
import pytest

from neurocampus.models.utils.feature_selectors import (
    FeatureSelectionResult,
    auto_detect_embed_prefix,
    pick_feature_cols,
    CANDIDATE_EMBED_PREFIXES,
    META_EXCLUDE_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_basic() -> pd.DataFrame:
    """DataFrame con 5 califs + p_* + embeds."""
    n = 20
    rng = np.random.default_rng(42)
    data = {
        "alumno_id": range(n),
        "periodo": ["2025-1"] * n,
        "calif_1": rng.random(n),
        "calif_2": rng.random(n),
        "calif_3": rng.random(n),
        "calif_4": rng.random(n),
        "calif_5": rng.random(n),
        "p_neg": rng.random(n),
        "p_neu": rng.random(n),
        "p_pos": rng.random(n),
        "y_sentimiento": rng.choice(["neg", "neu", "pos"], n),
    }
    # Añadir 4 columnas de embeddings
    for i in range(4):
        data[f"x_text_{i}"] = rng.random(n)
    return pd.DataFrame(data)


@pytest.fixture
def df_no_calif() -> pd.DataFrame:
    """DataFrame sin columnas calif_* — debe usar fallback numérico."""
    n = 10
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "alumno_id": range(n),
        "pregunta_1": rng.random(n),
        "pregunta_2": rng.random(n),
        "score_total": rng.random(n),
        "y": rng.choice([0, 1, 2], n),
    })


# ---------------------------------------------------------------------------
# Tests: selección básica de calificaciones
# ---------------------------------------------------------------------------

class TestCalifSelection:
    """Verifica la selección y ordenamiento de columnas calif_*."""

    def test_calif_selected_and_ordered(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic)
        assert result.calif_cols == ["calif_1", "calif_2", "calif_3", "calif_4", "calif_5"]
        # Sin text_probs ni embeds por defecto
        assert result.feature_cols == result.calif_cols

    def test_max_calif_respected(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic, max_calif=3)
        assert len(result.calif_cols) == 3
        assert result.calif_cols == ["calif_1", "calif_2", "calif_3"]

    def test_fallback_numeric_no_calif(self, df_no_calif: pd.DataFrame):
        result = pick_feature_cols(df_no_calif)
        # alumno_id y 'y' deben estar excluidos (META_EXCLUDE_COLS)
        for col in result.feature_cols:
            assert col.lower() not in META_EXCLUDE_COLS
        # Debe incluir pregunta_1, pregunta_2, score_total
        assert len(result.feature_cols) >= 2


# ---------------------------------------------------------------------------
# Tests: text_probs
# ---------------------------------------------------------------------------

class TestTextProbs:
    """Verifica inclusión condicional de p_neg/p_neu/p_pos."""

    def test_text_probs_off_by_default(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic)
        assert result.text_prob_cols == []
        assert "p_neg" not in result.feature_cols

    def test_text_probs_included_when_activated(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic, include_text_probs=True)
        assert result.text_prob_cols == ["p_neg", "p_neu", "p_pos"]
        assert "p_pos" in result.feature_cols

    def test_text_probs_missing_columns_graceful(self):
        """Si faltan columnas p_*, no se incluyen aunque se active el flag."""
        df = pd.DataFrame({"calif_1": [1.0], "p_neg": [0.5]})  # faltan p_neu, p_pos
        result = pick_feature_cols(df, include_text_probs=True)
        assert result.text_prob_cols == []


# ---------------------------------------------------------------------------
# Tests: text_embeds
# ---------------------------------------------------------------------------

class TestTextEmbeds:
    """Verifica inclusión de embeddings de texto con autodetección."""

    def test_embeds_off_by_default(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic)
        assert result.text_embed_cols == []

    def test_embeds_included_when_activated(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic, include_text_embeds=True)
        assert result.text_embed_cols == ["x_text_0", "x_text_1", "x_text_2", "x_text_3"]
        assert result.text_embed_prefix == "x_text_"

    def test_embeds_sorted_by_suffix(self):
        """Los embeds deben estar en orden numérico del sufijo."""
        df = pd.DataFrame({
            "calif_1": [1.0],
            "x_text_10": [0.1],
            "x_text_2": [0.2],
            "x_text_0": [0.3],
        })
        result = pick_feature_cols(df, include_text_embeds=True)
        assert result.text_embed_cols == ["x_text_0", "x_text_2", "x_text_10"]

    def test_embeds_explicit_prefix(self):
        """Prefijo explícito tiene prioridad sobre autodetección."""
        df = pd.DataFrame({
            "calif_1": [1.0],
            "feat_t_0": [0.1],
            "feat_t_1": [0.2],
            "x_text_0": [0.5],
        })
        result = pick_feature_cols(
            df,
            include_text_embeds=True,
            text_embed_prefix="feat_t_",
        )
        assert result.text_embed_cols == ["feat_t_0", "feat_t_1"]
        assert result.text_embed_prefix == "feat_t_"


# ---------------------------------------------------------------------------
# Tests: autodetección de prefijo
# ---------------------------------------------------------------------------

class TestAutoDetectPrefix:
    """Verifica ``auto_detect_embed_prefix``."""

    def test_detects_x_text(self):
        assert auto_detect_embed_prefix(["calif_1", "x_text_0", "x_text_1"]) == "x_text_"

    def test_detects_text_embed(self):
        assert auto_detect_embed_prefix(["text_embed_0", "other"]) == "text_embed_"

    def test_priority_order(self):
        """x_text_ tiene prioridad sobre text_embed_ (aparece primero en la lista)."""
        cols = ["x_text_0", "text_embed_0"]
        assert auto_detect_embed_prefix(cols) == "x_text_"

    def test_none_when_no_match(self):
        assert auto_detect_embed_prefix(["calif_1", "score"]) is None


# ---------------------------------------------------------------------------
# Tests: trazabilidad
# ---------------------------------------------------------------------------

class TestTraceability:
    """Verifica ``FeatureSelectionResult.traceability_dict``."""

    def test_traceability_no_text(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic)
        trace = result.traceability_dict()
        assert trace["n_features"] == 5  # 5 califs
        assert trace["n_text_features"] == 0
        assert trace["has_text_features"] is False
        assert trace["text_embed_prefix"] is None

    def test_traceability_with_all_text(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(
            df_basic,
            include_text_probs=True,
            include_text_embeds=True,
        )
        trace = result.traceability_dict()
        assert trace["n_features"] == 5 + 3 + 4  # califs + probs + embeds
        assert trace["n_text_features"] == 3 + 4
        assert trace["has_text_features"] is True
        assert trace["n_calif_features"] == 5
        assert trace["text_embed_prefix"] == "x_text_"
        assert len(trace["text_prob_cols"]) == 3
        assert len(trace["text_feat_cols"]) == 4

    def test_properties(self, df_basic: pd.DataFrame):
        result = pick_feature_cols(df_basic, include_text_embeds=True)
        assert result.n_features == 5 + 4
        assert result.n_text_features == 4
        assert result.has_text_features is True


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Casos borde."""

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = pick_feature_cols(df)
        assert result.feature_cols == []
        assert result.n_features == 0

    def test_single_column(self):
        df = pd.DataFrame({"calif_1": [1.0, 2.0]})
        result = pick_feature_cols(df)
        assert result.feature_cols == ["calif_1"]

    def test_all_features_combined_order(self, df_basic: pd.DataFrame):
        """El orden final es: califs → probs → embeds."""
        result = pick_feature_cols(
            df_basic,
            include_text_probs=True,
            include_text_embeds=True,
        )
        expected = (
            ["calif_1", "calif_2", "calif_3", "calif_4", "calif_5"]
            + ["p_neg", "p_neu", "p_pos"]
            + ["x_text_0", "x_text_1", "x_text_2", "x_text_3"]
        )
        assert result.feature_cols == expected
