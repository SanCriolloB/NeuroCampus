import json
from pathlib import Path

import pandas as pd

from neurocampus.data.features_prepare import prepare_feature_pack

import importlib
import pytest

# Este módulo depende de un engine parquet (pyarrow o fastparquet).
# En ambientes mínimos (ej. algunos runners de evaluación) puede no existir.
# El propio backend requiere parquet para feature-packs, así que en CI real
# este skip no debería activarse.
if importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('fastparquet') is None:
    pytest.skip('Parquet engine no disponible (pyarrow/fastparquet).', allow_module_level=True)


def _write_parquet(base_dir: Path, rel_path: str, df: pd.DataFrame) -> str:
    """Helper: escribe un parquet y retorna su ruta relativa."""
    p = (base_dir / rel_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return str(p.relative_to(base_dir).as_posix())


def test_prepare_feature_pack_generates_text_feats_tfidf_lsa(tmp_path: Path) -> None:
    """El feature-pack debe poder generar feat_t_* cuando se activa tfidf_lsa."""
    base_dir = tmp_path

    df = pd.DataFrame(
        {
            "cedula_profesor": ["1", "1", "2", "2"],
            "codigo_materia": ["MAT", "MAT", "MAT", "PHY"],
            "score_total_0_50": [10, 20, 30, 40],
            "comentario": ["muy bueno", "malo", "excelente", ""],
        }
    )

    input_uri = _write_parquet(base_dir, "data/labeled/2025-1_beto.parquet", df)
    out_dir = base_dir / "artifacts" / "features" / "2025-1"

    prepare_feature_pack(
        base_dir=base_dir,
        dataset_id="2025-1",
        input_uri=input_uri,
        output_dir=str(out_dir),
        text_feats_mode="tfidf_lsa",
        text_col="comentario",
        text_n_components=8,
        text_min_df=1,
        text_max_features=500,
    )

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["blocks"]["text_feats"] is True
    assert isinstance(meta.get("text_feat_cols"), list)
    assert len(meta["text_feat_cols"]) > 0

    # Trazabilidad explícita del bloque de texto
    assert meta["text"]["mode"] == "tfidf_lsa"
    assert bool(meta["text"].get("enabled")) is True
    assert meta["text"].get("source_col") == "comentario"

    train = pd.read_parquet(out_dir / "train_matrix.parquet")
    assert any(str(c).startswith("feat_t_") for c in train.columns)
    # La columna de texto raw NO debe entrar al train_matrix
    assert "comentario" not in train.columns


def test_prepare_feature_pack_text_feats_mode_no_text_column_is_safe(tmp_path: Path) -> None:
    """Si se activa tfidf_lsa pero no existe columna de texto, no debe fallar."""
    base_dir = tmp_path

    df = pd.DataFrame(
        {
            "cedula_profesor": ["1", "2", "3"],
            "codigo_materia": ["MAT", "MAT", "PHY"],
            "score_total_0_50": [10, 20, 30],
        }
    )

    input_uri = _write_parquet(base_dir, "data/labeled/2025-2_beto.parquet", df)
    out_dir = base_dir / "artifacts" / "features" / "2025-2"

    prepare_feature_pack(
        base_dir=base_dir,
        dataset_id="2025-2",
        input_uri=input_uri,
        output_dir=str(out_dir),
        text_feats_mode="tfidf_lsa",
        text_col=None,
        text_n_components=8,
        text_min_df=1,
        text_max_features=500,
    )

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["blocks"]["text_feats"] is False
    assert meta["text"]["mode"] == "tfidf_lsa"
    assert bool(meta["text"].get("enabled")) is False
