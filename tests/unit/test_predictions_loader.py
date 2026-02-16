from pathlib import Path
import json
import pytest

from neurocampus.predictions.bundle import bundle_paths, write_json, build_predictor_manifest
from neurocampus.predictions.loader import (
    ChampionNotFoundError,
    PredictorNotFoundError,
    PredictorNotReadyError,
    load_predictor_by_champion,
    load_predictor_by_run_id,
)
from neurocampus.utils.paths import artifacts_dir


def _write_champion_json(base: Path, *, family: str, dataset_id: str, run_id: str) -> Path:
    p = base / "champions" / family / dataset_id / "champion.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"source_run_id": run_id}, indent=2), encoding="utf-8")
    return p


def test_loader_raises_if_predictor_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(tmp_path))
    run_id = "r1"
    run_dir = artifacts_dir(refresh=True) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(PredictorNotFoundError):
        load_predictor_by_run_id(run_id)


def test_loader_raises_if_model_bin_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(tmp_path))
    run_id = "r2"
    run_dir = artifacts_dir(refresh=True) / "runs" / run_id
    bp = bundle_paths(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        bp.predictor_json,
        build_predictor_manifest(
            run_id=run_id,
            dataset_id="ds",
            model_name="rbm_general",
            task_type="classification",
            input_level="row",
            target_col="y",
        ),
    )
    bp.model_bin.write_bytes(b"PLACEHOLDER_MODEL_BIN_P2_1")

    with pytest.raises(PredictorNotReadyError):
        load_predictor_by_run_id(run_id)


def test_loader_by_champion_resolves_run_id_and_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(tmp_path))
    base = artifacts_dir(refresh=True)

    run_id = "r3"
    run_dir = base / "runs" / run_id
    bp = bundle_paths(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        bp.predictor_json,
        build_predictor_manifest(
            run_id=run_id,
            dataset_id="ds",
            model_name="rbm_general",
            task_type="classification",
            input_level="row",
            target_col="y",
        ),
    )
    # Simula model.bin "real" (cualquier bytes != placeholder)
    bp.model_bin.write_bytes(b"REAL_MODEL_BYTES_v1")

    _write_champion_json(base, family="sentiment_desempeno", dataset_id="ds", run_id=run_id)

    loaded = load_predictor_by_champion(dataset_id="ds", family="sentiment_desempeno")
    assert loaded.run_id == run_id
    assert loaded.model_bin_path.exists()


def test_loader_by_champion_raises_if_champion_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_ARTIFACTS_DIR", str(tmp_path))

    with pytest.raises(ChampionNotFoundError):
        load_predictor_by_champion(dataset_id="ds", family="sentiment_desempeno")
