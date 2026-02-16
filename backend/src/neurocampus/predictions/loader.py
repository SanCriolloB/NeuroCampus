"""
neurocampus.predictions.loader
==============================

Loader del predictor bundle para inferencia (P2.1).

Responsabilidades
-----------------
- Resolver un predictor por:
  - run_id directo, o
  - champion (dataset_id + family) => champion.json => source_run_id
- Validar existencia del bundle:
  - predictor.json obligatorio
  - model.bin obligatorio (en P2.1 puede ser placeholder; se detecta)
  - preprocess.json opcional (si no existe, se usa {})

Decisiones
----------
- Este módulo NO hace inferencia todavía. Solo carga y valida el bundle.
- El router P2 (en pasos siguientes) usará estas funciones y decidirá HTTP 404/422/501.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

from neurocampus.predictions.bundle import bundle_paths, read_json
from neurocampus.utils.paths import (
    abs_artifact_path,
    first_existing,
    resolve_champion_json_candidates,
    resolve_run_dir,
)


class PredictorNotFoundError(RuntimeError):
    """No existe predictor.json / bundle para el run solicitado."""


class ChampionNotFoundError(RuntimeError):
    """No se encontró champion.json para dataset/family."""


class PredictorNotReadyError(RuntimeError):
    """Bundle existe pero no está listo para inferencia (ej. model.bin placeholder)."""


@dataclass(frozen=True)
class LoadedPredictorBundle:
    """Bundle cargado desde artifacts/runs/<run_id>/."""
    run_id: str
    run_dir: Path
    predictor: Dict[str, Any]
    preprocess: Dict[str, Any]
    model_bin_path: Path


PLACEHOLDER_MAGIC = b"PLACEHOLDER_MODEL_BIN_P2_1"


def _read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise PredictorNotReadyError(f"JSON inválido: {path}") from e


def _is_placeholder_model_bin(path: Path) -> bool:
    """Detecta placeholder P2.1. En P2.2+ se reemplaza por dump real."""
    try:
        head = path.read_bytes()[:64]
    except FileNotFoundError:
        return False
    return PLACEHOLDER_MAGIC in head


def load_predictor_by_run_id(run_id: str) -> LoadedPredictorBundle:
    """Carga bundle por run_id.

    Raises:
        PredictorNotFoundError: si falta predictor.json o model.bin.
        PredictorNotReadyError: si el model.bin es placeholder (bundle no listo para inferencia).
    """
    run_dir = resolve_run_dir(run_id)
    bp = bundle_paths(run_dir)

    if not bp.predictor_json.exists():
        raise PredictorNotFoundError(f"predictor.json no existe para run_id={run_id}")

    if not bp.model_bin.exists():
        raise PredictorNotFoundError(f"model.bin no existe para run_id={run_id}")

    if _is_placeholder_model_bin(bp.model_bin):
        raise PredictorNotReadyError(
            "model.bin es placeholder (P2.1). Implementa dump real por estrategia antes de inferir."
        )

    predictor = _read_json_safe(bp.predictor_json)
    preprocess = _read_json_safe(bp.preprocess_json)

    return LoadedPredictorBundle(
        run_id=str(run_id),
        run_dir=Path(run_dir).expanduser().resolve(),
        predictor=predictor,
        preprocess=preprocess,
        model_bin_path=bp.model_bin,
    )


def resolve_run_id_from_champion(*, dataset_id: str, family: Optional[str]) -> str:
    """Resuelve source_run_id a partir de champion.json (layout nuevo y fallback legacy).

    Raises:
        ChampionNotFoundError: si no existe champion.json.
        PredictorNotReadyError: si champion.json existe pero no tiene source_run_id.
    """
    candidates = resolve_champion_json_candidates(dataset_id=dataset_id, family=family)
    champ_path = first_existing(candidates)
    if not champ_path:
        raise ChampionNotFoundError(f"No se encontró champion.json para dataset_id={dataset_id} family={family}")

    champ = _read_json_safe(champ_path)
    rid = champ.get("source_run_id") or champ.get("run_id")
    if not rid:
        raise PredictorNotReadyError(f"champion.json sin source_run_id: {champ_path}")
    return str(rid)


def load_predictor_by_champion(*, dataset_id: str, family: Optional[str]) -> LoadedPredictorBundle:
    """Carga bundle usando champion como entrada (dataset_id + family)."""
    run_id = resolve_run_id_from_champion(dataset_id=dataset_id, family=family)
    return load_predictor_by_run_id(run_id)
