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
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, Optional
import json

from neurocampus.predictions.bundle import bundle_paths, read_json
from neurocampus.utils.paths import (
    artifacts_dir,
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


# Cache (P2.4-D)
# -------------
# Cache LRU en memoria (por proceso) para evitar recargar predictor.json/model.bin
# repetidamente en flujos como Dashboard/UI.
#
# Importante: el cache key incluye el artifacts_root actual para que tests (y
# entornos) que cambian NC_ARTIFACTS_DIR no se contaminen entre sí.
PREDICTOR_CACHE_MAXSIZE = int(os.getenv("NC_PREDICTOR_CACHE_MAXSIZE", "16"))


def _artifacts_root_key() -> str:
    """Clave estable de cache dependiente del artifacts_dir actual."""

    return str(artifacts_dir().expanduser().resolve())


def _read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise PredictorNotReadyError(f"JSON inválido: {path}") from e


def _is_placeholder_model_bin(path: Path) -> bool:
    """Detecta placeholder P2.1.

    En P2.2+ el entrenamiento puede persistir el modelo real en ``<run_dir>/model``
    (p.ej. ``rbm.pt``, ``head.pt``, ``meta.json``) y dejar ``model.bin`` como
    placeholder por compatibilidad. En ese caso el bundle *sí* puede estar listo
    para inferencia aunque ``model.bin`` sea placeholder.
    """
    try:
        head = path.read_bytes()[:64]
    except FileNotFoundError:
        return False
    return PLACEHOLDER_MAGIC in head


def _load_predictor_by_run_id_uncached(run_id: str) -> LoadedPredictorBundle:
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

    # En P2.2+ el modelo real puede estar en <run_dir>/model/ aunque model.bin sea placeholder.
    # Solo consideramos "no listo" si model.bin es placeholder *y* no existe un model_dir válido.
    if _is_placeholder_model_bin(bp.model_bin):
        model_dir = (run_dir / "model").resolve()
        meta_ok = (model_dir / "meta.json").exists() if model_dir.is_dir() else False
        rbm_ok = any((model_dir / n).exists() for n in ("rbm.pt", "head.pt")) if model_dir.is_dir() else False
        dbm_ok = ((model_dir / "dbm_state.npz").exists() and (model_dir / "ridge_head.npz").exists()) if model_dir.is_dir() else False
        weights_ok = bool(rbm_ok or dbm_ok)
        if not (model_dir.exists() and model_dir.is_dir() and meta_ok and weights_ok):
            raise PredictorNotReadyError(
                "model.bin es placeholder (P2.1) y no hay dump real en <run_dir>/model. "
                "Implementa dump real por estrategia antes de inferir."
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


@lru_cache(maxsize=PREDICTOR_CACHE_MAXSIZE)
def _load_predictor_by_run_id_cached(run_id: str, artifacts_root: str):
    """Carga bundle por run_id usando cache LRU.

    `artifacts_root` se usa solo como parte de la clave del cache para aislar
    cambios de `NC_ARTIFACTS_DIR` entre tests/entornos.
    """

    # Importante: no usamos artifacts_root explícitamente; `resolve_run_dir`
    # ya lee `NC_ARTIFACTS_DIR` y el key asegura aislamiento.
    return _load_predictor_by_run_id_uncached(run_id)


def clear_predictor_cache() -> None:
    """Limpia el cache LRU del loader (útil para tests/debug)."""

    try:
        _load_predictor_by_run_id_cached.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def load_predictor_by_run_id(run_id: str, *, use_cache: bool = True):
    """Carga bundle por run_id con cache opcional.

    Args:
        run_id: id del run.
        use_cache: si False, fuerza lectura desde disco.

    Notes:
        El cache se desactiva si `NC_PREDICTOR_CACHE_MAXSIZE<=0`.
    """

    if not use_cache or PREDICTOR_CACHE_MAXSIZE <= 0:
        return _load_predictor_by_run_id_uncached(run_id)

    return _load_predictor_by_run_id_cached(str(run_id), _artifacts_root_key())


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


def load_predictor_by_champion(*, dataset_id: str, family: Optional[str], use_cache: bool = True) -> LoadedPredictorBundle:
    """Carga bundle usando champion como entrada (dataset_id + family)."""
    run_id = resolve_run_id_from_champion(dataset_id=dataset_id, family=family)
    return load_predictor_by_run_id(run_id, use_cache=use_cache)
