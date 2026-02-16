"""
neurocampus.utils.paths
======================

Resolver único (y estable) de rutas para artifacts del backend.

Motivación (P2)
---------------
Para Predicciones necesitamos resolver *siempre* las rutas correctas de:

- Feature-pack: artifacts/features/<dataset_id>/{train_matrix.parquet, meta.json, ...}
- Runs:         artifacts/runs/<run_id>/{metrics.json, history.json, ...}
- Champions:    artifacts/champions/<family>/<dataset_id>/champion.json (layout nuevo)
               + fallback legacy: artifacts/champions/<dataset_id>/champion.json

Reglas de resolución
--------------------
- Si existe `NC_ARTIFACTS_DIR`, ese directorio es la fuente de verdad.
- Si no existe, se usa `<repo_root>/artifacts`.
- `repo_root` se infiere:
  1) `NC_PROJECT_ROOT` si está definido.
  2) Subiendo desde este archivo buscando `Makefile` o carpeta `backend/`.
  3) Fallback: `Path.cwd()`.

Este módulo es intencionalmente **pequeño** y **sin side effects** (no crea carpetas).
Otros módulos (runs_io/features_prepare/routers) pueden importar esto y crear dirs cuando corresponda.

Compatibilidad
--------------
- No cambia contratos P0/P1.
- Está diseñado para soportar layout nuevo y legacy en champions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import os
import re


# ---------------------------------------------------------------------------
# Helpers de sanitización (evitar path traversal accidental)
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_segment(value: Any) -> str:
    """Convierte `value` en un segmento seguro de path.

    - Reemplaza separadores de path y caracteres raros por "_".
    - Evita ".." (traversal) por defensa en profundidad.
    - No fuerza lower/upper: respeta la intención del caller, salvo sanitización.

    Args:
        value: Valor a convertir (dataset_id, family, run_id, etc.)

    Returns:
        Segmento apto para usar como nombre de carpeta/archivo.
    """
    s = str(value or "").strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = s.replace("..", "_")
    s = _SEGMENT_RE.sub("_", s)
    s = s.strip("_")
    return s or "x"


def _find_project_root() -> Path:
    """Encuentra una raíz razonable del repo.

    Orden:
    1) NC_PROJECT_ROOT si existe.
    2) Buscar hacia arriba un dir con Makefile o backend/.
    3) Fallback al cwd.
    """
    env_root = os.getenv("NC_PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "Makefile").exists() or (p / "backend").is_dir():
            return p

    return Path.cwd().resolve()


def project_root(*, refresh: bool = False) -> Path:
    """Retorna la raíz del repo (ver `_find_project_root`)."""
    # Cache simple para evitar recalcular en cada call
    # (pero damos opción de refresh por tests si se necesita).
    global _PROJECT_ROOT_CACHE
    if refresh or _PROJECT_ROOT_CACHE is None:
        _PROJECT_ROOT_CACHE = _find_project_root()
    return _PROJECT_ROOT_CACHE


_PROJECT_ROOT_CACHE: Optional[Path] = None


def artifacts_dir(*, refresh: bool = False) -> Path:
    """Retorna el directorio base de artifacts.

    - Prioriza NC_ARTIFACTS_DIR si está definido.
    - Si no, usa <repo_root>/artifacts.
    """
    global _ARTIFACTS_DIR_CACHE

    env_art = os.getenv("NC_ARTIFACTS_DIR")
    if env_art:
        # Si el env var cambia entre calls, refrescamos cache automáticamente.
        p = Path(env_art).expanduser().resolve()
        _ARTIFACTS_DIR_CACHE = p
        return p

    if refresh or _ARTIFACTS_DIR_CACHE is None:
        _ARTIFACTS_DIR_CACHE = (project_root(refresh=refresh) / "artifacts").resolve()

    return _ARTIFACTS_DIR_CACHE


_ARTIFACTS_DIR_CACHE: Optional[Path] = None


def rel_artifact_path(path: Path) -> str:
    """Devuelve un path 'lógico' para respuestas HTTP/logs.

    Si `path` cae dentro de `artifacts_dir()`, retorna `artifacts/<rel>`.
    Si no, retorna el path absoluto (string).

    Esto mantiene coherencia con P0/P1 (que exponen artifact_path como 'artifacts/...').

    Args:
        path: Path real (absoluto o relativo).

    Returns:
        Ruta lógica tipo 'artifacts/...' o string absoluto.
    """
    p = Path(path).expanduser().resolve()
    base = artifacts_dir().expanduser().resolve()
    try:
        rel = p.relative_to(base)
        return str(Path("artifacts") / rel).replace("\\", "/")
    except Exception:
        return str(p)


def abs_artifact_path(ref: str | Path) -> Path:
    """Resuelve una referencia a un path absoluto.

    Reglas:
    - Si `ref` es absoluto -> se respeta.
    - Si empieza por 'artifacts/' -> se resuelve dentro de `artifacts_dir()`.
    - Si es relativo (no artifacts/) -> se resuelve relativo a `project_root()`.

    Args:
        ref: referencia (string o Path).

    Returns:
        Path absoluto.
    """
    if isinstance(ref, Path):
        p = ref
    else:
        p = Path(str(ref))

    if p.is_absolute():
        return p.expanduser().resolve()

    s = str(p).replace("\\", "/")
    if s.startswith("artifacts/"):
        tail = s[len("artifacts/") :]
        return (artifacts_dir() / tail).expanduser().resolve()

    return (project_root() / p).expanduser().resolve()


# ---------------------------------------------------------------------------
# Dataclasses de contratos de paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturePackPaths:
    """Rutas relevantes de un feature-pack para un dataset_id."""
    dataset_id: str
    base_dir: Path
    train_matrix: Path
    meta: Path
    pair_matrix: Path
    pair_meta: Path


def resolve_feature_pack_paths(dataset_id: str) -> FeaturePackPaths:
    """Resuelve rutas del feature-pack para `dataset_id`.

    Nota:
    - No valida existencia; eso lo debe hacer el caller (router/service) para decidir 404/422.
    - Retorna paths tanto row-level como pair-level (pair puede no existir en datasets sin pairs).

    Args:
        dataset_id: Identificador del dataset.

    Returns:
        FeaturePackPaths con paths esperados.
    """
    ds = safe_segment(dataset_id)
    base = artifacts_dir() / "features" / ds
    return FeaturePackPaths(
        dataset_id=ds,
        base_dir=base,
        train_matrix=base / "train_matrix.parquet",
        meta=base / "meta.json",
        pair_matrix=base / "pair_matrix.parquet",
        pair_meta=base / "pair_meta.json",
    )


def resolve_run_dir(run_id: str) -> Path:
    """Directorio de un run: artifacts/runs/<run_id>/."""
    rid = safe_segment(run_id)
    return artifacts_dir() / "runs" / rid


def resolve_champion_json_candidates(*, dataset_id: str, family: Optional[str]) -> list[Path]:
    """Lista de candidatos a champion.json (layout nuevo + fallback legacy)."""
    ds = safe_segment(dataset_id)
    fam = safe_segment(family) if family else None

    out: list[Path] = []

    # Layout nuevo: artifacts/champions/<family>/<dataset_id>/champion.json
    if fam:
        out.append(artifacts_dir() / "champions" / fam / ds / "champion.json")

    # Layout legacy/mirror: artifacts/champions/<dataset_id>/champion.json
    out.append(artifacts_dir() / "champions" / ds / "champion.json")
    return out


def resolve_champion_json_path(*, dataset_id: str, family: Optional[str]) -> Path:
    """Retorna el path canónico (preferido) para champion.json.

    No garantiza que exista; solo devuelve el path preferido (layout nuevo si hay family).
    """
    ds = safe_segment(dataset_id)
    if family:
        return artifacts_dir() / "champions" / safe_segment(family) / ds / "champion.json"
    return artifacts_dir() / "champions" / ds / "champion.json"


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    """Retorna el primer path existente en `paths`."""
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None
