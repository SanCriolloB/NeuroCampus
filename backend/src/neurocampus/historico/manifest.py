"""neurocampus.historico.manifest

Manifest del histórico (fuente de verdad para Dashboard).

El Dashboard **NO** debe leer datasets individuales. En su lugar, consulta:
- historico/unificado.parquet (processed histórico)
- historico/unificado_labeled.parquet (labeled histórico)

Para que el frontend pueda:
- saber si el histórico está listo,
- conocer cuándo se actualizó por última vez,
- listar periodos disponibles sin tener que leer/parquear archivos grandes,

...mantenemos un archivo de metadatos pequeño y estable:

    historico/manifest.json

Estructura (v1):

{
  "version": 1,
  "updated_at": "2026-02-14T00:00:00+00:00",
  "periodos_disponibles": ["2024-1", "2024-2"],
  "modes": {
    "acumulado": {
      "updated_at": "...",
      "paths": {"parquet": "historico/unificado.parquet"},
      "row_counts": {"rows": 123},
      "datasets": ["2024-1", "2024-2"]
    },
    "acumulado_labeled": { ... }
  }
}

Notas de diseño:
- El manifest se escribe de forma **atómica** (tmp + os.replace) para evitar
  archivos corruptos si el proceso se interrumpe.
- Incluimos un lock en memoria (threading.Lock) para proteger escrituras dentro
  del mismo proceso (útil en background tasks / endpoints).
- No se implementa file-lock aquí; el control de concurrencia "Eager Update" se
  aborda en la Fase B (lock a nivel de job/unificación).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union


_WRITE_LOCK: Lock = Lock()


MANIFEST_VERSION: int = 1
MANIFEST_RELATIVE_PATH: Path = Path("historico") / "manifest.json"


def _find_project_root() -> Path:
    """Encuentra la raíz del repo de NeuroCampus de forma robusta.

    Criterio:
    - Un directorio que contenga `data/` y `datasets/`.
    - Si no se encuentra, fallback al layout estándar (parents[5]).

    Esta función replica el criterio utilizado en otras partes del backend
    (p.ej. UnificacionStrategy) para evitar errores cuando el CWD no es el repo.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data").exists() and (p / "datasets").exists():
            return p
    # Fallback conservador (layout actual del proyecto).
    return here.parents[5]


def _manifest_path() -> Path:
    """Devuelve la ruta absoluta al manifest del histórico."""
    return (_find_project_root() / MANIFEST_RELATIVE_PATH).resolve()


def _now_iso() -> str:
    """Timestamp ISO-8601 en UTC (estable para logs/consumidores)."""
    return datetime.now(timezone.utc).isoformat()


def _empty_manifest() -> Dict[str, Any]:
    """Estructura mínima del manifest (v1)."""
    return {
        "version": MANIFEST_VERSION,
        "updated_at": None,
        "periodos_disponibles": [],
        "modes": {},
    }


def load_manifest() -> Dict[str, Any]:
    """Carga el manifest del histórico.

    Returns
    -------
    dict
        Manifest deserializado. Si no existe, retorna un manifest vacío (v1).

    Notas
    -----
    - Si el JSON está corrupto, retorna un manifest vacío y conserva la
      trazabilidad incluyendo una marca `corrupt_manifest=True`.
    """
    path = _manifest_path()
    if not path.exists():
        return _empty_manifest()

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        m = _empty_manifest()
        m["corrupt_manifest"] = True
        return m

    # Normalización defensiva para evitar KeyError en consumidores.
    if not isinstance(data, dict):
        return _empty_manifest()

    data.setdefault("version", MANIFEST_VERSION)
    data.setdefault("updated_at", None)
    data.setdefault("periodos_disponibles", [])
    data.setdefault("modes", {})
    return data


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Escribe JSON de forma atómica (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Escritura en texto para preservar legibilidad del manifest.
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update_manifest(
    mode: str,
    dataset_id: Optional[Union[str, List[str]]],
    paths: Dict[str, str],
    row_counts: Dict[str, int],
    updated_at: Optional[Union[str, datetime]] = None,
) -> Dict[str, Any]:
    """Actualiza el manifest del histórico.

    Parameters
    ----------
    mode:
        Nombre del modo que se actualiza. Valores esperados (plan):
        - ``acumulado``
        - ``acumulado_labeled``
    dataset_id:
        Identificador del dataset que disparó la actualización.
        Para compatibilidad, se acepta también una lista de dataset_ids cuando
        el caller ya conoce todos los periodos incluidos (unificación completa).
    paths:
        Rutas relativas (a repo root) de artefactos escritos, por ejemplo:
        ``{"parquet": "historico/unificado.parquet"}``.
    row_counts:
        Conteos asociados, por ejemplo:
        ``{"rows": 12345}``.
    updated_at:
        Timestamp ISO o datetime. Si es None, se usa ahora en UTC.

    Returns
    -------
    dict
        Manifest actualizado (en memoria) tal como quedó persistido.

    Contratos / decisiones
    ----------------------
    - `periodos_disponibles` se actualiza con:
      - la lista completa si `dataset_id` es list[str], o
      - agregando el `dataset_id` si es str.
    - `modes[mode].datasets` guarda el mejor "hint" disponible para ese modo.
    """
    ts: str
    if updated_at is None:
        ts = _now_iso()
    elif isinstance(updated_at, datetime):
        # Normalizar a UTC para evitar mezclas de tz.
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        ts = updated_at.astimezone(timezone.utc).isoformat()
    else:
        ts = str(updated_at)

    with _WRITE_LOCK:
        manifest = load_manifest()
        manifest["version"] = MANIFEST_VERSION
        manifest["updated_at"] = ts

        modes = manifest.setdefault("modes", {})
        entry = modes.setdefault(mode, {})
        entry["updated_at"] = ts
        entry["paths"] = dict(paths or {})
        entry["row_counts"] = dict(row_counts or {})

        # Actualizar datasets del modo
        existing: List[str] = list(entry.get("datasets") or [])
        if isinstance(dataset_id, list):
            # Caller ya conoce el universo de periodos incluidos.
            existing = [str(x) for x in dataset_id]
        elif isinstance(dataset_id, str) and dataset_id.strip():
            if dataset_id not in existing:
                existing.append(dataset_id)
        entry["datasets"] = existing

        # Actualizar periodos_disponibles (hint global para UI)
        periodos: List[str] = list(manifest.get("periodos_disponibles") or [])
        if isinstance(dataset_id, list):
            periodos = [str(x) for x in dataset_id]
        elif isinstance(dataset_id, str) and dataset_id.strip():
            if dataset_id not in periodos:
                periodos.append(dataset_id)
        else:
            # Fallback: unión de datasets presentes en todos los modos
            union: set[str] = set()
            for v in modes.values():
                for ds in (v.get("datasets") or []):
                    union.add(str(ds))
            if union:
                periodos = sorted(union)

        # Normalización final: ordenar para UX consistente
        manifest["periodos_disponibles"] = sorted({p for p in periodos if str(p).strip()})

        path = _manifest_path()
        _atomic_write_json(path, manifest)
        return manifest


def list_periodos_from_manifest() -> List[str]:
    """Lista periodos disponibles usando el manifest como fallback rápido.

    Esta función existe para que endpoints/UI puedan poblar selectores de periodo
    sin leer el parquet completo.

    Returns
    -------
    list[str]
        Lista ordenada de periodos. Si no hay manifest o está vacío, retorna [].
    """
    m = load_manifest()
    periodos = m.get("periodos_disponibles") or []
    if periodos:
        return sorted([str(p) for p in periodos if str(p).strip()])

    # Fallback: intentar inferir desde modes
    modes = m.get("modes") or {}
    union: set[str] = set()
    for v in modes.values():
        for ds in (v.get("datasets") or []):
            union.add(str(ds))
    return sorted([p for p in union if p.strip()])
