# tests/api/test_dashboard_rankings_order.py
"""Tests de integración para /dashboard/rankings (orden asc/desc).

Objetivo
--------
Verificar que el endpoint respete el parámetro `order` y que:
- `order=desc` devuelva los mejores primero.
- `order=asc` devuelva los peores primero.

Esto previene regresiones donde el frontend reordena localmente y termina
mostrando la misma lista para "Top Mejores" y "A Intervenir".

Nota
----
El schema del item de ranking puede variar ligeramente (por ejemplo `key`, `name`,
`label` o incluso `docente/asignatura/...`), por eso el test detecta el campo de
identidad de forma robusta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi.testclient import TestClient

from neurocampus.app.main import app

import neurocampus.app.routers.dashboard as dash_router
import neurocampus.dashboard.queries as dash_queries
import neurocampus.historico.manifest as hist_manifest


def _prep_tmp_historico(tmp_path: Path) -> None:
    """Crea un histórico mínimo (processed + manifest) en tmp_path/historico."""
    hist_dir = tmp_path / "historico"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Tres docentes con scores distintos (0–50)
    processed = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-1"],
            "docente": ["Alice", "Bob", "Carla"],
            "asignatura": ["MAT101", "MAT101", "MAT101"],
            "programa": ["Ing", "Ing", "Ing"],
            "score_total": [48.0, 35.0, 42.0],
        }
    )
    processed.to_parquet(hist_dir / "unificado.parquet")

    manifest = {
        "version": 1,
        "updated_at": "2026-02-16T00:00:00+00:00",
        "periodos_disponibles": ["2024-1"],
        "modes": {
            "acumulado": {
                "updated_at": "2026-02-16T00:00:00+00:00",
                "paths": {"parquet": "historico/unificado.parquet"},
                "row_counts": {"rows": int(len(processed))},
                "datasets": ["2024-1"],
            }
        },
    }
    (hist_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _patch_dashboard_paths(tmp_path: Path, monkeypatch) -> None:
    """Redirige rutas del Dashboard a tmp_path para tests herméticos."""
    hist_dir = tmp_path / "historico"

    monkeypatch.setattr(dash_router, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(dash_router, "HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_router, "MANIFEST_PATH", hist_dir / "manifest.json", raising=True)
    monkeypatch.setattr(dash_router, "UNIFICADO_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(dash_router, "UNIFICADO_LABELED_PATH", hist_dir / "unificado_labeled.parquet", raising=True)

    monkeypatch.setattr(dash_queries, "_HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_queries, "_PROCESSED_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(dash_queries, "_LABELED_PATH", hist_dir / "unificado_labeled.parquet", raising=True)

    # Manifest resolver (si se usa en alguna ruta)
    monkeypatch.setattr(hist_manifest, "_find_project_root", lambda: tmp_path, raising=True)

    # Evita contaminación de cachés (si load_* usa lru_cache)
    for fn_name in ("load_processed", "load_labeled"):
        fn = getattr(dash_queries, fn_name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _items(body: Any) -> List[Dict[str, Any]]:
    if isinstance(body, dict) and "items" in body:
        return list(body["items"] or [])
    if isinstance(body, list):
        return list(body)
    raise AssertionError(f"Formato inesperado en rankings: {type(body)} -> {body!r}")


def _get_identity(it: Dict[str, Any]) -> str:
    """Extrae el campo identidad del ranking de forma robusta.

    Soporta variaciones comunes del schema: `key`, `name`, `label`, o nombres
    específicos (`docente`, `asignatura`, etc.). Si no existe ninguno, toma el
    primer campo distinto de `value`.
    """
    for k in (
        "key",
        "name",
        "label",
        "docente",
        "profesor",
        "asignatura",
        "materia",
        "programa",
        "codigo_materia",
        "grupo",
        "cedula_profesor",
        "text",
    ):
        if k in it and it[k] is not None:
            return str(it[k])

    for k, v in it.items():
        if k != "value":
            return str(v)

    raise AssertionError(f"Item de ranking sin identidad: {it!r}")


def test_rankings_respects_order(tmp_path, monkeypatch):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    base_params = {
        "by": "docente",
        "metric": "score_promedio",
        "limit": 10,
        "periodo_from": "2024-1",
        "periodo_to": "2024-1",
        # sin filtros extra
    }

    r_desc = client.get("/dashboard/rankings", params={**base_params, "order": "desc"})
    assert r_desc.status_code == 200, r_desc.text
    desc = _items(r_desc.json())

    r_asc = client.get("/dashboard/rankings", params={**base_params, "order": "asc"})
    assert r_asc.status_code == 200, r_asc.text
    asc = _items(r_asc.json())

    assert len(desc) >= 3
    assert len(asc) >= 3

    # Verifica ordenamiento por valor
    desc_vals = [float(it.get("value") or 0) for it in desc[:3]]
    asc_vals = [float(it.get("value") or 0) for it in asc[:3]]
    assert desc_vals == sorted(desc_vals, reverse=True)
    assert asc_vals == sorted(asc_vals)

    # Top-1 debe diferir entre asc y desc (mejor vs peor)
    desc_id = _get_identity(desc[0])
    asc_id = _get_identity(asc[0])
    assert desc_id != asc_id
    assert float(desc[0].get("value") or 0) > float(asc[0].get("value") or 0)
