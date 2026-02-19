# tests/api/test_dashboard_kpis_nan.py
"""Tests de regresión para evitar NaN/inf en KPIs del Dashboard.

Motivación
---------
Se detectó un 500 en ``GET /dashboard/kpis`` cuando ``score_promedio`` quedaba
como NaN (p.ej. promedio sobre serie vacía o toda NaN). Starlette serializa JSON
con `allow_nan=False`, por lo que devolver NaN/inf rompe el endpoint.

Estos tests cubren:
- `compute_kpis()` no devuelve NaN/inf (usa None).
- `/dashboard/kpis` responde 200 incluso si el filtro deja el histórico vacío.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from neurocampus.app.main import app

import neurocampus.app.routers.dashboard as dash_router
import neurocampus.dashboard.queries as dash_queries
import neurocampus.historico.manifest as hist_manifest


def _prep_tmp_historico(tmp_path: Path) -> None:
    """Crea un histórico mínimo (processed + manifest) en tmp_path/historico."""
    hist_dir = tmp_path / "historico"
    hist_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-2"],
            "docente": ["Alice", "Bob", "Alice"],
            "asignatura": ["MAT101", "MAT101", "FIS201"],
            "programa": ["Ing", "Ing", "Fis"],
            # Score ya normalizado en 0–50 para dashboard (ej: 45.0 equivale a 4.5/5)
            "score_total": [45.0, 38.0, 40.0],
        }
    )
    df.to_parquet(hist_dir / "unificado.parquet")

    manifest = {
        "version": 1,
        "updated_at": "2026-02-16T00:00:00+00:00",
        "periodos_disponibles": ["2024-1", "2024-2"],
        "modes": {
            "acumulado": {
                "updated_at": "2026-02-16T00:00:00+00:00",
                "paths": {"parquet": "historico/unificado.parquet"},
                "row_counts": {"rows": int(len(df))},
                "datasets": ["2024-1", "2024-2"],
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

    monkeypatch.setattr(hist_manifest, "_find_project_root", lambda: tmp_path, raising=True)


def test_compute_kpis_empty_df_no_nan():
    from neurocampus.dashboard.queries import compute_kpis

    df = pd.DataFrame(columns=["periodo", "docente", "asignatura", "score_total"])
    kpis = compute_kpis(df)

    assert kpis["evaluaciones"] == 0
    assert kpis["score_promedio"] is None


def test_compute_kpis_all_nan_score_no_nan():
    from neurocampus.dashboard.queries import compute_kpis

    df = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1"],
            "docente": ["Alice", "Bob"],
            "asignatura": ["MAT101", "MAT101"],
            "score_total": [float("nan"), None],
        }
    )
    kpis = compute_kpis(df)

    assert kpis["evaluaciones"] == 2
    assert kpis["score_promedio"] is None


def test_dashboard_kpis_filter_empty_returns_200_and_null(tmp_path, monkeypatch):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    # Filtro que deja el histórico vacío: no debe dar 500.
    r = client.get(
        "/dashboard/kpis",
        params={"periodo_from": "2024-1", "periodo_to": "2024-2", "docente": "NO_EXISTE"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["evaluaciones"] == 0
    assert body["score_promedio"] is None
