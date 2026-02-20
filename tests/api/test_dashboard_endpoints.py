# tests/api/test_dashboard_endpoints.py
"""Tests de API para endpoints del Dashboard.

Objetivo
--------
Validar que los endpoints ``/dashboard/*`` se comporten correctamente usando
**exclusivamente histórico**, sin depender de datasets individuales.

Estos tests son herméticos: crean un histórico mínimo en un ``tmp_path`` y
monkeypatch-ean las rutas usadas por los módulos del Dashboard.
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
    """Crea un histórico mínimo en `tmp_path/historico`.

    Estructura creada
    -----------------
    - historico/unificado.parquet
    - historico/manifest.json

    También se crean ``data/`` y ``datasets/`` para mantener compatibilidad con
    heurísticas de discovery en algunos módulos.
    """
    (tmp_path / "historico").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-2", "2024-2"],
            "docente": ["Alice", "Bob", "Alice", "Carla"],
            "asignatura": ["MAT101", "MAT101", "FIS201", "FIS201"],
            "programa": ["Ing", "Ing", "Fis", "Fis"],
            "score": [4.5, 3.8, 4.0, 4.2],
        }
    )
    df.to_parquet(tmp_path / "historico" / "unificado.parquet")

    manifest = {
        "version": 1,
        "updated_at": "2026-02-14T00:00:00+00:00",
        "periodos_disponibles": ["2024-1", "2024-2"],
        "modes": {
            "acumulado": {
                "updated_at": "2026-02-14T00:00:00+00:00",
                "paths": {"parquet": "historico/unificado.parquet"},
                "row_counts": {"rows": 4},
                "datasets": ["2024-1", "2024-2"],
            }
        },
    }
    (tmp_path / "historico" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _patch_dashboard_paths(tmp_path: Path, monkeypatch) -> None:
    """Redirige módulos a `tmp_path` para tests herméticos."""
    hist_dir = tmp_path / "historico"

    # Router: usa constants module-level para existencia/mtime.
    monkeypatch.setattr(dash_router, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(dash_router, "HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_router, "MANIFEST_PATH", hist_dir / "manifest.json", raising=True)
    monkeypatch.setattr(dash_router, "UNIFICADO_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(
        dash_router,
        "UNIFICADO_LABELED_PATH",
        hist_dir / "unificado_labeled.parquet",
        raising=True,
    )

    # Queries: paths se calculan al importar; parcheamos rutas ya resueltas.
    monkeypatch.setattr(dash_queries, "_HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_queries, "_PROCESSED_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(
        dash_queries,
        "_LABELED_PATH",
        hist_dir / "unificado_labeled.parquet",
        raising=True,
    )

    # Manifest: load_manifest depende de _find_project_root.
    monkeypatch.setattr(hist_manifest, "_find_project_root", lambda: tmp_path, raising=True)


def test_dashboard_status_and_periodos_ok(tmp_path, monkeypatch):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    r = client.get("/dashboard/status")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["manifest_exists"] is True
    assert body["manifest_corrupt"] is False
    assert body["periodos_disponibles"] == ["2024-1", "2024-2"]
    assert body["processed"]["exists"] is True
    assert body["ready_processed"] is True
    assert body["labeled"]["exists"] is False
    assert body["ready_labeled"] is False

    r2 = client.get("/dashboard/periodos")
    assert r2.status_code == 200, r2.text
    assert r2.json()["items"] == ["2024-1", "2024-2"]


def test_dashboard_catalogos_y_kpis_filtrados(tmp_path, monkeypatch):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    # Catálogos filtrados por periodo
    r = client.get("/dashboard/catalogos", params={"periodo": "2024-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["docentes"] == ["Alice", "Bob"]
    assert body["asignaturas"] == ["MAT101"]
    assert body["programas"] == ["Ing"]

    # KPIs filtrados por periodo
    r2 = client.get("/dashboard/kpis", params={"periodo": "2024-1"})
    assert r2.status_code == 200, r2.text
    kpis = r2.json()
    assert kpis["evaluaciones"] == 2
    assert kpis["docentes"] == 2
    assert kpis["asignaturas"] == 1
    assert kpis["score_promedio"] == pytest.approx(4.15)

    # KPIs por rango (incl.)
    r3 = client.get(
        "/dashboard/kpis",
        params={"periodo_from": "2024-1", "periodo_to": "2024-2"},
    )
    assert r3.status_code == 200, r3.text
    kpis_rango = r3.json()
    assert kpis_rango["evaluaciones"] == 4
    assert kpis_rango["docentes"] == 3
    assert kpis_rango["asignaturas"] == 2
    assert kpis_rango["score_promedio"] == pytest.approx(4.125)
