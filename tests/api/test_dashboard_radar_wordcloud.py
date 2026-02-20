# tests/api/test_dashboard_radar_wordcloud.py
"""Tests de integración (mínimos) para Radar y Wordcloud del Dashboard.

Cubre:
- `/dashboard/radar` retorna 10 items (pregunta_1..pregunta_10) y no todos en 0.
- `/dashboard/wordcloud` usa fallback de columna (cuando `sugerencias_lemmatizadas` existe pero está vacía)
  y retorna tokens desde `texto_lemmas`.

Los tests usan un histórico mínimo creado en un directorio temporal (tmp_path) y monkeypatch
para redirigir las rutas del Dashboard a ese histórico.
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
    """Crea un histórico mínimo (processed + labeled + manifest) en tmp_path/historico."""
    hist_dir = tmp_path / "historico"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Processed: incluye preguntas 1..10 en escala 0-50
    processed = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-2"],
            "docente": ["Alice", "Alice"],
            "asignatura": ["MAT101", "MAT101"],
            "programa": ["Ing", "Ing"],
            "pregunta_1": [40.0, 45.0],
            "pregunta_2": [41.0, 46.0],
            "pregunta_3": [42.0, 47.0],
            "pregunta_4": [43.0, 48.0],
            "pregunta_5": [44.0, 49.0],
            "pregunta_6": [45.0, 50.0],
            "pregunta_7": [40.0, 45.0],
            "pregunta_8": [41.0, 46.0],
            "pregunta_9": [42.0, 47.0],
            "pregunta_10": [43.0, 48.0],
            # Score canónico para otros endpoints (0-50)
            "score_total": [42.0, 46.0],
        }
    )
    processed.to_parquet(hist_dir / "unificado.parquet")

    # Labeled: sugerencias vacías, texto_lemmas con tokens (fallback esperado)
    labeled = pd.DataFrame(
        {
            "periodo": ["2024-2", "2024-2"],
            "docente": ["Alice", "Alice"],
            "asignatura": ["MAT101", "MAT101"],
            "programa": ["Ing", "Ing"],
            "sugerencias_lemmatizadas": [None, None],
            "texto_lemmas": [
                "docente excelente profesora clase",
                "docente excelente clase",
            ],
            "has_text": [1, 1],
            "has_text_processed": [1, 1],
        }
    )
    labeled.to_parquet(hist_dir / "unificado_labeled.parquet")

    manifest = {
        "version": 1,
        "updated_at": "2026-02-16T00:00:00+00:00",
        "periodos_disponibles": ["2024-1", "2024-2"],
        "modes": {
            "acumulado": {
                "updated_at": "2026-02-16T00:00:00+00:00",
                "paths": {"parquet": "historico/unificado.parquet"},
                "row_counts": {"rows": int(len(processed))},
                "datasets": ["2024-1", "2024-2"],
            }
        },
    }
    (hist_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _patch_dashboard_paths(tmp_path: Path, monkeypatch) -> None:
    """Redirige rutas del Dashboard a tmp_path para tests herméticos."""
    hist_dir = tmp_path / "historico"

    # Router: paths usados por endpoints /dashboard/*
    monkeypatch.setattr(dash_router, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(dash_router, "HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_router, "MANIFEST_PATH", hist_dir / "manifest.json", raising=True)
    monkeypatch.setattr(dash_router, "UNIFICADO_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(dash_router, "UNIFICADO_LABELED_PATH", hist_dir / "unificado_labeled.parquet", raising=True)

    # Queries: loaders internos
    monkeypatch.setattr(dash_queries, "_HIST_DIR", hist_dir, raising=True)
    monkeypatch.setattr(dash_queries, "_PROCESSED_PATH", hist_dir / "unificado.parquet", raising=True)
    monkeypatch.setattr(dash_queries, "_LABELED_PATH", hist_dir / "unificado_labeled.parquet", raising=True)

    # Manifest resolver (si se usa)
    monkeypatch.setattr(hist_manifest, "_find_project_root", lambda: tmp_path, raising=True)


@pytest.mark.parametrize("periodo_from,periodo_to", [("2024-1", "2024-2")])
def test_dashboard_radar_returns_items(tmp_path, monkeypatch, periodo_from, periodo_to):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    r = client.get(
        "/dashboard/radar",
        params={"periodo_from": periodo_from, "periodo_to": periodo_to, "docente": "Alice", "asignatura": "MAT101"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert "items" in body
    assert len(body["items"]) == 10

    keys = {it["key"] for it in body["items"]}
    assert keys == {f"pregunta_{i}" for i in range(1, 11)}

    values = [float(it["value"]) for it in body["items"]]
    # No deberían ser todos 0 si hay datos y preguntas válidas
    assert any(v > 0 for v in values)


def test_dashboard_wordcloud_fallback_to_texto_lemmas(tmp_path, monkeypatch):
    _prep_tmp_historico(tmp_path)
    _patch_dashboard_paths(tmp_path, monkeypatch)

    client = TestClient(app)

    r = client.get(
        "/dashboard/wordcloud",
        params={"periodo_from": "2024-2", "periodo_to": "2024-2", "docente": "Alice", "asignatura": "MAT101", "limit": 20},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert "items" in body
    assert len(body["items"]) > 0

    tokens = {it["text"] for it in body["items"]}
    assert "docente" in tokens
    assert "excelente" in tokens
