# tests/unit/test_rbm_audit_schema.py
"""
Tests sobre el esquema del archivo de métricas generado por la auditoría RBM.

Este test asume que se ha ejecutado previamente `make rbm-audit`, que genera
archivos del tipo:

    artifacts/runs/rbm_audit_YYYYMMDD_HHMMSS/metrics.json

Si no existen esos artefactos, el test se marca como SKIPPED para no romper
el pipeline de pruebas en entornos donde la auditoría no se ha corrido aún.
"""

import json
import glob
import os
import pytest


def test_metrics_schema_exists():
    # Buscamos cualquier run de auditoría previo
    runs = glob.glob("artifacts/runs/rbm_audit_*/metrics.json")

    if not runs:
        pytest.skip(
            "No hay resultados de auditoría. Ejecuta `make rbm-audit` si quieres "
            "validar el esquema de metrics.json."
        )

    # Tomamos el último run y validamos el esquema mínimo esperado
    with open(sorted(runs)[-1], "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data and isinstance(data["models"], list)
    assert "dataset" in data and "evaluation" in data

    for m in data["models"]:
        assert "name" in m and "summary" in m and "folds" in m
        assert "target" in m
        for _, agg in m["summary"].items():
            assert "mean" in agg and "std" in agg
