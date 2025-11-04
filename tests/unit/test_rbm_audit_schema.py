# tests/unit/test_rbm_audit_schema.py
import json, glob, os

def test_metrics_schema_exists():
    runs = glob.glob("artifacts/runs/rbm_audit_*/metrics.json")
    assert runs, "No hay resultados de auditoría. Ejecuta: make rbm-audit"
    with open(sorted(runs)[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "models" in data and isinstance(data["models"], list)
    assert "dataset" in data and "evaluation" in data
    for m in data["models"]:
        assert "name" in m and "summary" in m and "folds" in m
        assert "target" in m
        for k, agg in m["summary"].items():
            assert "mean" in agg and "std" in agg
