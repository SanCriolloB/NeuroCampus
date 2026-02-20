"""
tests/unit/test_metrics_contract.py
=====================================
Tests unitarios para metrics_contract.py (P2 Parte 4).
Sin dependencias de torch, FastAPI ni el backend completo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend/src"))

import pytest
from neurocampus.models.utils.metrics_contract import (
    primary_metric_for_family,
    standardize_run_metrics,
    METRICS_VERSION,
)


# ---------------------------------------------------------------------------
# primary_metric_for_family
# ---------------------------------------------------------------------------

def test_sentiment_desempeno_primary_metric():
    pm, mode = primary_metric_for_family("sentiment_desempeno")
    assert pm == "val_f1_macro"
    assert mode == "max"


def test_score_docente_primary_metric():
    pm, mode = primary_metric_for_family("score_docente")
    assert pm == "val_rmse"
    assert mode == "min"


def test_unknown_family_classification_fallback():
    pm, mode = primary_metric_for_family("familia_inexistente", task_type="classification")
    assert pm == "val_f1_macro"
    assert mode == "max"


def test_unknown_family_regression_fallback():
    pm, mode = primary_metric_for_family("familia_inexistente", task_type="regression")
    assert pm == "val_rmse"
    assert mode == "min"


def test_unknown_family_no_task_type():
    pm, mode = primary_metric_for_family("", task_type="")
    assert pm == "loss"
    assert mode == "min"


# ---------------------------------------------------------------------------
# standardize_run_metrics — score_docente (regresión)
# ---------------------------------------------------------------------------

def test_score_docente_val_rmse_present():
    raw = {"val_rmse": 5.2, "val_mae": 3.1, "val_r2": 0.8}
    out = standardize_run_metrics(raw, family="score_docente", task_type="regression")
    assert out["primary_metric"] == "val_rmse"
    assert out["primary_metric_mode"] == "min"
    assert out["primary_metric_value"] == pytest.approx(5.2)
    assert out["metrics_version"] == METRICS_VERSION


def test_score_docente_val_rmse_absent_fallback_to_train():
    raw = {"train_rmse": 6.0, "train_mae": 4.0}
    out = standardize_run_metrics(raw, family="score_docente", task_type="regression")
    # primary_metric es val_rmse pero no está, fallback → train_rmse
    assert out["primary_metric"] == "val_rmse"
    assert out["primary_metric_value"] == pytest.approx(6.0)


def test_score_docente_nada_disponible():
    raw = {"loss": 100.0}
    out = standardize_run_metrics(raw, family="score_docente", task_type="regression")
    assert out["primary_metric_value"] is None
    # Campos requeridos existen pero son None
    assert "val_rmse" in out
    assert out["val_rmse"] is None


# ---------------------------------------------------------------------------
# standardize_run_metrics — sentiment_desempeno (clasificación)
# ---------------------------------------------------------------------------

def test_sentiment_val_f1_present():
    raw = {"val_f1_macro": 0.75, "val_accuracy": 0.80}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    assert out["primary_metric"] == "val_f1_macro"
    assert out["primary_metric_mode"] == "max"
    assert out["primary_metric_value"] == pytest.approx(0.75)


def test_sentiment_val_f1_absent_fallback_to_train():
    raw = {"f1_macro": 0.65}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    # val_f1_macro no está → fallback a f1_macro (train)
    assert out["primary_metric_value"] == pytest.approx(0.65)


def test_sentiment_required_fields_created_as_none():
    raw = {}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    assert "val_f1_macro" in out
    assert "val_accuracy" in out
    assert "f1_macro" in out
    assert "accuracy" in out
    assert out["val_f1_macro"] is None


# ---------------------------------------------------------------------------
# Legacy key mapping
# ---------------------------------------------------------------------------

def test_legacy_f1_mapped_to_f1_macro():
    raw = {"f1": 0.72}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    assert out["f1_macro"] == pytest.approx(0.72)
    # y es usable como fallback
    assert out["primary_metric_value"] == pytest.approx(0.72)


def test_legacy_acc_mapped_to_accuracy():
    raw = {"acc": 0.85}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    assert out["accuracy"] == pytest.approx(0.85)


def test_legacy_does_not_overwrite_if_new_key_present():
    raw = {"f1": 0.50, "f1_macro": 0.72}
    out = standardize_run_metrics(raw, family="sentiment_desempeno", task_type="classification")
    assert out["f1_macro"] == pytest.approx(0.72)  # no pisado


# ---------------------------------------------------------------------------
# Idempotencia y preservación de campos originales
# ---------------------------------------------------------------------------

def test_does_not_delete_original_fields():
    raw = {"loss": 99.0, "recon_error": 10.0, "custom_field": "hola"}
    out = standardize_run_metrics(raw, family="score_docente", task_type="regression")
    assert out["loss"] == pytest.approx(99.0)
    assert out["recon_error"] == pytest.approx(10.0)
    assert out["custom_field"] == "hola"


def test_metrics_version_always_present():
    out = standardize_run_metrics({}, family="score_docente", task_type="regression")
    assert out["metrics_version"] == METRICS_VERSION
