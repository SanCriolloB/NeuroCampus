# backend/src/neurocampus/models/audit_kfold.py
"""
Auditoría k-fold para RBM_general y RBM_restringido usando las implementaciones existentes
en neurocampus.models.strategies.* (NO cambia arquitectura, solo mide baseline).

Uso:
  PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.audit_kfold --config configs/rbm_audit.yaml
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder

import torch  # para compatibilidad con strategies basadas en PyTorch

from neurocampus.utils.metrics_io import (
    load_yaml,
    prepare_run_dir,
    write_metrics,
    save_config_snapshot,
)

# ---------- Resolución dinámica de modelos ----------
def _resolve_model(model_name: str):
    if model_name.lower() == "rbm_general":
        from neurocampus.models.strategies.modelo_rbm_general import (
            ModeloRBMGeneral as RBMGeneral,
        )
        return RBMGeneral
    elif model_name.lower() == "rbm_restringido":
        from neurocampus.models.strategies.modelo_rbm_restringida import (
            ModeloRBMRestringida as RBMRestringida,
        )
        return RBMRestringida
    raise ValueError(f"Modelo no soportado: {model_name}")

_METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "mcc": matthews_corrcoef,
}

# ---------- Utils de dataset ----------
def _pick_target_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in ["y", "label", "target", "y_sentimiento", "sentiment_label_teacher"]:
        if c in df.columns:
            return c
    raise ValueError(
        "No se encontró columna objetivo (y/label/target/y_sentimiento/sentiment_label_teacher)"
    )

def _to_numpy(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
    X = (
        df.drop(columns=[target_col])
        .select_dtypes(include=[np.number])
        .astype(np.float32)
        .values
    )
    y_raw = df[target_col].values
    if pd.api.types.is_numeric_dtype(df[target_col].dtype):
        y = df[target_col].astype(int).values
        mapping = None
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        mapping = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
    return X, y, mapping

def _coerce_pred_labels(yhat: np.ndarray, mapping: Optional[Dict[str, int]], y_true: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    if np.issubdtype(getattr(yhat, "dtype", None), np.number):
        return yhat.astype(int), mapping
    yhat = np.asarray(yhat)
    if mapping is not None:
        try:
            yhat_int = np.array([mapping[str(lbl)] for lbl in yhat], dtype=int)
            return yhat_int, mapping
        except KeyError:
            pass
    classes_true = np.unique(y_true)
    max_id = int(classes_true.max()) if classes_true.size else -1
    unique_yhat = [str(v) for v in np.unique(yhat)]
    local_map = {lbl: i + max_id + 1 for i, lbl in enumerate(unique_yhat)}
    yhat_int = np.array([local_map[str(lbl)] for lbl in yhat], dtype=int)
    return yhat_int, local_map

# ---------- Dimensión de entrada + Fix interno ----------
_FEATURE_KEYS = ["visible_units", "n_visible", "n_features", "input_dim", "in_features", "v_dim"]

def _inject_feature_dims(model_params: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    params = dict(model_params)
    for k in _FEATURE_KEYS:
        if k in params:
            return params
    params["visible_units"] = n_features
    return params

def _as_tensor_like(ref, array_np: np.ndarray):
    if isinstance(ref, torch.Tensor):
        return torch.tensor(array_np, device=ref.device, dtype=ref.dtype)
    return array_np

def _force_resize_rbm(holder, n_features: int):
    """
    Redimensiona en caliente W y b_v para que W sea (n_features, h).
    'holder' puede ser model.rbm o model.
    """
    W = getattr(holder, "W", None)
    if W is None:
        return False
    # Forma actual
    if isinstance(W, torch.Tensor):
        v, h = W.shape
    else:
        W_np = np.asarray(W)
        v, h = W_np.shape
    if v == n_features:
        return True  # ya coincide

    # Re-armar pesos visibles
    std = 0.01
    W_new_np = np.random.normal(0.0, std, size=(n_features, h)).astype(np.float32)
    W_new = _as_tensor_like(W, W_new_np)

    # b_v
    b_v_old = getattr(holder, "b_v", None)
    if isinstance(b_v_old, torch.Tensor):
        b_v_new = torch.zeros(n_features, device=b_v_old.device, dtype=b_v_old.dtype)
    else:
        b_v_new = np.zeros((n_features,), dtype=np.float32)

    # Asignar
    setattr(holder, "W", W_new)
    setattr(holder, "b_v", b_v_new)

    # Actualizar metadata si existe
    for k in ("n_visible", "visible_units", "input_dim", "in_features", "v_dim"):
        if hasattr(holder, k):
            try:
                setattr(holder, k, n_features)
            except Exception:
                pass
    return True

def _fix_model_internal_shapes(model, n_features: int):
    """
    Intenta arreglar shapes en model.rbm; si no existe, en model.
    Devuelve True si pudo ajustar algo.
    """
    fixed = False
    if hasattr(model, "rbm"):
        try:
            fixed = _force_resize_rbm(model.rbm, n_features) or fixed
        except Exception:
            pass
    try:
        fixed = _force_resize_rbm(model, n_features) or fixed
    except Exception:
        pass
    return fixed

# ---------- Métricas ----------
def _compute_metrics(
    y_true: np.ndarray,
    y_proba_for_metrics: Optional[np.ndarray],
    y_pred: np.ndarray,
    requested: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_classes = len(np.unique(y_true))
    for m in requested:
        if m == "roc_auc":
            try:
                if n_classes == 2:
                    if y_proba_for_metrics is None or y_proba_for_metrics.ndim != 1:
                        continue
                    out[m] = float(roc_auc_score(y_true, y_proba_for_metrics))
                else:
                    if (
                        y_proba_for_metrics is None
                        or y_proba_for_metrics.ndim != 2
                        or y_proba_for_metrics.shape[1] != n_classes
                    ):
                        continue
                    out[m] = float(
                        roc_auc_score(
                            y_true,
                            y_proba_for_metrics,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            except Exception:
                pass
        elif m in _METRICS:
            if m in ("f1", "precision", "recall"):
                avg = "binary" if n_classes == 2 else "macro"
                out[m] = float(_METRICS[m](y_true, y_pred, average=avg, zero_division=0))
            else:
                out[m] = float(_METRICS[m](y_true, y_pred))
    return out

# ---------- Lógica principal ----------
def run_kfold_audit(
    df: pd.DataFrame,
    target: Optional[str],
    model_name: str,
    model_params: Dict[str, Any],
    n_splits: int,
    shuffle: bool,
    stratify: bool,
    random_seed: int,
    metrics: List[str],
) -> Dict[str, Any]:
    target_col = _pick_target_column(df, target)
    X, y, mapping = _to_numpy(df, target_col)
    n_features = X.shape[1]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    Model = _resolve_model(model_name)

    folds: List[Dict[str, float]] = []
    per_metric = {m: [] for m in metrics}

    for k, (tr, va) in enumerate(splitter.split(X, y), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # 1) Inyectar dimensión de entrada al constructor si no existe
        params = _inject_feature_dims(model_params, n_features)
        model = Model(**params)

        # 2) Primer intento de ajuste preventivo (por si la strategy ignora el param)
        _fix_model_internal_shapes(model, n_features)

        # 3) Entrenar con retry si hay mismatch de shapes
        tried_fix_after_error = False
        while True:
            try:
                model.fit(Xtr, ytr)
                break  # ok
            except RuntimeError as e:
                msg = str(e)
                shape_err = "mat1 and mat2 shapes cannot be multiplied" in msg or "size mismatch" in msg
                if not shape_err or tried_fix_after_error:
                    raise  # otro error o ya reintentamos
                # Arreglar y reintentar una única vez
                _fix_model_internal_shapes(model, n_features)
                tried_fix_after_error = True
                continue

        # 4) Probabilidades
        proba_pos: Optional[np.ndarray] = None
        proba_mat: Optional[np.ndarray] = None
        try:
            proba = model.predict_proba(Xva)  # [N] o [N, C]
            if isinstance(proba, list):
                proba = np.asarray(proba)
            if proba.ndim == 1:
                proba_pos = proba
            elif proba.ndim == 2:
                if proba.shape[1] <= 1:
                    proba_pos = proba[:, 0]
                else:
                    proba_mat = proba
                    if proba.shape[1] > 1:
                        proba_pos = proba[:, 1]
        except Exception:
            pass

        # 5) Predicción dura
        try:
            yhat = model.predict(Xva)
        except Exception:
            if proba_pos is not None and len(np.unique(y)) == 2:
                yhat = (proba_pos >= 0.5).astype(int)
            elif proba_mat is not None:
                yhat = np.argmax(proba_mat, axis=1)
            else:
                raise

        # 6) Forzar entero compatible con yva
        yhat = np.asarray(yhat)
        if not np.issubdtype(yhat.dtype, np.number):
            yhat, _mapping_used = _coerce_pred_labels(yhat, mapping, yva)

        # 7) Métricas
        yproba_for_metrics = proba_mat if (proba_mat is not None and len(np.unique(y)) > 2) else proba_pos
        mvals = _compute_metrics(yva, yproba_for_metrics, yhat, metrics)
        for mk, mv in mvals.items():
            per_metric[mk].append(float(mv))
        folds.append({"fold": k, **mvals})

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_metric.items() if v}
    result: Dict[str, Any] = {"folds": folds, "summary": summary, "target": target_col}
    if mapping is not None:
        result["label_mapping"] = mapping
    return result

# ---------- Carga de dataset y CLI ----------
def _load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Extensión no soportada: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    df = _load_dataset(cfg["dataset"]["path"])

    run_dir = prepare_run_dir(cfg["artifacts"]["root"])
    save_config_snapshot(run_dir, args.config)

    results: Dict[str, Any] = {"dataset": cfg["dataset"], "evaluation": cfg["evaluation"], "models": []}
    for mm in cfg["models"]:
        res = run_kfold_audit(
            df=df,
            target=cfg["dataset"].get("target"),
            model_name=mm["name"],
            model_params=mm["params"],
            n_splits=cfg["evaluation"]["n_splits"],
            shuffle=cfg["evaluation"]["shuffle"],
            stratify=cfg["evaluation"]["stratify"],
            random_seed=cfg["evaluation"]["random_seed"],
            metrics=cfg["evaluation"]["metrics"],
        )
        results["models"].append({"name": mm["name"], "params": mm["params"], **res})

    out = write_metrics(run_dir, results)
    print(f"[AUDIT] Métricas escritas en: {out}\nRun dir: {run_dir}")

if __name__ == "__main__":
    main()
