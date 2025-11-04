# backend/src/neurocampus/models/audit_kfold.py
"""
Auditoría k-fold para RBM_general y RBM_restringido usando las implementaciones existentes
en neurocampus.models.strategies.*  (NO cambia arquitectura, solo mide baseline).

Uso:
  PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.audit_kfold --config configs/rbm_audit.yaml
"""

from __future__ import annotations
import argparse, os
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef

from neurocampus.utils.metrics_io import load_yaml, prepare_run_dir, write_metrics, save_config_snapshot

# Import dinámico para mantener independencia de nombres internos
def _resolve_model(model_name: str):
    if model_name.lower() == "rbm_general":
        from neurocampus.models.strategies.modelo_rbm_general import ModeloRBMGeneral as RBMGeneral
        return RBMGeneral
    elif model_name.lower() == "rbm_restringido":
        from neurocampus.models.strategies.modelo_rbm_restringida import ModeloRBMRestringida as RBMRestringida
        return RBMRestringida
    raise ValueError(f"Modelo no soportado: {model_name}")

_METRICS = {
    "accuracy": accuracy_score,
    "f1": lambda y, yhat: f1_score(y, yhat, average="binary") if len(np.unique(y))==2 else f1_score(y, yhat, average="macro"),
    "roc_auc": roc_auc_score,
    "precision": lambda y, yhat: precision_score(y, yhat, average="binary") if len(np.unique(y))==2 else precision_score(y, yhat, average="macro"),
    "recall": lambda y, yhat: recall_score(y, yhat, average="binary") if len(np.unique(y))==2 else recall_score(y, yhat, average="macro"),
    "mcc": matthews_corrcoef,
}

def _pick_target_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in ["y", "label", "target", "y_sentimiento", "sentiment_label_teacher"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna objetivo (y/label/target/y_sentimiento/sentiment_label_teacher)")

def _to_numpy(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).astype(np.float32).values
    y = df[target_col].astype(int).values
    return X, y

def _compute_metrics(y_true, y_proba, y_pred, requested: List[str]) -> Dict[str, float]:
    out = {}
    for m in requested:
        if m == "roc_auc":
            # Si multiclase, usar one-vs-rest si es necesario
            try:
                out[m] = _METRICS[m](y_true, y_proba)
            except Exception:
                pass
        elif m in _METRICS:
            out[m] = float(_METRICS[m](y_true, y_pred))
    return out

def run_kfold_audit(df: pd.DataFrame, target: str | None, model_name: str, model_params: Dict[str, Any],
                    n_splits: int, shuffle: bool, stratify: bool, random_seed: int, metrics: List[str]) -> Dict[str, Any]:
    target_col = _pick_target_column(df, target)
    X, y = _to_numpy(df, target_col)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    Model = _resolve_model(model_name)

    folds = []
    per_metric = {m: [] for m in metrics}

    for k, (tr, va) in enumerate(splitter.split(X, y), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # Los strategies existentes exponen .fit(X,y) y .predict_proba(X) / .predict(X)
        model = Model(**model_params)
        model.fit(Xtr, ytr)

        # Probabilidad de clase positiva (o escore); si no existe, usamos predict() como respaldo
        try:
            proba = model.predict_proba(Xva)  # esperado shape [N] o [N, n_classes]
            if proba.ndim == 2 and proba.shape[1] > 1:
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba
        except Exception:
            proba_pos = None

        try:
            yhat = model.predict(Xva)
        except Exception:
            if proba_pos is None:
                raise
            yhat = (proba_pos >= 0.5).astype(int)

        mvals = _compute_metrics(yva, proba_pos if proba_pos is not None else yhat, yhat, metrics)
        for mk, mv in mvals.items():
            per_metric[mk].append(float(mv))
        folds.append({"fold": k, **mvals})

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_metric.items() if v}
    return {"folds": folds, "summary": summary, "target": target_col}

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

    results = {"dataset": cfg["dataset"], "evaluation": cfg["evaluation"], "models": []}

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
