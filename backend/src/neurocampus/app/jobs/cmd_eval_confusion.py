# backend/src/neurocampus/app/jobs/cmd_eval_confusion.py
"""
Imprime matriz de confusión y clasificación para:
- un job específico (--job-dir), o
- el campeón de una familia (--family), o
- por defecto, el último job entrenado.

Uso:
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_eval_confusion --family with_text
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_eval_confusion --job-dir artifacts/jobs/20250101_120000
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def _latest_job_meta():
    metas = sorted(Path("artifacts/jobs").glob("*/job_meta.json"), key=lambda p: p.stat().st_mtime)
    return metas[-1] if metas else None

def _champion_meta_for_family(family: str):
    fam = Path("artifacts/champions") / family
    best = fam / "best_meta.json"
    if best.exists():
        return best
    lat = fam / "latest.txt"
    if lat.exists():
        job_dir = Path(lat.read_text(encoding="utf-8").strip())
        return job_dir / "job_meta.json"
    return None

def _predict_from_meta(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job_dir = meta_path.parent
    # Si el job guardó predicciones de validación, úsalo
    val_preds = job_dir / "val_preds.parquet"
    if val_preds.exists():
        dfp = pd.read_parquet(val_preds)
        y_true = dfp["y_true"].astype(str)
        y_pred = dfp["y_pred"].astype(str)
        labels = meta.get("classes") or ["neg","neu","pos"]
        return labels, y_true, y_pred

    # Fallback: recomponer split (puede no coincidir al 100%)
    import joblib
    model = joblib.load(job_dir / "model.pkl")

    data_path = meta["data_path"]
    df = pd.read_parquet(data_path) if data_path.lower().endswith(".parquet") else pd.read_csv(data_path)

    feat_cols = meta.get("feature_cols") or meta.get("feature_columns") or getattr(model, "feat_cols_", None)
    if feat_cols is None:
        raise RuntimeError("No se encontraron columnas de características en job_meta ni en el modelo.")
    X = df[feat_cols].copy()
    y = df[meta.get("target_col","y_sentimiento")].astype(str)

    seed = int(meta.get("hparams",{}).get("seed", 42))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    proba = model.predict_proba(X_va)
    classes = meta.get("classes") or ["neg","neu","pos"]
    import numpy as np
    y_hat = pd.Series(np.argmax(proba, axis=1)).map(dict(enumerate(classes))).astype(str)
    return classes, y_va, y_hat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", default=None, help="Directorio del job (contiene job_meta.json)")
    ap.add_argument("--family", default=None, help="Familia del champion a evaluar (ej. with_text)")
    args = ap.parse_args()

    if args.job_dir:
        mp = Path(args.job_dir) / "job_meta.json"
    elif args.family:
        mp = _champion_meta_for_family(args.family)
    else:
        mp = _latest_job_meta()

    if not mp or not mp.exists():
        raise SystemExit("No se pudo localizar job_meta.json.")

    labels, y_true, y_pred = _predict_from_meta(mp)

    print("job_dir:", mp.parent)
    print("labels:", labels)
    print("\nconfusion_matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nclassification_report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))

if __name__ == "__main__":
    main()
