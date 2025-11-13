# backend/src/neurocampus/app/jobs/cmd_train_rbm_manual.py
"""
Entrena RBM/BM manual (NumPy) con el mismo estilo del pipeline:
- Lee .parquet/.csv
- Toma solo columnas numéricas
- Entrena y calcula métricas:
  * Reconstrucción MSE
  * (Opcional) Clasificación proxy: LogisticRegression sobre H para predecir 'sentiment_label_teacher'
- Guarda reporte JSON en --out-dir
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neurocampus.models.strategies.rbm_manual_strategy import RBMManualStrategy
from neurocampus.models.strategies.bm_manual_strategy import BMManualStrategy


def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    X = (
        df.select_dtypes(include=[np.number])
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return X


def _reconstruction_mse(model_strategy, X: np.ndarray) -> float:
    X_rec = model_strategy.reconstruct(X)
    return float(mean_squared_error(X, X_rec))


def _logreg_proxy_on_hidden(model_strategy, df: pd.DataFrame, X: np.ndarray) -> dict:
    """
    Si existe 'sentiment_label_teacher' en df, entrena un clasificador lineal en H = transform(X)
    como proxy de calidad de embedding. Split holdout sencillo.
    """
    if "sentiment_label_teacher" not in df.columns:
        return {"enabled": False}

    y_raw = df["sentiment_label_teacher"].astype(str).fillna("")
    mask = y_raw != ""
    if mask.sum() < 50:
        return {"enabled": False, "reason": "insuficiente etiquetado"}

    y_raw = y_raw[mask]
    X_sub = X[mask.values]

    # Hidden features
    H = model_strategy.transform(X_sub)
    scaler = StandardScaler()
    Hs = scaler.fit_transform(H)

    # Holdout simple 80/20
    n = Hs.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(Hs[tr], y[tr])
    yhat = clf.predict(Hs[te])

    return {
        "enabled": True,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "classes": le.classes_.tolist(),
        "acc": float(accuracy_score(y[te], yhat)),
        "f1_macro": float(f1_score(y[te], yhat, average="macro")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Entrenamiento manual de RBM/BM (NumPy) con métricas de reconstrucción y proxy de clasificación."
    )
    ap.add_argument(
        "--in",
        dest="src",
        required=True,
        help="Ruta a parquet/csv preprocesado",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directorio donde guardar reporte JSON",
    )
    ap.add_argument(
        "--model",
        choices=["rbm", "bm"],
        default="rbm",
        help="Tipo de modelo a entrenar (rbm o bm)",
    )

    # Hiperparámetros comunes
    ap.add_argument(
        "--n-hidden",
        type=int,
        default=64,
        help="Número de neuronas ocultas",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número de epochs",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de mini-batch",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidad",
    )
    ap.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="Factor de regularización L2",
    )
    ap.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Umbral de clipping de gradiente",
    )

    # Binarización de entradas
    ap.add_argument(
        "--binarize-input",
        action="store_true",
        help="Si se pasa, binariza las entradas usando input_bin_threshold",
    )
    ap.add_argument(
        "--input-bin-threshold",
        type=float,
        default=0.5,
        help="Umbral para binarizar las entradas (si --binarize-input está activo)",
    )

    # Parámetros específicos de entrenamiento tipo CD-k / PCD
    ap.add_argument(
        "--cd-k",
        type=int,
        default=1,
        help="Número de pasos de Gibbs para CD-k (k >= 1)",
    )
    ap.add_argument(
        "--pcd",
        action="store_true",
        help="Usar Persistent Contrastive Divergence (PCD) en lugar de CD-k clásico",
    )

    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df = _load_table(args.src)
    X = _numeric_matrix(df)

    # Instanciar strategy
    if args.model == "rbm":
        strat = RBMManualStrategy(
            n_hidden=args.n_hidden,
            learning_rate=args.lr,
            seed=args.seed,
            l2=args.l2,
            clip_grad=args.clip_grad,
            binarize_input=args.binarize_input,
            input_bin_threshold=args.input_bin_threshold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cd_k=args.cd_k,
            use_pcd=args.pcd,
        )
    else:
        strat = BMManualStrategy(
            n_hidden=args.n_hidden,
            learning_rate=args.lr,
            seed=args.seed,
            l2=args.l2,
            clip_grad=args.clip_grad,
            binarize_input=args.binarize_input,
            input_bin_threshold=args.input_bin_threshold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cd_k=args.cd_k,
            use_pcd=args.pcd,
        )

    # Entrenar
    strat.fit(X)

    # Métricas
    mse = _reconstruction_mse(strat, X)
    proxy = _logreg_proxy_on_hidden(strat, df, X)

    report = {
        "dataset": args.src,
        "model": args.model,
        "params": strat.get_params(),
        "metrics": {
            "reconstruction_mse": mse,
            "proxy_logreg_on_hidden": proxy,
        },
    }

    out_path = Path(args.out_dir) / f"report_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
