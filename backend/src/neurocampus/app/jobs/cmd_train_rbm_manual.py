"""
Entrena un modelo RBM/BM manual (NumPy) sobre un dataset preprocesado.

Este script forma parte del pipeline de experimentación de NeuroCampus y se
usa principalmente como comando de línea de comandos.

Resumen de funcionalidad
------------------------

- Lee un archivo de entrada en formato ``.parquet`` o ``.csv``.
- Selecciona únicamente las columnas numéricas.
- Entrena el modelo y calcula métricas de calidad:

  - Error cuadrático medio de reconstrucción (MSE).
  - Opcionalmente, una clasificación proxy con ``LogisticRegression`` sobre
    las representaciones ocultas para predecir la etiqueta
    ``sentiment_label_teacher``.

- Guarda un reporte JSON en el directorio indicado por ``--out-dir`` con
  parámetros y métricas.

Modos de entrenamiento
----------------------

- RBM: usa ``RestrictedBoltzmannMachine`` junto con ``RBMTrainer`` para
  gestionar entrenamiento, early stopping, callbacks y registro de métricas.
- BM: utiliza ``BMManualStrategy`` para mantener compatibilidad con la
  versión previa del pipeline.

El punto de entrada principal es la función :func:`main`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neurocampus.models.rbm_manual import RestrictedBoltzmannMachine
from neurocampus.trainers.rbm_trainer import RBMTrainer
from neurocampus.models.strategies.bm_manual_strategy import BMManualStrategy


def _load_table(path: str) -> pd.DataFrame:
    """
    Carga una tabla desde disco en formato ``.parquet`` o ``.csv``.

    Parameters
    ----------
    path:
        Ruta al archivo de entrada.

    Returns
    -------
    pandas.DataFrame
        Tabla con los datos cargados.
    """
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extrae únicamente las columnas numéricas de un DataFrame y las devuelve
    como matriz NumPy de ``float32`` rellenando ausentes con cero.

    Parameters
    ----------
    df:
        DataFrame de entrada.

    Returns
    -------
    numpy.ndarray
        Matriz de características numéricas.
    """
    X = (
        df.select_dtypes(include=[np.number])
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return X


def _reconstruction_mse(model_like, X: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio (MSE) de reconstrucción.

    Se asume que ``model_like`` implementa el método
    ``reconstruct(X) -> X_rec``, que devuelve una reconstrucción del
    mismo tamaño que ``X``.

    Parameters
    ----------
    model_like:
        Modelo con un método ``reconstruct``.
    X:
        Matriz de entrada original.

    Returns
    -------
    float
        Valor de MSE entre ``X`` y su reconstrucción.
    """
    X_rec = model_like.reconstruct(X)
    return float(mean_squared_error(X, X_rec))


def _logreg_proxy_on_hidden(model_like, df: pd.DataFrame, X: np.ndarray) -> dict:
    """
    Entrena un clasificador lineal sobre las representaciones ocultas como
    proxy de calidad del embedding, si existe la etiqueta
    ``sentiment_label_teacher``.

    El procedimiento es:

    - Filtrar filas con etiqueta no vacía.
    - Obtener las representaciones ocultas ``H`` a partir de ``X``.
    - Normalizar ``H`` con ``StandardScaler``.
    - Entrenar una ``LogisticRegression`` con un split holdout simple (80/20).
    - Devolver métricas de exactitud y F1 macro.

    Se asume que ``model_like`` implementa al menos uno de:

    * ``transform_hidden(X) -> H`` (por ejemplo, ``RBMManual``).
    * ``transform(X) -> H`` (estrategias que solo exponen ``transform``).

    Parameters
    ----------
    model_like:
        Modelo que proporciona representaciones ocultas.
    df:
        DataFrame original con la columna ``sentiment_label_teacher``.
    X:
        Matriz numérica correspondiente a ``df``.

    Returns
    -------
    dict
        Diccionario con información sobre si se habilitó el proxy y, en caso
        afirmativo, métricas de entrenamiento y prueba.
    """
    if "sentiment_label_teacher" not in df.columns:
        return {"enabled": False}

    y_raw = df["sentiment_label_teacher"].astype(str).fillna("")
    mask = y_raw != ""
    if mask.sum() < 50:
        return {"enabled": False, "reason": "insuficiente etiquetado"}

    y_raw = y_raw[mask]
    X_sub = X[mask.values]

    # Representaciones ocultas
    if hasattr(model_like, "transform_hidden"):
        H = model_like.transform_hidden(X_sub)
    else:
        H = model_like.transform(X_sub)

    scaler = StandardScaler()
    Hs = scaler.fit_transform(H)

    # Split holdout 80/20 reproducible
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
    """
    Punto de entrada del script de entrenamiento RBM/BM manual.

    Lee argumentos de línea de comandos, carga el dataset preprocesado,
    entrena el modelo especificado (RBM o BM) y genera un reporte JSON
    con parámetros y métricas en el directorio de salida.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Entrenamiento manual de RBM/BM (NumPy) con métricas de "
            "reconstrucción y proxy de clasificación."
        )
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
        help="Directorio donde guardar reporte JSON y métricas de entrenamiento",
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
        help="Número máximo de epochs",
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

    # Parámetros específicos de entrenamiento tipo CD-k / PCD (para RBM)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df = _load_table(args.src)
    X = _numeric_matrix(df)

    # ----------------------------------------
    # Entrenamiento según tipo de modelo
    # ----------------------------------------
    trainer_metrics = None  # historial de entrenamiento (solo RBM)

    if args.model == "rbm":
        # Entrenamiento de RBM con RBMTrainer
        rbm = RestrictedBoltzmannMachine(
            n_visible=X.shape[1],
            n_hidden=args.n_hidden,
            learning_rate=args.lr,
            seed=args.seed,
            l2=args.l2,
            clip_grad=args.clip_grad,
            binarize_input=args.binarize_input,
            input_bin_threshold=args.input_bin_threshold,
            cd_k=args.cd_k,
            use_pcd=args.pcd,
        )

        trainer = RBMTrainer(
            model=rbm,
            out_dir=str(out_dir),
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=5,
        )

        def log_callback(epoch: int, metrics: dict) -> None:
            """
            Callback sencillo para registrar por consola el avance
            de cada epoch durante el entrenamiento de la RBM.
            """
            print(
                f"[RBMTrainer] epoch={epoch:03d} "
                f"mse_recon={metrics.get('mse_recon', float('nan')):.6f} "
                f"time={metrics.get('time_sec', float('nan')):.2f}s"
            )

        trainer.add_callback(log_callback)
        trainer.fit(X)
        trainer_metrics = trainer.history  # lista de dicts

        model_for_metrics = rbm
        params = {
            "type": "rbm",
            "n_visible": rbm.n_visible,
            "n_hidden": rbm.n_hidden,
            "learning_rate": rbm.learning_rate,
            "l2": rbm.l2,
            "clip_grad": rbm.clip_grad,
            "binarize_input": rbm.binarize_input,
            "input_bin_threshold": rbm.input_bin_threshold,
            "cd_k": rbm.cd_k,
            "use_pcd": rbm.use_pcd,
        }

    else:
        # Entrenamiento de BM mediante estrategia BMManualStrategy
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
        strat.fit(X)
        model_for_metrics = strat
        params = strat.get_params()

    # Métricas de reconstrucción y proxy de clasificación
    mse = _reconstruction_mse(model_for_metrics, X)
    proxy = _logreg_proxy_on_hidden(model_for_metrics, df, X)

    report = {
        "dataset": args.src,
        "model": args.model,
        "params": params,
        "metrics": {
            "reconstruction_mse": mse,
            "proxy_logreg_on_hidden": proxy,
            "trainer_history": trainer_metrics,  # solo lleno para RBM
        },
    }

    out_path = out_dir / f"report_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # También se imprime el reporte por consola para inspección rápida.
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
