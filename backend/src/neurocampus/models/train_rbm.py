# backend/src/neurocampus/models/train_rbm.py
# Entrena el Student (RBM General o Restringida) a partir de un dataset ya etiquetado
# (humano>teacher). Deja artefactos por job en artifacts/jobs/<JOBID> y un metrics.json
# para que select_champion pueda elegir el mejor.
#
# Ejemplos:
#   python -m neurocampus.models.train_rbm --type general \
#       --data data/labeled/evaluaciones_2025_beto.parquet --job-id auto \
#       --epochs 12 --n-hidden 64 --cd-k 2 --epochs-rbm 2 --scale-mode scale_0_5 --use-text-probs
#
#   python -m neurocampus.models.train_rbm --type restringida \
#       --data data/labeled/evaluaciones_2025_beto.parquet --job-id 2025S2_run1 \
#       --n-hidden 64 --cd-k 2 --epochs 8 --scale-mode scale_0_5

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral
from neurocampus.models.strategies.modelo_rbm_restringida import RBMRestringida

CLASSES = ["neg", "neu", "pos"]


# -----------------------
# Utilidades de evaluación
# -----------------------

def _pick_feature_cols(df: pd.DataFrame, max_n: int = 10, include_text_probs: bool = False) -> List[str]:
    """
    Toma calif_1..calif_10 si existen (ordenadas), y opcionalmente añade p_neg/p_neu/p_pos
    como features adicionales si están presentes y include_text_probs=True.
    """
    cols = [c for c in df.columns if c.startswith("calif_")]
    if not cols:
        return []
    # ordenar por índice numérico
    def _idx(c: str) -> int:
        try:
            return int(c.split("_")[1])
        except Exception:
            return 999
    cols = sorted(cols, key=_idx)[:max_n]

    if include_text_probs and all(k in df.columns for k in ["p_neg", "p_neu", "p_pos"]):
        cols = cols + ["p_neg", "p_neu", "p_pos"]

    return cols


def _resolve_labels(df: pd.DataFrame, require_teacher_accept: bool = True, accept_threshold: float = 0.80) -> np.ndarray:
    """
    Regresa y en {0,1,2} siguiendo humano > teacher, con normalización básica.
    Si no puede resolver nada y require_teacher_accept=True deja todo en -1.
    """
    # Si existe módulo robusto, úsalo
    try:
        from neurocampus.models.data.labels import resolve_sentiment_labels  # type: ignore
        y_ser = resolve_sentiment_labels(df, require_teacher_accept=require_teacher_accept, accept_threshold=accept_threshold)
        y_map = {"neg": 0, "neu": 1, "pos": 2}
        y = np.array([y_map.get(str(v), -1) for v in y_ser], dtype=np.int64)
        return y
    except Exception:
        pass

    # Fallback sencillo (por si no está el módulo):
    if "y_sentimiento" in df.columns:
        y_raw = df["y_sentimiento"].astype(str).str.lower()
    elif "sentiment_label_teacher" in df.columns:
        y_raw = df["sentiment_label_teacher"].astype(str).str.lower()
        if require_teacher_accept:
            if "accepted_by_teacher" in df.columns:
                ok = df["accepted_by_teacher"].fillna(0).astype(int) == 1
                y_raw = y_raw.where(ok)
            elif "sentiment_conf" in df.columns:
                ok = df["sentiment_conf"].fillna(0.0) >= float(accept_threshold)
                y_raw = y_raw.where(ok)
    else:
        return np.full(len(df), -1, dtype=np.int64)

    map_ = {"neg": 0, "negative": 0, "negativo": 0,
            "neu": 1, "neutral": 1,
            "pos": 2, "positive": 2, "positivo": 2}
    y = np.array([map_.get(v, -1) for v in y_raw], dtype=np.int64)
    return y


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels=(0, 1, 2)) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    f1s = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _train_val_split(n: int, seed: int = 42, frac_train: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    cut = int(frac_train * n)
    return idx[:cut], idx[cut:]


# -----------
# Entrenador
# -----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["general", "restringida"], required=True,
                    help="Tipo de Student: RBM general o restringida.")
    ap.add_argument("--data", required=True, help="Ruta CSV/Parquet ya etiquetado (humano>teacher).")
    ap.add_argument("--job-id", default="auto", help="Identificador del job. Usa 'auto' para timestamp.")
    ap.add_argument("--out-dir", default="artifacts/jobs", help="Directorio base donde dejar el job.")

    # Hiperparámetros y opciones
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5, help="Épocas de entrenamiento (ciclos de train_step).")

    ap.add_argument("--n-hidden", type=int, default=None, help="Unidades ocultas RBM (por defecto: 32 general, 64 restringida).")
    ap.add_argument("--cd-k", type=int, default=None, help="Pasos de Contrastive Divergence (por defecto: 1 general, 2 restringida).")
    ap.add_argument("--epochs-rbm", type=int, default=None, help="Pasadas de CD por época (por defecto: 1 general, 2 restringida).")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-rbm", type=float, default=None)
    ap.add_argument("--lr-head", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scale-mode", choices=["minmax", "scale_0_5"], default="scale_0_5",
                    help="Escalado de features: minmax según datos o 0–5 -> [0,1].")

    # Aceptación del Teacher / humano
    ap.add_argument("--accept-teacher", action="store_true", default=True,
                    help="Usar solo pseudolabels aceptados por el Teacher (o etiqueta humana cuando exista).")
    ap.add_argument("--accept-threshold", type=float, default=0.80,
                    help="Umbral de aceptación para sentiment_conf cuando no exista accepted_by_teacher.")

    # GPU
    ap.add_argument("--use-cuda", action="store_true", help="Intenta usar CUDA si está disponible.")

    # Texto como features ligeros
    ap.add_argument("--use-text-probs", action="store_true",
                    help="Añade p_neg/p_neu/p_pos como features, si existen en el dataset.")

    args = ap.parse_args()

    # Instanciar estrategia
    Strat = RBMGeneral if args.type == "general" else RBMRestringida
    strat = Strat()

    # Defaults por tipo
    default_n_hidden   = 32 if args.type == "general" else 64
    default_cd_k       = 1  if args.type == "general" else 2
    default_epochs_rbm = 1  if args.type == "general" else 2
    default_lr_rbm     = 1e-2 if args.type == "general" else 5e-3

    # Hparams para setup
    hparams: Dict = dict(
        seed=args.seed,
        n_hidden=args.n_hidden if args.n_hidden is not None else default_n_hidden,
        cd_k=args.cd_k if args.cd_k is not None else default_cd_k,
        epochs_rbm=args.epochs_rbm if args.epochs_rbm is not None else default_epochs_rbm,
        batch_size=args.batch_size,
        lr_rbm=args.lr_rbm if args.lr_rbm is not None else default_lr_rbm,
        lr_head=args.lr_head,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scale_mode=args.scale_mode,
        accept_teacher=args.accept_teacher,
        accept_threshold=args.accept_threshold,
        use_cuda=args.use_cuda,
        use_text_probs=args.use_text_probs,  # <<--- NUEVO: pasa p_* como features al Student
    )

    # Setup (carga datos, vectoriza, inicializa RBM y cabeza)
    strat.setup(data_ref=args.data, hparams=hparams)

    # Entrenamiento por épocas
    for epoch in range(1, args.epochs + 1):
        loss, metrics = strat.train_step(epoch)
        print({"epoch": epoch, **metrics})

    # --------- Evaluación (holdout simple 80/20) ---------
    df = _load_df(args.data)

    # y final (humano > teacher)
    y_all = _resolve_labels(df, require_teacher_accept=args.accept_teacher, accept_threshold=args.accept_threshold)
    mask_labeled = y_all >= 0

    # features (mismas columnas que el Student)
    feat_cols = _pick_feature_cols(df, include_text_probs=args.use_text_probs)
    if not feat_cols:
        raise RuntimeError("No se encontraron columnas calif_1..calif_10 en el dataset.")
    X_all = df[feat_cols].to_numpy(np.float32)

    # filtrar filas con etiqueta
    X_all = X_all[mask_labeled]
    y_all = y_all[mask_labeled]

    # split (solo para evaluar; el entrenamiento arriba usa todo el set preparado en setup)
    if len(y_all) >= 10:
        tr, va = _train_val_split(len(y_all), seed=args.seed, frac_train=0.8)
    else:
        # si el set es muy pequeño, evalúa sobre todo
        tr = np.arange(len(y_all), dtype=int)
        va = np.arange(len(y_all), dtype=int)

    X_va, y_va = X_all[va], y_all[va]
    proba = strat.predict_proba(X_va)
    y_hat = proba.argmax(axis=1)

    metrics = {
        "f1_macro": _f1_macro(y_va, y_hat),
        "accuracy": _accuracy(y_va, y_hat),
        "classes": CLASSES,
        "n_val": int(len(y_va)),
        "n_labeled_used": int(len(y_all)),
        "n_features": int(X_all.shape[1]),
        "type": args.type,
        "seed": args.seed,
    }

    # Info útil adicional
    labels_dist = {c: int(np.sum(y_all == i)) for i, c in enumerate(CLASSES)}
    metrics["labels_dist_trainable"] = labels_dist

    # Tasa de aceptación (si viene en el dataset)
    if "accepted_by_teacher" in df.columns:
        acc_rate = float(df["accepted_by_teacher"].fillna(0).astype(int).mean())
        metrics["teacher_accept_rate"] = acc_rate

    # --------- Guardar artefactos del job ---------
    job_id = args.job_id if args.job_id != "auto" else time.strftime("%Y%m%d_%H%M%S")
    job_dir = Path(args.out_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Modelo (vectorizer.json, rbm.pt, head.pt)
    strat.save(str(job_dir))

    # Métricas (para select_champion)
    with open(job_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Metadatos del job (útil para auditoría)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_ref": args.data,
        "job_id": job_id,
        "hparams": hparams,
        "feature_cols": feat_cols,
    }
    with open(job_dir / "job_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print({"job_dir": str(job_dir), **metrics})


if __name__ == "__main__":
    main()
