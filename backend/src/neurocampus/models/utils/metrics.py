"""
neurocampus.models.utils.metrics
================================

Funciones de métricas ligeras (sin sklearn) para modelos de clasificación.

Motivación
----------
En NeuroCampus, los modelos RBM/RBMRestringida reportan métricas por época y al final
del entrenamiento para mostrarlas en la UI (curvas + confusion matrix). Para evitar
dependencias pesadas o inconsistencias entre estrategias, estas métricas se implementan
con NumPy.

Incluye
-------
- :func:`accuracy`
- :func:`confusion_matrix`
- :func:`f1_macro`
- :func:`normalize_probs`
- :func:`soft_to_hard`

.. note::
   Para el caso de NeuroCampus se usa comúnmente ``n_classes=3`` (neg/neu/pos),
   pero las funciones están generalizadas a N clases cuando aplica.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_probs(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normaliza probabilidades por fila para que sumen 1.

    Útil cuando las probabilidades p_neg/p_neu/p_pos provienen de procesos
    estocásticos o pueden venir con ruido.

    :param probs: Array (n_samples, n_classes).
    :param eps: Epsilon para evitar división por cero.
    :return: Array normalizado (n_samples, n_classes).
    """
    p = np.asarray(probs, dtype=np.float32)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    return (p / s).astype(np.float32, copy=False)


def soft_to_hard(probs: np.ndarray) -> np.ndarray:
    """
    Convierte probabilidades (soft labels) a clases hard usando argmax.

    :param probs: Array (n_samples, n_classes).
    :return: Array (n_samples,) con clase predicha por argmax.
    """
    p = np.asarray(probs)
    if p.ndim != 2:
        raise ValueError("soft_to_hard espera un array 2D (n_samples, n_classes).")
    return np.argmax(p, axis=1).astype(np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy = mean(y_true == y_pred).

    :param y_true: Array (n_samples,) con clases enteras.
    :param y_pred: Array (n_samples,) con clases enteras.
    :return: Accuracy como float.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.shape[0] == 0:
        return 0.0
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> List[List[int]]:
    """
    Confusion matrix NxN, serializable a JSON.

    ``cm[i][j]`` = número de ejemplos con clase verdadera i y predicha j.

    :param y_true: Array (n_samples,) con clases enteras en [0, n_classes-1].
    :param y_pred: Array (n_samples,) con clases enteras en [0, n_classes-1].
    :param n_classes: Número de clases.
    :return: Matriz como lista de listas (int) tamaño (n_classes, n_classes).
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)

    for t, p in zip(yt.tolist(), yp.tolist()):
        t = int(t)
        p = int(p)
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    return cm.tolist()


def f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    """
    F1 macro para clasificación multiclase sin sklearn.

    Para cada clase c:
      - precision_c = tp / (tp + fp)
      - recall_c = tp / (tp + fn)
      - f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)

    Luego:
      - f1_macro = mean_c(f1_c)

    :param y_true: Array (n_samples,) con clases enteras.
    :param y_pred: Array (n_samples,) con clases enteras.
    :param n_classes: Número de clases.
    :return: F1 macro como float.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    f1s: List[float] = []
    for c in range(int(n_classes)):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
        f1s.append(float(f1))

    return float(np.mean(f1s)) if f1s else 0.0

# -------------------------------------------------------------------
# Métricas ligeras para regresión (score_docente 0–50)
# -------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    if yt.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coeficiente de determinación)."""
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    if yt.size == 0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
