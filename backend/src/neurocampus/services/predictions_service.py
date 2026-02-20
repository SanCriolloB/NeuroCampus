"""neurocampus.services.predictions_service
======================================

Service layer para **Predicciones** (fase P2).

Este módulo encapsula la lógica de:

- Cargar un modelo listo para inferencia a partir de un ``LoadedPredictorBundle``
  (obtenido por ``neurocampus.predictions.loader``).
- Ejecutar predicción **desde feature_pack** (row/pair) y normalizar la salida
  para el contrato HTTP.

Diseño
------
- Importa modelos pesados (p.ej. torch) de forma *lazy* para no afectar tests
  que solo validan resolve/validate (P2.2).
- Soporta dos formatos de modelo:

  1) Directorio ``<run_dir>/model`` (persistencia real del entrenamiento)
     - Ej: ``rbm.pt``, ``head.pt``, ``meta.json``.

  2) ``model.bin`` serializado con ``pickle`` (útil para tests y compat).

Errores
-------
Este módulo no conoce HTTP. Expone excepciones específicas para que el router
mapee a 404/422 de forma consistente.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle

import numpy as np
import pandas as pd

from neurocampus.predictions.loader import LoadedPredictorBundle, PredictorNotReadyError
from neurocampus.data.features_prepare import load_feature_pack
from neurocampus.utils.paths import artifacts_dir, safe_segment, abs_artifact_path


class InferenceNotAvailableError(RuntimeError):
    """El bundle está resuelto pero no se pudo cargar un modelo usable."""


@dataclass(frozen=True)
class FeaturePackSlice:
    """Subconjunto del feature_pack a predecir."""

    df: pd.DataFrame
    meta: Optional[Dict[str, Any]]
    kind: str


def _infer_input_kind(*, input_level: str) -> str:
    """Mapea input_level -> kind esperado por :func:`load_feature_pack`."""

    lvl = str(input_level or "").strip().lower()
    if lvl in ("pair", "teacher_materia", "teacher-materia"):
        return "pair"
    return "train"  # default row-level


def _slice_df(
    df: pd.DataFrame,
    *,
    limit: int,
    offset: int,
    ids: Optional[List[int]],
) -> pd.DataFrame:
    """Aplica slicing defensivo sobre el DF.

    - Si `ids` viene, selecciona por índices posicionales (iloc).
    - En otro caso aplica offset/limit.
    """

    if ids:
        # Limpiar ids (solo ints >= 0)
        clean: List[int] = []
        for v in ids:
            try:
                i = int(v)
            except Exception:
                continue
            if i >= 0:
                clean.append(i)
        if not clean:
            return df.iloc[0:0].copy()
        return df.iloc[clean].copy()

    off = max(0, int(offset or 0))
    lim = max(1, int(limit or 1))
    return df.iloc[off : off + lim].copy()


def _load_pickle_model(model_bin_path: Path) -> Any:
    """Carga un modelo serializado en ``model.bin`` con pickle."""

    try:
        with open(model_bin_path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as e:
        raise InferenceNotAvailableError(f"No se pudo deserializar model.bin con pickle: {model_bin_path}") from e

    # Contrato mínimo
    if not (hasattr(obj, "predict_df") or hasattr(obj, "predict")):
        raise InferenceNotAvailableError(
            "Objeto deserializado no implementa predict_df(df) ni predict(X)."
        )
    return obj


def _load_model_from_model_dir(*, model_name: str, model_dir: Path) -> Any:
    """Carga el modelo desde ``<run_dir>/model``.

    Nota: los imports de estrategias son lazy para no forzar torch en tests.
    """

    name = str(model_name or "").lower()
    if "restring" in name:
        from neurocampus.models.strategies.modelo_rbm_restringida import RBMRestringida  # lazy import

        return RBMRestringida.load(str(model_dir))

    # Default: rbm_general
    from neurocampus.models.strategies.modelo_rbm_general import RBMGeneral  # lazy import

    return RBMGeneral.load(str(model_dir))


def load_inference_model(bundle: LoadedPredictorBundle) -> Any:
    """Carga un objeto de modelo listo para inferencia.

    Estrategia:
    1) Si existe ``<run_dir>/model`` => cargar con la estrategia correspondiente.
    2) Si no, intentar cargar ``model.bin`` con pickle (compat/tests).

    Raises:
        InferenceNotAvailableError: si no hay forma de cargar modelo.
    """

    model_name = str(bundle.predictor.get("model_name") or "")
    model_dir = (bundle.run_dir / "model").resolve()

    if model_dir.exists() and model_dir.is_dir():
        try:
            return _load_model_from_model_dir(model_name=model_name, model_dir=model_dir)
        except Exception as e:
            raise InferenceNotAvailableError(f"No se pudo cargar modelo desde dir: {model_dir}") from e

    return _load_pickle_model(bundle.model_bin_path)


def load_feature_pack_slice(
    *,
    dataset_id: str,
    input_level: str,
    limit: int,
    offset: int,
    ids: Optional[List[int]],
) -> FeaturePackSlice:
    """Carga y corta el feature_pack del dataset."""

    kind = _infer_input_kind(input_level=input_level)
    df, meta = load_feature_pack(dataset_id=dataset_id, kind=kind)
    sliced = _slice_df(df, limit=limit, offset=offset, ids=ids)
    return FeaturePackSlice(df=sliced, meta=meta, kind=kind)


def _get_label_names(model: Any, proba: np.ndarray) -> List[str]:
    """Intenta obtener el orden de labels para proba."""

    if hasattr(model, "labels"):
        try:
            labels = list(getattr(model, "labels"))
            if len(labels) == proba.shape[1]:
                return [str(x) for x in labels]
        except Exception:
            pass

    # Fallback clasificación 3-clases usada en el proyecto
    if proba.shape[1] == 3:
        return ["neg", "neu", "pos"]

    return [f"c{i}" for i in range(proba.shape[1])]

def _safe_int(df: pd.DataFrame, row_idx: int, col: str) -> Optional[int]:
    """Convierte de forma segura un valor de una columna a ``int``.

    Returns:
        int o None si el valor es nulo/no convertible.
    """
    try:
        v = df.iloc[int(row_idx)][col]  # type: ignore[index]
    except Exception:
        return None

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if v is None:
        return None

    try:
        return int(v)
    except Exception:
        try:
            return int(str(v).strip())
        except Exception:
            return None

def predict_dataframe(
    model: Any,
    df: pd.DataFrame,
    *,
    return_proba: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Ejecuta predicción sobre un DataFrame y normaliza a JSON.

    Soporta dos modos:

    - **Clasificación**: usa ``predict_df``/``predict`` y opcionalmente
      ``predict_proba_df`` para poblar ``proba``.
    - **Regresión (score 0–50)**: usa ``predict_score_df`` y retorna
      ``score_total_pred``.

    Raises:
        InferenceNotAvailableError: si se requiere regresión y el modelo no
        implementa ``predict_score_df``.
    """
    n = int(len(df))
    if n == 0:
        return [], {"prediction_fields": []}

    # Detectar tipo de tarea (contrato esperado: model.task_type).
    task_type = str(getattr(model, "task_type", "classification")).lower()
    is_regression = task_type == "regression"

    # Fallback: algunos modelos de score no declaran task_type, pero exponen
    # predict_score_df() y NO exponen predict_proba_df().
    if (
        not is_regression
        and hasattr(model, "predict_score_df")
        and not hasattr(model, "predict_proba_df")
    ):
        is_regression = True
        task_type = "regression"

    has_teacher = "teacher_id" in df.columns
    has_materia = "materia_id" in df.columns

    # -------------------------
    # REGRESIÓN: score 0–50
    # -------------------------
    if is_regression:
        if not hasattr(model, "predict_score_df"):
            raise InferenceNotAvailableError(
                "El modelo es de regresión pero no implementa predict_score_df()"
            )

        scores = np.asarray(model.predict_score_df(df), dtype=float).reshape(-1)
        scores = np.clip(scores, 0.0, 50.0)

        out: List[Dict[str, Any]] = []
        for i in range(n):
            rec: Dict[str, Any] = {
                "row_index": int(i),
                "score_total_pred": float(scores[i]),
            }
            if has_teacher:
                rec["teacher_id"] = _safe_int(df, i, "teacher_id")
            if has_materia:
                rec["materia_id"] = _safe_int(df, i, "materia_id")
            out.append(rec)

        schema: Dict[str, Any] = {
            "prediction_fields": sorted({k for r in out for k in r.keys()}),
            "task_type": task_type,
            "score_field": "score_total_pred",
            "proba_labels": None,
        }
        return out, schema

    # -------------------------
    # CLASIFICACIÓN: label + proba opcional
    # -------------------------
    proba = None
    if return_proba and hasattr(model, "predict_proba_df"):
        proba = np.asarray(model.predict_proba_df(df), dtype=float)

    if hasattr(model, "predict_df"):
        labels = list(model.predict_df(df))
    elif hasattr(model, "predict"):
        labels = list(model.predict(df))
    else:
        labels = [None] * n

    out: List[Dict[str, Any]] = []

    label_names: Optional[List[str]] = None
    if proba is not None and proba.ndim == 2 and proba.shape[0] == n:
        label_names = _get_label_names(model, proba)

    for i in range(n):
        rec: Dict[str, Any] = {"row_index": int(i), "label": str(labels[i])}

        if has_teacher:
            rec["teacher_id"] = _safe_int(df, i, "teacher_id")
        if has_materia:
            rec["materia_id"] = _safe_int(df, i, "materia_id")

        if label_names is not None and proba is not None:
            rec["proba"] = {
                label_names[j]: float(proba[i, j]) for j in range(len(label_names))
            }

        out.append(rec)

    schema: Dict[str, Any] = {
        "prediction_fields": sorted({k for r in out for k in r.keys()}),
        "task_type": "classification",
        "proba_labels": label_names,
    }
    return out, schema


def predict_from_feature_pack(
    *,
    bundle: LoadedPredictorBundle,
    dataset_id: str,
    input_level: str,
    limit: int,
    offset: int,
    ids: Optional[List[int]],
    return_proba: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Ejecuta predicción usando el feature_pack ya generado.

    Returns
    -------
    (predictions, schema, warnings)

    Raises
    ------
    FileNotFoundError
        Si no existe feature_pack.
    ValueError
        Si parámetros inválidos.
    InferenceNotAvailableError
        Si no se puede cargar el modelo.
    PredictorNotReadyError
        Si el modelo no está listo para inferencia.
    """

    # Validación ligera
    ds = str(dataset_id or "").strip()
    if not ds:
        raise ValueError("dataset_id vacío")

    model = load_inference_model(bundle)

    pack = load_feature_pack_slice(dataset_id=ds, input_level=input_level, limit=limit, offset=offset, ids=ids)

    predictions, schema = predict_dataframe(model, pack.df, return_proba=return_proba)

    warnings: List[str] = []
    if pack.kind == "pair":
        warnings.append("input_level=pair: las predicciones son por par docente-materia")

    return predictions, schema, warnings


def save_predictions_parquet(
    *,
    run_id: str,
    dataset_id: str,
    family: Optional[str],
    input_level: str,
    predictions: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Persiste predicciones en artifacts para consumo posterior.

    Layout (P2.4-C)
    -------------
    ``artifacts/predictions/<dataset_id>/<family>/<run_id>/<input_level>/predictions.parquet``

    Notes:
    - Este método NO asume HTTP. Retorna paths absolutos para que el router decida
      cómo exponerlos (p.ej. vía :func:`neurocampus.utils.paths.rel_artifact_path`).
    - Si `predictions` está vacío, se escribe un parquet vacío (válido para flujos batch).
    """

    ds = safe_segment(dataset_id)
    fam = safe_segment(family) if family else "no_family"
    rid = safe_segment(run_id)
    lvl = safe_segment(input_level or "row")

    out_dir = artifacts_dir() / "predictions" / ds / fam / rid / lvl
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "predictions.parquet"

    # Aplanamos nested dicts (p.ej. proba.{neg,neu,pos}) a columnas.
    df = pd.json_normalize(predictions, sep=".")
    df.to_parquet(parquet_path, index=False)

    paths: Dict[str, Path] = {"predictions": parquet_path}

    if schema is not None:
        schema_path = out_dir / "schema.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["schema"] = schema_path

    return paths

# ---------------------------------------------------------------------------
# P2.4-E: lectura de outputs persistidos (predictions.parquet)
# ---------------------------------------------------------------------------

def resolve_predictions_parquet_path(predictions_uri: str) -> Path:
    """Resuelve un `predictions_uri` (lógico) a un path absoluto y seguro.

    Reglas de seguridad:
    - Solo se permite leer dentro de `artifacts/predictions/`.
    - El path debe apuntar a un archivo llamado `predictions.parquet`.

    Raises
    ------
    ValueError
        Si `predictions_uri` es inválido o apunta fuera del sandbox permitido.
    FileNotFoundError
        Si el archivo no existe.
    """
    ref = str(predictions_uri or "").strip()
    if not ref:
        raise ValueError("predictions_uri vacío")

    p = abs_artifact_path(ref).expanduser().resolve()
    base = (artifacts_dir() / "predictions").expanduser().resolve()

    try:
        _ = p.relative_to(base)
    except Exception as e:
        raise ValueError("predictions_uri debe estar bajo artifacts/predictions") from e

    if p.name != "predictions.parquet" or p.suffix.lower() != ".parquet":
        raise ValueError("predictions_uri debe apuntar a predictions.parquet")

    if not p.exists():
        raise FileNotFoundError(f"No existe predictions.parquet: {ref}")

    return p


def load_predictions_preview(
    *,
    predictions_uri: str,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[Dict[str, Any]]]:
    """Carga una vista previa (preview) de predicciones persistidas.

    Notes
    -----
    - Implementación simple para P2: usa pandas para leer el parquet.
    - Para outputs grandes, se puede optimizar con lectura por row-groups (pyarrow).

    Returns
    -------
    (rows, columns, schema)
    """
    path = resolve_predictions_parquet_path(predictions_uri)
    df = pd.read_parquet(path)

    off = max(0, int(offset or 0))
    lim = max(1, int(limit or 1))
    sliced = df.iloc[off : off + lim].copy()

    rows: List[Dict[str, Any]] = sliced.to_dict(orient="records")
    columns: List[str] = [str(c) for c in list(sliced.columns)]

    schema_path = path.parent / "schema.json"
    schema: Optional[Dict[str, Any]] = None
    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            schema = None

    return rows, columns, schema
