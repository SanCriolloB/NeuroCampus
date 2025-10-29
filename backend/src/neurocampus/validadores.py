# backend/src/neurocampus/validadores.py
# Adaptador del pipeline de validación (neurocampus.data.chain) con
# firma canónica unificada:
#   - run_validations(df, *, dataset_id, **kwargs)
# y compatibilidad retro:
#   - run(df, **kwargs) | validar(df, **kwargs) | validar_archivo(df, **kwargs)
#
# La salida esperada por /datos/validar es:
#   { ok: bool, sample: [...], message?: str, missing?: [...], extra?: [...], dataset_id?: str }

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Callable
import importlib
import inspect
import pandas as pd


# ------------ Utilidades de respuesta en el formato que espera /datos/validar ------------
def _sample(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            return df.head(5).to_dict(orient="records")
        except Exception:
            return []
    return []

def _resp(
    ok: bool,
    df: Optional[pd.DataFrame],
    *,
    message: str = "",
    missing: Optional[Iterable[str]] = None,
    extra: Optional[Iterable[str]] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": ok, "sample": _sample(df)}
    if message:
        out["message"] = message
    if missing:
        out["missing"] = list(missing)
    if extra:
        out["extra"] = list(extra)
    if dataset_id:
        out["dataset_id"] = dataset_id
    return out


# ------------------- Descubrimiento de una función de validación en chain -------------------
_CANDIDATE_NAMES = [
    "run_validations",
    "run",
    "validar",
    "validate",
    "validate_df",
    "validate_file",
    "validate_dataframe",
]

def _load_chain_validator() -> Optional[Callable[..., Any]]:
    """
    Intenta importar neurocampus.data.chain y encontrar una función de validación.
    No lanza excepción si no se encuentra; devuelve None.
    """
    try:
        chain_mod = importlib.import_module("neurocampus.data.chain")
    except Exception:
        return None

    for name in _CANDIDATE_NAMES:
        fn = getattr(chain_mod, name, None)
        if callable(fn):
            return fn

    # Alternativa: primera función pública que reciba ≥1 parámetro
    for _, obj in inspect.getmembers(chain_mod, inspect.isfunction):
        try:
            sig = inspect.signature(obj)
            if len(sig.parameters) >= 1:
                return obj
        except Exception:
            continue
    return None


# ------------------- Utilidades para compatibilidad de signaturas -------------------
def _filter_kwargs_for_callable(fn: Callable[..., Any], **kwargs) -> Dict[str, Any]:
    """
    Filtra kwargs para pasar solo aquellos aceptados por la firma de fn.
    Si fn acepta **kwargs, se devolverán todos.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # acepta **kwargs
    accepted = {name for name in params.keys()}
    return {k: v for k, v in kwargs.items() if k in accepted}


# ------------------- Adaptador de salida -------------------
def _coerce_output_to_expected(data: Any, df: Optional[pd.DataFrame], *, dataset_id: Optional[str]) -> Dict[str, Any]:
    """
    Convierte la salida arbitraria de chain.* a:
      { ok: bool, sample: [...], missing?: [...], extra?: [...], message?: str, dataset_id?: str }
    """
    # Caso ideal: ya devuelve el dict correcto
    if isinstance(data, dict) and "ok" in data and "sample" in data:
        out = dict(data)
        # Asegurar sample y dataset_id
        if not isinstance(out.get("sample"), list):
            out["sample"] = _sample(df)
        if dataset_id and "dataset_id" not in out:
            out["dataset_id"] = dataset_id
        return out

    # Si devuelve una tupla (ok, info)
    if isinstance(data, tuple) and data:
        ok = bool(data[0])
        details = data[1] if len(data) > 1 else None
        msg = ""
        missing = None
        extra = None
        if isinstance(details, dict):
            msg = details.get("message") or details.get("msg") or ""
            missing = details.get("missing")
            extra = details.get("extra")
        elif isinstance(details, str):
            msg = details
        return _resp(ok, df, message=msg, missing=missing, extra=extra, dataset_id=dataset_id)

    # Si solo devuelve bool
    if isinstance(data, bool):
        return _resp(bool(data), df, message="Validación ejecutada (bool).", dataset_id=dataset_id)

    # Cualquier otra cosa: lo metemos en message
    return _resp(True, df, message=f"Validación ejecutada. Resultado: {type(data).__name__}", dataset_id=dataset_id)


# ------------------- Firma canónica (Día 5) -------------------
def run_validations(df: pd.DataFrame, *, dataset_id: str = "default", **kwargs) -> Dict[str, Any]:
    """
    Punto de entrada CANÓNICO para el backend y para integraciones nuevas.

    Args:
        df: DataFrame a validar.
        dataset_id: Identificador lógico del dataset (p. ej., 'docentes').
        **kwargs: Parámetros adicionales que puedan requerir validadores específicos.

    Returns:
        Dict con llaves:
          - ok: bool
          - sample: list[dict] (primeras 5 filas)
          - message?: str
          - missing?: list[str]
          - extra?: list[str]
          - dataset_id?: str
    """
    if not isinstance(df, pd.DataFrame):
        return _resp(False, None, message="Entrada no es un DataFrame.", dataset_id=dataset_id)

    if df.empty:
        return _resp(False, df, message="Archivo vacío o sin filas.", dataset_id=dataset_id)

    fn = _load_chain_validator()
    if fn is None:
        # Fallback: validación mínima sin romper el endpoint
        return _resp(
            True,
            df,
            message="Validación básica OK (no se encontró función en neurocampus.data.chain).",
            dataset_id=dataset_id,
        )

    # Intentar pasar solo los argumentos soportados por el validador descubierto
    call_kwargs = _filter_kwargs_for_callable(fn, df=df, dataset_id=dataset_id, **kwargs)

    try:
        # Si el validador descubierto acepta 'df' como posicional, lo pasamos así;
        # si espera 'df' como keyword, también está en call_kwargs.
        if "df" in call_kwargs:
            _ = call_kwargs.pop("df")
            result = fn(df, **call_kwargs)
        else:
            # fallback: intentar solo df posicional
            result = fn(df, **call_kwargs)
    except TypeError:
        # Si la firma esperaba menos args/kwargs, intentar con solo df
        try:
            result = fn(df)
        except Exception as e:
            return _resp(False, df, message=f"Error ejecutando validador de chain: {e!s}", dataset_id=dataset_id)
    except Exception as e:
        return _resp(False, df, message=f"Error ejecutando validador de chain: {e!s}", dataset_id=dataset_id)

    try:
        return _coerce_output_to_expected(result, df, dataset_id=dataset_id)
    except Exception as e:
        return _resp(False, df, message=f"No se pudo adaptar la salida del validador: {e!s}", dataset_id=dataset_id)


# ------------------- Aliases de compatibilidad -------------------
def validar(df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    """
    Compatibilidad retro. Permite llamadas antiguas que no pasaban dataset_id.
    Se tomará de kwargs ('dataset_id') o se usará 'default'.
    """
    ds = kwargs.pop("dataset_id", "default")
    return run_validations(df, dataset_id=ds, **kwargs)

def run(df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    ds = kwargs.pop("dataset_id", "default")
    return run_validations(df, dataset_id=ds, **kwargs)

def validar_archivo(df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    ds = kwargs.pop("dataset_id", "default")
    return run_validations(df, dataset_id=ds, **kwargs)
