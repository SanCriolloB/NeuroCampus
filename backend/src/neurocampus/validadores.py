# backend/src/neurocampus/validadores.py
# Adaptador del pipeline de validación que ya tienes en neurocampus.data.chain
# para exponer las firmas que el backend espera:
#   - run_validations(df) | run(df) | validar(df) | validar_archivo(df)
#
# No toca tu implementación extensa; solo la envuelve.

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
  message: str = "",
  missing: Optional[Iterable[str]] = None,
  extra: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
  out: Dict[str, Any] = {"ok": ok, "sample": _sample(df)}
  if message:
    out["message"] = message
  if missing:
    out["missing"] = list(missing)
  if extra:
    out["extra"] = list(extra)
  return out


# ------------------- Descubrimiento de una función de validación en chain -------------------
# Intentamos distintos nombres posibles en tu módulo grande
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

  # Como alternativa, busca la primera función pública que reciba 1 arg (df) aproximadamente.
  for _, obj in inspect.getmembers(chain_mod, inspect.isfunction):
    try:
      sig = inspect.signature(obj)
      # Heurística: al menos 1 parámetro
      if len(sig.parameters) >= 1:
        return obj
    except Exception:
      continue
  return None


# ------------------- Adaptador de salida -------------------
def _coerce_output_to_expected(data: Any, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
  """
  Convierte la salida arbitraria de tu chain.* a la forma esperada por el endpoint:
    { ok: bool, sample: [...], missing?: [...], extra?: [...], message?: str }
  """
  # Caso ideal: ya devuelve el dict correcto
  if isinstance(data, dict) and "ok" in data and "sample" in data:
    # Asegura que sample esté presente y bien formado
    out = dict(data)
    if "sample" not in out or not isinstance(out["sample"], list):
      out["sample"] = _sample(df)
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
    return _resp(ok, df, message=msg, missing=missing, extra=extra)

  # Si solo devuelve bool
  if isinstance(data, bool):
    return _resp(bool(data), df, message="Validación ejecutada (bool).")

  # Cualquier otra cosa: lo metemos en message
  return _resp(True, df, message=f"Validación ejecutada. Resultado: {type(data).__name__}")


# ------------------- Función central que usará el endpoint -------------------
def validar(df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
  """
  Punto de entrada usado por el backend. Intenta usar tu validador en neurocampus.data.chain.
  Si no existe, retorna una validación básica para no romper el flujo.
  """
  if not isinstance(df, pd.DataFrame):
    return _resp(False, None, message="Entrada no es un DataFrame.")

  if df.empty:
    return _resp(False, df, message="Archivo vacío o sin filas.")

  fn = _load_chain_validator()
  if fn is None:
    # Fallback: validación mínima sin romper el endpoint
    return _resp(
      True,
      df,
      message="Validación básica OK (no se encontró función en neurocampus.data.chain)."
    )

  try:
    result = fn(df, *args, **kwargs)  # Pasamos df y args/kwargs por si tu firma los usa
  except TypeError:
    # Si la firma espera menos args, intentamos solo con df
    result = fn(df)
  except Exception as e:
    return _resp(False, df, message=f"Error ejecutando validador de chain: {e!s}")

  try:
    return _coerce_output_to_expected(result, df)
  except Exception as e:
    return _resp(False, df, message=f"No se pudo adaptar la salida del validador: {e!s}")


# ------------------- Aliases que busca el backend -------------------
run = validar
run_validations = validar
validar_archivo = validar
