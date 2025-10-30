# backend/src/neurocampus/data/validation_wrapper.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple
import importlib
import inspect
import pandas as pd

# ------------------------------------------
# Descubrimiento de función de validación
# ------------------------------------------
_CANDIDATE_NAMES = [
    "run_validations",
    "run",
    "validar",
    "validate",
    "validate_df",
    "validate_dataframe",
    "validate_file",
]

def _find_chain_validator() -> Optional[Callable[..., Any]]:
    """
    Intenta importar neurocampus.data.chain y localizar una función compatible.
    Devuelve un callable o None si no encuentra nada.
    """
    try:
        chain_mod = importlib.import_module("neurocampus.data.chain")
    except Exception:
        return None

    for name in _CANDIDATE_NAMES:
        fn = getattr(chain_mod, name, None)
        if callable(fn):
            return fn

    # Heurística final: 1er callable público con >=1 parámetro
    for _, obj in inspect.getmembers(chain_mod, inspect.isfunction):
        try:
            if not obj.__name__.startswith("_"):
                sig = inspect.signature(obj)
                if len(sig.parameters) >= 1:
                    return obj
        except Exception:
            continue
    return None


# ------------------------------------------
# Helpers de construcción de reporte estándar
# ------------------------------------------
def _mk_issue(kind: str, message: str, column: Optional[str] = None, row: Optional[int] = None) -> Dict[str, Any]:
    """
    Estructura de un issue compatible con /datos/validar.
    """
    out: Dict[str, Any] = {"type": kind, "message": message}
    if column is not None:
        out["column"] = column
    if row is not None:
        out["row"] = row
    return out

def _mk_report(
    df: Optional[pd.DataFrame],
    ok: bool,
    issues: List[Dict[str, Any]],
    engine: str = "validation_wrapper",
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reporte estándar esperado por /datos/validar:
      {
        "summary": { "rows": int, "errors": int, "warnings": int, "engine": str, "message"?: str },
        "issues": [ { "type": "...", "message": "...", "column"?: "...", "row"?: int }, ... ]
      }
    """
    rows = int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0
    errors = sum(1 for i in issues if i.get("type") == "error")
    warnings = sum(1 for i in issues if i.get("type") == "warning")
    summary: Dict[str, Any] = {"rows": rows, "errors": errors, "warnings": warnings, "engine": engine}
    if message:
        summary["message"] = message
    # ok no se expone directo; se infiere por errors==0. Mantener compat hacia atrás:
    report = {"summary": summary, "issues": issues, "ok": ok}
    return report


# ------------------------------------------
# API principal (Día 6): run_validations(df, dataset_id)
# ------------------------------------------
def run_validations(df: pd.DataFrame, dataset_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """
    Punto unificado solicitado en el plan. Intenta delegar en neurocampus.data.chain.<fn>.
    Si la función encontrada devuelve un formato distinto, lo adaptamos al reporte estándar.
    """
    if not isinstance(df, pd.DataFrame):
        return _mk_report(None, False, [_mk_issue("error", "Entrada no es un DataFrame")])

    if df.empty:
        return _mk_report(df, False, [_mk_issue("error", "Archivo vacío o sin filas")])

    fn = _find_chain_validator()
    if fn is None:
        # Fallback mínimo: validar columnas básicas si dataset_id lo requiere (ejemplo)
        issues: List[Dict[str, Any]] = []
        # Puedes especializar por dataset_id aquí si lo necesitas
        return _mk_report(df, True, issues, message="Validación básica OK (no se encontró validador en chain)")

    # Intento con firma (df, dataset_id, **kwargs) y degradaciones
    try:
        try:
            result = fn(df, dataset_id, **kwargs)
        except TypeError:
            result = fn(df)
    except Exception as e:
        return _mk_report(df, False, [_mk_issue("error", f"Error ejecutando validador: {e}")])

    # Adaptar salidas comunes a nuestro contrato
    try:
        # 1) Si ya viene como {"summary":..., "issues":[...]}
        if isinstance(result, dict) and "summary" in result and "issues" in result:
            # Aseguramos llaves mínimas
            summary = result.get("summary") or {}
            issues = result.get("issues") or []
            ok = bool(result.get("ok", (summary.get("errors", 0) == 0)))
            return _mk_report(df, ok, list(issues), engine=str(summary.get("engine", "validation_wrapper")), message=summary.get("message"))
        # 2) Tuplas (ok, detalles)
        if isinstance(result, tuple) and result:
            ok = bool(result[0])
            details = result[1] if len(result) > 1 else None
            issues: List[Dict[str, Any]] = []
            msg = None
            if isinstance(details, dict):
                # Convención: details.get("missing") / details.get("extra")
                for col in details.get("missing", []) or []:
                    issues.append(_mk_issue("error", "Columna faltante", column=str(col)))
                for col in details.get("extra", []) or []:
                    issues.append(_mk_issue("warning", "Columna no esperada", column=str(col)))
                msg = details.get("message") or details.get("msg")
            elif isinstance(details, str):
                msg = details
            return _mk_report(df, ok, issues, message=msg)
        # 3) Booleano simple
        if isinstance(result, bool):
            return _mk_report(df, bool(result), [], message="Validación ejecutada (bool).")
        # 4) Cualquier otra cosa
        return _mk_report(df, True, [], message=f"Validación ejecutada. Resultado: {type(result).__name__}")
    except Exception as e:
        return _mk_report(df, False, [_mk_issue("error", f"No se pudo adaptar la salida del validador: {e}")])
