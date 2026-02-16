# backend/src/neurocampus/data/score_total.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


SCORE_META_KEYS = (
    "score_delta_max",
    "score_calib_q",
    "score_beta",
    "score_beta_source",
    "score_calib_abs_q",
    "score_version",
)


def load_sidecar_score_meta(labeled_path: str | Path, *, base_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Lee el sidecar meta generado por BETO:
      data/labeled/<ds>_beto.parquet.meta.json

    Retorna dict (si existe y es JSON válido) o None.
    """
    p = Path(labeled_path)
    if base_dir and not p.is_absolute():
        p = (base_dir / p).resolve()

    meta_path = Path(str(p) + ".meta.json")
    if not meta_path.exists():
        return None

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _detect_calif_cols(df: pd.DataFrame) -> List[str]:
    """Detecta columnas calif_* (calif_1..calif_5).

    Retorna una lista ordenada por sufijo numérico para estabilidad reproducible.
    """
    cols: List[str] = []
    for c in df.columns:
        s = str(c)
        if s.startswith("calif_"):
            suf = s.split("_", 1)[-1]
            if suf.isdigit():
                cols.append(s)

    cols.sort(key=lambda x: int(str(x).split("_", 1)[-1]))
    return cols


def _scale_to_0_50(values: pd.Series) -> pd.Series:
    """
    Heurística backward compatible:
    - Si parece 0..5 → *10
    - Si parece 0..10 → *5
    - Si ya parece 0..50 → 그대로

    Se usa quantil 95% para evitar outliers.
    """
    x = pd.to_numeric(values, errors="coerce").astype(float)
    arr = x.to_numpy(dtype=float)
    try:
        q95 = float(np.nanquantile(arr, 0.95)) if arr.size else 0.0
    except Exception:
        q95 = 0.0
    if not np.isfinite(q95):
        q95 = 0.0

    if q95 <= 5.25:   # Likert 0..5
        x = x * 10.0
    elif q95 <= 10.5:  # Escala 0..10
        x = x * 5.0

    return x.clip(0.0, 50.0)


def compute_score_base_0_50(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Garantiza un score base 0..50.

    Preferencia:
    1) score_base_0_50
    2) rating / score_0_50 / calificacion / score
    3) mean(calif_*) con re-escalado robusto

    Retorna (serie, debug)
    """
    debug: Dict[str, Any] = {"base_source": None, "calif_cols_detected": []}

    if "score_base_0_50" in df.columns:
        debug["base_source"] = "explicit_score_base_0_50"
        return _scale_to_0_50(df["score_base_0_50"]), debug

    for cand in ("rating", "score_0_50", "calificacion", "score"):
        if cand in df.columns:
            debug["base_source"] = f"explicit_{cand}"
            return _scale_to_0_50(df[cand]), debug

    calif_cols = _detect_calif_cols(df)
    debug["calif_cols_detected"] = calif_cols

    if calif_cols:
        base_raw = df[calif_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        debug["base_source"] = "derived_from_calif_mean"
        return _scale_to_0_50(base_raw), debug

    # Último fallback defensivo: cero
    debug["base_source"] = "fallback_zero"
    return pd.Series(0.0, index=df.index).astype(float), debug


def compute_score_total_0_50(
    df: pd.DataFrame,
    *,
    delta_max: float = 8.0,
    calib_q: float = 0.95,
    beta_fixed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calcula (o recalcula) score_total_0_50.

    - Usa score_base_0_50 (si no existe lo crea)
    - Si no hay p_pos/p_neg → score_total = base y beta=0.0 (no rompe)
    - Si hay has_text==0 → señal=0 para NO_TEXT (no sesga)

    Retorna meta dict (para sidecar / auditoría).
    """
    meta: Dict[str, Any] = {"score_version": "v3"}

    base, base_dbg = compute_score_base_0_50(df)
    df["score_base_0_50"] = base
    meta.update(base_dbg)

    has_sent = ("p_pos" in df.columns) and ("p_neg" in df.columns)
    if not has_sent:
        # No hay señal: total = base
        df["score_total_0_50"] = df["score_base_0_50"].astype(float).clip(0.0, 50.0)
        df["sentiment_delta"] = 0.0
        df["sentiment_signal"] = 0.0
        df["sentiment_delta_points"] = 0.0

        meta.update(
            {
                "score_delta_max": float(delta_max),
                "score_calib_q": float(calib_q),
                "score_beta": 0.0,
                "score_beta_source": "no_sentiment",
                "score_calib_abs_q": 0.0,
            }
        )
        return meta

    p_pos = pd.to_numeric(df["p_pos"], errors="coerce").fillna(0.0).astype(float)
    p_neg = pd.to_numeric(df["p_neg"], errors="coerce").fillna(0.0).astype(float)

    # Robustez: si falta sentiment_conf, df.get(...) retornaría un float y rompería .fillna().
    # Creamos una Serie default para mantener contrato vectorizado.
    if "sentiment_conf" in df.columns:
        conf_raw = df["sentiment_conf"]
    else:
        conf_raw = pd.Series(1.0, index=df.index)

    conf = pd.to_numeric(conf_raw, errors="coerce").fillna(1.0).astype(float).clip(0.0, 1.0)


    sent_delta = (p_pos - p_neg).astype(float)
    sent_signal = (sent_delta * conf).astype(float)

    # Política NO_TEXT: si has_text existe y es 0 ⇒ señal = 0
    if "has_text" in df.columns:
        mask_no_text = df["has_text"].fillna(0).astype(int) == 0
        sent_signal = sent_signal.where(~mask_no_text, 0.0)

    df["sentiment_delta"] = sent_delta
    df["sentiment_signal"] = sent_signal

    # Calibración beta (y qv para auditoría)
    # calibrar sobre filas con texto (si existe has_text)
    if "has_text" in df.columns:
        mask_text = df["has_text"].fillna(0).astype(int) == 1
        vals = sent_signal[mask_text].to_numpy(dtype=float)
    else:
        vals = sent_signal.to_numpy(dtype=float)

    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        qv = 0.0
    else:
        qv = float(np.quantile(np.abs(vals), float(calib_q)))

    if beta_fixed is not None:
        beta = float(beta_fixed)
        beta_source = "fixed"
    else:
        beta = 0.0 if qv <= 1e-9 else float(float(delta_max) / qv)
        beta_source = f"q{calib_q}"

    delta_points = (beta * sent_signal).clip(-float(delta_max), float(delta_max))
    df["sentiment_delta_points"] = delta_points

    df["score_total_0_50"] = (df["score_base_0_50"].astype(float) + delta_points).clip(0.0, 50.0)

    meta.update(
        {
            "score_delta_max": float(delta_max),
            "score_calib_q": float(calib_q),
            "score_beta": float(beta),
            "score_beta_source": str(beta_source),
            "score_calib_abs_q": (float(qv) if qv is not None else None),
        }
    )
    return meta


def ensure_score_columns(
    df: pd.DataFrame,
    *,
    labeled_meta: Optional[Dict[str, Any]] = None,
    prefer_total: bool = True,
    allow_derive: bool = True,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Backward compat para feature-pack:

    - Garantiza score_base_0_50 y (si prefer_total) score_total_0_50.
    - Si no existe beta/meta, fallback beta=0.0 (no rompe).
    - Si ya existe score_total_0_50 y prefer_total=True, no recalcula total; solo asegura base.

    Retorna: (df, score_col, score_debug)

    score_debug:
      - created_columns: columnas efectivamente creadas en esta llamada
      - source: "explicit_score_total" | "derived" | "missing"
      - beta_used/delta_max_used/calif_cols_detected: solo si se derivó
    """
    before_cols = set(df.columns)
    score_debug: Dict[str, Any] = {"created_columns": [], "source": None}

    # 1) Si ya existe score_total y preferimos total → úsalo, pero asegura base
    if "score_total_0_50" in df.columns and prefer_total:
        if "score_base_0_50" not in df.columns and allow_derive:
            base, dbg = compute_score_base_0_50(df)
            df["score_base_0_50"] = base
            if "score_base_0_50" not in before_cols:
                score_debug["created_columns"].append("score_base_0_50")
            score_debug.update(dbg)

        score_debug["source"] = "explicit_score_total"
        return df, "score_total_0_50", score_debug

    # 2) Preparar parámetros desde meta (si existe)
    beta_fixed: Optional[float] = None
    delta_max = 8.0
    calib_q = 0.95

    if isinstance(labeled_meta, dict):
        if labeled_meta.get("score_beta") is not None:
            try:
                beta_fixed = float(labeled_meta.get("score_beta"))
            except Exception:
                beta_fixed = None
        try:
            delta_max = float(labeled_meta.get("score_delta_max", delta_max))
        except Exception:
            pass
        try:
            calib_q = float(labeled_meta.get("score_calib_q", calib_q))
        except Exception:
            pass

    # 3) Derivar (si aplica)
    if allow_derive:
        meta = compute_score_total_0_50(df, delta_max=delta_max, calib_q=calib_q, beta_fixed=beta_fixed)

        after_cols = set(df.columns)
        desired_order = [
            "score_base_0_50",
            "score_total_0_50",
            "sentiment_delta",
            "sentiment_signal",
            "sentiment_delta_points",
        ]
        score_debug["created_columns"] = [c for c in desired_order if (c in after_cols and c not in before_cols)]
        score_debug["source"] = "derived"
        score_debug["beta_used"] = meta.get("score_beta")
        score_debug["delta_max_used"] = meta.get("score_delta_max")
        score_debug["calif_cols_detected"] = meta.get("calif_cols_detected", [])
    else:
        score_debug["source"] = "missing"

    # 4) Selección final
    if prefer_total and "score_total_0_50" in df.columns:
        return df, "score_total_0_50", score_debug
    if "score_base_0_50" in df.columns:
        return df, "score_base_0_50", score_debug
    if "rating" in df.columns:
        return df, "rating", score_debug

    # Último fallback (el caller debe validar existencia real antes de usar)
    return df, "score_base_0_50", score_debug
