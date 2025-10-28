# backend/src/neurocampus/models/train_rbm.py
# Entrenamiento de RBM (general/restringida) con soporte para:
# - Mezclar calificaciones + embeddings de texto (prefijo autodetectado)
# - Evitar fuga de etiquetas cuando el target proviene del teacher (p_* opcionales)
# - Guardar metadatos extendidos y artefactos de evaluación (incluye feat_cols/n_features)
# - Verbosidad controlable (--quiet) y chequeo de columnas post-fit

from __future__ import annotations
import argparse, json, os, time, random, re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch

# Estrategias
from neurocampus.models.strategies.modelo_rbm_general import ModeloRBMGeneral
from neurocampus.models.strategies.modelo_rbm_restringida import ModeloRBMRestringida


# -------------------------
# Utilidades
# -------------------------

# Prefijos candidatos para autodetección de embeddings de texto
CANDIDATE_PREFIXES = ["x_text_", "text_embed_", "text_", "feat_text_", "feat_t_"]

def pick_feature_cols(
    df: pd.DataFrame,
    *,
    include_text_embeds: bool,
    include_text_probs: bool,
    text_prefix: str = "x_text_",
    max_calif: int = 10,
) -> List[str]:
    """
    Construye la lista de columnas de entrada (feat_cols):
    - calif_1..N
    - p_neg/p_neu/p_pos (si include_text_probs=True)
    - embeddings de texto (prefijo autodetectado si no hay coincidencias con text_prefix)
    """
    cols = list(df.columns)
    feat: List[str] = []

    # 1) calif_1..N
    for i in range(max_calif):
        c = f"calif_{i+1}"
        if c in cols:
            feat.append(c)

    # 2) (opcional) p_neg/p_neu/p_pos
    if include_text_probs:
        for c in ("p_neg", "p_neu", "p_pos"):
            if c in cols:
                feat.append(c)

    # 3) (opcional) embeddings x_text_* (con autodetección de prefijo)
    if include_text_embeds:
        emb = [c for c in cols if c.startswith(text_prefix)]
        if not emb:
            # Autodetección para no depender del prefijo pasado por CLI
            for pr in CANDIDATE_PREFIXES:
                emb = [c for c in cols if c.startswith(pr)]
                if emb:
                    text_prefix = pr
                    break
        # ordenar por sufijo numérico (si existe)
        def _idx(c):
            m = re.search(r"(\d+)$", c)
            return int(m.group(1)) if m else 0
        emb = sorted(emb, key=_idx)
        feat.extend(emb)

    # deduplicar preservando orden
    seen, feat_dedup = set(), []
    for c in feat:
        if c not in seen:
            seen.add(c)
            feat_dedup.append(c)

    return feat_dedup


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_job_dir(out_dir: str, job_id: str | None) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (job_id is None) or (job_id == "auto"):
        job_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    job = root / job_id
    job.mkdir(parents=True, exist_ok=True)
    return job


def _pick_target_column(df: pd.DataFrame) -> Tuple[str, bool]:
    """
    Retorna (colname, has_teacher_hard)
    - has_teacher_hard=True cuando el target proviene del teacher de sentimiento
      y_sentimiento/sentiment_label_teacher.
    """
    for c in ["y", "label", "target"]:
        if c in df.columns:
            return c, False
    for c in ["y_sentimiento", "sentiment_label_teacher"]:
        if c in df.columns:
            return c, True
    raise ValueError(
        "No se encontró columna objetivo (y / label / target / y_sentimiento / sentiment_label_teacher)."
    )


def _to_numpy(df: pd.DataFrame, cols: List[str], mode: str) -> np.ndarray:
    X = df[cols].astype(np.float32).to_numpy()
    if mode == "minmax":
        mn = np.nanmin(X, axis=0)
        mx = np.nanmax(X, axis=0)
        X = (X - mn) / (mx - mn + 1e-9)
    elif mode == "scale_0_5":
        X = X / 5.0
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return X


def _encode_labels(y: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Devuelve y codificado a índices y el mapa inverso {idx: nombre_clase}.
    Si detecta exactamente las tres clases 'neg','neu','pos', fija ese orden.
    """
    cats = sorted(pd.Series(y).dropna().astype(str).str.lower().unique().tolist())
    order = ["neg", "neu", "pos"]
    if all(v in order for v in cats) and len(cats) == 3:
        cats = order
    mapping = {v: i for i, v in enumerate(cats)}
    inv = {i: v for v, i in mapping.items()}
    return pd.Series(y).astype(str).str.lower().map(mapping).to_numpy(), inv


def _instantiate_strategy(
    kind: str,
    n_features: int,
    n_hidden: int,
    cd_k: int,
    lr_rbm: float,
    lr_head: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device: str,
):
    if kind == "general":
        return ModeloRBMGeneral(
            n_visible=n_features,
            n_hidden=n_hidden,
            lr_rbm=lr_rbm,
            lr_head=lr_head,
            momentum=momentum,
            weight_decay=weight_decay,
            cd_k=cd_k,
            seed=seed,
            device=device,
        )
    elif kind == "restringida":
        return ModeloRBMRestringida(
            n_visible=n_features,
            n_hidden=n_hidden,
            lr_rbm=lr_rbm,
            lr_head=lr_head,
            momentum=momentum,
            weight_decay=weight_decay,
            cd_k=cd_k,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {kind}")


def _predict_proba_safe(strat, df_val: pd.DataFrame, X_val: Optional[np.ndarray]) -> np.ndarray:
    # Preferimos predict_proba_df si existe (usa columnas con nombres)
    if hasattr(strat, "predict_proba_df"):
        return strat.predict_proba_df(df_val)
    return strat.predict_proba(X_val)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["general", "restringida"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--job-id", default="auto")
    ap.add_argument("--out-dir", default="artifacts/jobs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-hidden", type=int, default=64)
    ap.add_argument("--cd-k", type=int, default=1)
    ap.add_argument("--epochs-rbm", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-rbm", type=float, default=5e-3)
    ap.add_argument("--lr-head", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scale-mode", choices=["minmax", "scale_0_5"], default="minmax")
    ap.add_argument(
        "--accept-teacher",
        action="store_true",
        help="Usa solo filas aceptadas por el teacher si existe 'accepted_by_teacher==1'.",
    )
    ap.add_argument("--accept-threshold", type=float, default=0.0)
    ap.add_argument("--use-cuda", action="store_true")

    # texto
    ap.add_argument(
        "--use-text-probs",
        action="store_true",
        help="Añade p_neg/p_neu/p_pos como features (bloqueado si target=teacher a menos que --distill-soft).",
    )
    ap.add_argument(
        "--use-text-embeds",
        action="store_true",
        help="Añade columnas de embeddings de texto con prefijo autodetectado.",
    )
    ap.add_argument(
        "--text-embed-prefix",
        default="x_text_",  # defecto cambiado para coincidir con tu dataset
        help="Prefijo preferido de columnas de embeddings (se autodetecta si no están).",
    )

    # distillation
    ap.add_argument(
        "--distill-soft",
        action="store_true",
        help="Si target=teacher, permite entrenar contra p_* como objetivos suaves (MSE/KL en la cabeza).",
    )

    # warm start (si tu estrategia lo implementa)
    ap.add_argument(
        "--warm-start-from",
        default=None,
        help="Ruta a un job_dir anterior para iniciar pesos/normalizadores. (Opcional)",
    )

    # otros
    ap.add_argument("--max-calif", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)

    args = ap.parse_args()

    _seed_everything(args.seed)
    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"

    # === Cargar dataset ===
    df = pd.read_parquet(args.data) if args.data.lower().endswith(".parquet") else pd.read_csv(args.data)

    # Asegurar columna objetivo
    ycol, has_teacher_hard = _pick_target_column(df)
    y_raw = df[ycol].astype(str).str.lower()

    # Guardarraíl anti-fuga: si target es del teacher, prohibimos p_* en features salvo distill
    if args.use_text_probs and has_teacher_hard and not args.distill_soft:
        raise ValueError(
            "Guardarraíl anti-fuga: --use-text-probs está activado y el target proviene del teacher "
            "(sentiment_label_teacher / y_sentimiento). Esto introduce fuga (el modelo 've' su objetivo). "
            "Opciones: 1) quita --use-text-probs; 2) usa --distill-soft para entrenar contra la distribución "
            "p_neg/p_neu/p_pos como target suave (y sin p_* en las features)."
        )

    # Filtrado por aceptación si se pide (nota: si tu dataset usa umbral sobre p_*, ese filtro se hace aguas arriba)
    if args.accept_teacher and ("accepted_by_teacher" in df.columns):
        df = df[df["accepted_by_teacher"].fillna(0).astype(int) >= int(args.accept_threshold)].copy()

    # Determinar features con autodetección de prefijo
    feat_cols = pick_feature_cols(
        df,
        include_text_embeds=args.use_text_embeds,
        include_text_probs=args.use_text_probs,
        text_prefix=args.text_embed_prefix,
        max_calif=args.max_calif,
    )
    if len(feat_cols) == 0:
        raise ValueError("No se encontraron columnas de características. Verifica calif_* / prefijos de embeddings.")

    # Codificar etiquetas
    y_enc, inv_map = _encode_labels(y_raw)

    # Split train/val estratificado
    idx_tr, idx_va = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=args.seed, stratify=y_enc
    )
    X_all = _to_numpy(df, feat_cols, mode=args.scale_mode)
    X_tr, X_va = X_all[idx_tr], X_all[idx_va]
    y_tr, y_va = y_enc[idx_tr], y_enc[idx_va]
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_va = df.iloc[idx_va].reset_index(drop=True)

    # Instanciar estrategia
    strat = _instantiate_strategy(
        kind=args.type,
        n_features=X_tr.shape[1],
        n_hidden=args.n_hidden,
        cd_k=args.cd_k,
        lr_rbm=args.lr_rbm,
        lr_head=args.lr_head,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    # Reportar a estrategia el orden exacto de columnas (para predict_proba_df)
    if hasattr(strat, "set_feature_columns"):
        strat.set_feature_columns(feat_cols)
    strat.feat_cols_ = list(feat_cols)

    # Entrenar usando DataFrame (mantiene nombres/orden)
    strat.fit(
        df_tr[feat_cols],
        y_tr,
        epochs=args.epochs,
        log_every=getattr(args, "log_every", 1),
        log_callback=(None if args.quiet else lambda e, d: print({"epoch": float(e), **{k: float(v) for k, v in d.items()}})),
    )

    # --- Mini-chequeo tras el fit ---
    try:
        feat_cols_set = len(getattr(strat, "feat_cols_", []) or [])
        first_cols = (getattr(strat, "feat_cols_", []) or [])[:12]
        if not args.quiet:
            print({"feat_cols_set": int(feat_cols_set), "first_cols": first_cols})
        if feat_cols_set == 0 and isinstance(df_va, pd.DataFrame):
            raise RuntimeError("feat_cols_ no está configurado pero se va a validar con DataFrame.")
    except Exception as e:
        if not args.quiet:
            print({"post_fit_sanity_check": "failed", "reason": str(e)})
    # --- fin mini-chequeo ---

    # Validación con DF (prefiere predict_proba_df)
    proba = _predict_proba_safe(strat, df_va[feat_cols], X_val=None)
    y_pred = proba.argmax(axis=1)

    # ====== Métricas: SOLO neg/neu/pos ======
    # Mapeamos qué índices corresponden a neg/neu/pos en inv_map
    valid_label_names = ["neg", "neu", "pos"]
    valid_indices = {i for i, name in inv_map.items() if name in valid_label_names}
    # Filtramos filas cuya etiqueta no sea una de las 3
    mask_valid = np.array([yi in valid_indices for yi in y_va])
    y_va_f = y_va[mask_valid]
    y_pred_f = y_pred[mask_valid]
    # Construimos target_names en el orden consistente neg, neu, pos (de existir)
    ordered_valid = [name for name in valid_label_names if name in inv_map.values()]
    # Si por alguna razón no hay suficientes, nos quedamos con lo que haya en valid_indices
    if not ordered_valid:
        ordered_valid = [inv_map[i] for i in sorted(valid_indices)]
    # Reporte
    report = classification_report(
        y_va_f,
        y_pred_f,
        target_names=ordered_valid,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_va_f, y_pred_f, labels=sorted(valid_indices)).tolist()
    f1_macro = float(report["macro avg"]["f1-score"])
    acc = float(report["accuracy"])

    # === Guardar todo ===
    job_dir = _resolve_job_dir(args.out_dir, args.job_id)

    # Artefactos de evaluación
    cm_path = job_dir / "confusion_matrix.json"
    rep_path = job_dir / "classification_report.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({"labels": ordered_valid, "matrix": cm}, f, ensure_ascii=False, indent=2)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Metadatos del job (útil para auditoría)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_ref": args.data,
        "job_id": job_dir.name,
        "hparams": {
            "type": args.type,
            "seed": args.seed,
            "epochs": args.epochs,
            "n_hidden": args.n_hidden,
            "cd_k": args.cd_k,
            "epochs_rbm": args.epochs_rbm,
            "batch_size": args.batch_size,
            "lr_rbm": args.lr_rbm,
            "lr_head": args.lr_head,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "scale_mode": args.scale_mode,
            "accept_teacher": args.accept_teacher,
            "accept_threshold": args.accept_threshold,
            "use_cuda": args.use_cuda,
            "use_text_probs": args.use_text_probs,
            "use_text_embeds": args.use_text_embeds,
            "text_embed_prefix": args.text_embed_prefix,
            "distill_soft": args.distill_soft,
            "max_calif": args.max_calif,
        },
        # Guardamos feat_cols bajo dos claves por compatibilidad
        "feat_cols": feat_cols,
        "feature_cols": feat_cols,
        "n_features": int(len(feat_cols)),
        "target_classes": ordered_valid,
        "y_source": ("teacher_hard" if has_teacher_hard else "manual_or_external"),
        "split": {
            "train_size": int(len(idx_tr)),
            "val_size": int(len(idx_va)),
            "stratified": True,
            "random_state": args.seed,
        },
        "text_features": {
            "used": bool(args.use_text_embeds),
            "prefix": (args.text_embed_prefix if args.use_text_embeds else None),
            "n_text_embed_cols": int(len([c for c in feat_cols if any(c.startswith(p) for p in CANDIDATE_PREFIXES)])) if args.use_text_embeds else 0,
        },
        "leak_guard": {
            "blocked_use_text_probs": bool(args.use_text_probs and has_teacher_hard and not args.distill_soft),
            "distill_soft": bool(args.distill_soft),
            "target_has_teacher_hard": bool(has_teacher_hard),
        },
        "eval_artifacts": {
            "confusion_matrix_json": str(cm_path),
            "classification_report_json": str(rep_path),
        },
        "metrics": {
            "f1_macro": f1_macro,
            "accuracy": acc,
        },
        "f1_macro": f1_macro,
        "accuracy": acc,
    }
    with open(job_dir / "job_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Guardar pesos/modelo si la estrategia lo permite
    if hasattr(strat, "save"):
        strat.save(job_dir)

    # Resumen final (conciso si --quiet)
    summary = {
        "job_dir": str(job_dir).replace("\\", "/"),
        "f1_macro": f1_macro,
        "accuracy": acc,
        "classes": ordered_valid,
        "n_val": int(len(y_va_f)),
        "n_labeled_used": int(len(idx_tr) + len(idx_va)),
        "n_features": int(len(feat_cols)),
        "type": args.type,
        "seed": args.seed,
    }
    if "accepted_by_teacher" in df.columns:
        summary["teacher_accept_rate"] = float(
            df["accepted_by_teacher"].fillna(0).astype(int).mean()
        )
    print(summary)


if __name__ == "__main__":
    main()
