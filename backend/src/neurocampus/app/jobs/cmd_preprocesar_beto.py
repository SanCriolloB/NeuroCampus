# backend/src/neurocampus/app/jobs/cmd_preprocesar_beto.py
import argparse, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd

from neurocampus.services.nlp.preprocess import limpiar_texto, tokenizar_y_lematizar_batch
from neurocampus.services.nlp.teacher import run_transformer, accept_mask

TEXT_CANDIDATES = [
    "Sugerencias", "sugerencias", "comentario", "comentarios",
    "observaciones", "obs", "texto", "review", "opinion"
]


def _pick_text_col(df: pd.DataFrame, prefer: str | None) -> str:
    """Intenta detectar la columna de texto de manera robusta (insensible a espacios/case)."""
    if prefer and prefer in df.columns:
        return prefer
    norm = {c: re.sub(r"\s+","",c).lower() for c in df.columns}
    for cand in TEXT_CANDIDATES:
        if cand in df.columns:
            return cand
        n_cand = re.sub(r"\s+","",cand).lower()
        for raw, n in norm.items():
            if n == n_cand:
                return raw
    raise ValueError("No se encontró columna de texto. Usa --text-col para forzar.")


def _cols_from_arg_or_auto(text_col_arg: str | None, df: pd.DataFrame) -> list[str] | None:
    """
    Devuelve lista de columnas si existen; si no hay ninguna:
      - devuelve None para indicar 'no hay texto' (caller decide fallback).
    """
    if text_col_arg is None or str(text_col_arg).strip().lower() == "auto":
        try:
            col = _pick_text_col(df, None)
            return [col]
        except Exception:
            return None
    cols = [c.strip() for c in str(text_col_arg).split(",") if c.strip()]
    existing = [c for c in cols if c in df.columns]
    return existing if existing else None


def _concat_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [df[c].astype(str).fillna("") for c in cols]
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" . ")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="Ruta CSV/Parquet de entrada.")
    ap.add_argument("--out", dest="dst", required=True, help="Ruta de salida (CSV/Parquet).")
    ap.add_argument("--text-col", default=None,
                    help="Columna(s) de texto: 'a,b,c' o 'auto' (por defecto autodetección).")
    ap.add_argument("--beto-model", default="finiteautomata/beto-sentiment-analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--neu-min", type=float, default=0.75)
    ap.add_argument("--beto-mode", choices=["probs", "simple"], default="probs",
                    help="probs: devuelve p_neg/p_neu/p_pos; simple: pipeline HF con top-1.")
    ap.add_argument("--min-tokens", type=int, default=1,
                    help="Tokens mínimos (lemmas) para considerar que hay texto.")
    ap.add_argument("--text-feats", choices=["none", "tfidf_lsa"], default="none",
                    help="Generación de embeddings clásicos para texto.")
    ap.add_argument("--text-feats-out-dir", default=None,
                    help="Directorio para guardar el featurizador (recomendado).")
    ap.add_argument("--keep-empty-text", action="store_true",
                    help="Si no hay texto, mantener filas etiquetándolas como neutrales.")
    ap.add_argument("--tfidf-min-df", type=float, default=None,
                    help="min_df para TF-IDF (entero >=1 o fracción 0-1).")
    ap.add_argument("--tfidf-max-df", type=float, default=None,
                    help="max_df para TF-IDF (fracción 0-1, o None).")
    args = ap.parse_args()

    # 1) Cargar datos
    df = pd.read_parquet(args.src) if args.src.lower().endswith(".parquet") else pd.read_csv(args.src)

    # 2) Selección de columnas de texto (tolerante)
    text_cols = _cols_from_arg_or_auto(args.text_col, df)
    if text_cols is None:
        if args.keep_empty_text:
            df["_texto_raw_concat"] = ""
            text_cols = []
        else:
            raise ValueError("No se encontró columna de texto y --keep-empty-text no está activo.")
    else:
        df["_texto_raw_concat"] = _concat_text_cols(df, text_cols)

    # 2.1) Limpiar + lematizar
    df["_texto_clean"]  = df["_texto_raw_concat"].astype(str).map(limpiar_texto)
    df["_texto_lemmas"] = tokenizar_y_lematizar_batch(df["_texto_clean"].tolist(), batch_size=512)

    # 2.2) Cobertura
    toklen = df["_texto_lemmas"].fillna("").str.split().map(len)
    df["has_text"] = (toklen >= args.min_tokens).astype(int)
    mask_has_text = df["has_text"] == 1
    mask_no_text  = ~mask_has_text

    # Inicialización de salidas
    df["p_neg"], df["p_neu"], df["p_pos"] = np.nan, np.nan, np.nan
    df["sentiment_label_teacher"] = pd.NA
    df["sentiment_conf"] = np.nan
    df["accepted_by_teacher"] = 0

    # 3) Etiquetado con BETO
    if args.beto_mode == "simple":
        if mask_has_text.any():
            try:
                from transformers import pipeline  # importación perezosa
            except Exception as e:
                raise RuntimeError(
                    "Falta 'transformers' para --beto-mode simple. Instálalo o usa --beto-mode probs."
                ) from e
            sentiment_analyzer = pipeline("sentiment-analysis", model=args.beto_model)
            df['Sugerencias_lemmatizadas'] = df['_texto_lemmas'].fillna('').astype(str)
            results = sentiment_analyzer(df.loc[mask_has_text, 'Sugerencias_lemmatizadas'].tolist())
            lbl_map = {"NEG":"neg","NEU":"neu","POS":"pos","negative":"neg","neutral":"neu","positive":"pos"}
            labels_simple = [lbl_map.get(r['label'], str(r['label']).lower()) for r in results]
            conf_simple   = [r.get("score", 1.0) for r in results]
            df.loc[mask_has_text, "sentiment_label_teacher"] = labels_simple
            df.loc[mask_has_text, "sentiment_conf"] = conf_simple
            df.loc[mask_has_text, "accepted_by_teacher"] = (
                df.loc[mask_has_text, "sentiment_conf"] >= args.threshold
            ).astype(int)
    else:
        if mask_has_text.any():
            P_text = run_transformer(
                df.loc[mask_has_text, "_texto_lemmas"].astype(str).tolist(),
                args.beto_model,
                batch_size=args.batch_size
            )
            df.loc[mask_has_text, "p_neg"] = P_text[:, 0]
            df.loc[mask_has_text, "p_neu"] = P_text[:, 1]
            df.loc[mask_has_text, "p_pos"] = P_text[:, 2]
            idx = P_text.argmax(axis=1)
            labels = np.array(["neg","neu","pos"], dtype=object)[idx]
            conf   = P_text.max(axis=1)
            df.loc[mask_has_text, "sentiment_label_teacher"] = labels
            df.loc[mask_has_text, "sentiment_conf"] = conf
            acc = accept_mask(P_text, labels, threshold=args.threshold, margin=args.margin, neu_min=args.neu_min)
            df.loc[mask_has_text, "accepted_by_teacher"] = acc.astype(int)

    # 3.1) Mantener filas sin texto si se pide
    if args.keep_empty_text:
        df.loc[mask_no_text, "p_neg"] = 0.0
        df.loc[mask_no_text, "p_neu"] = 1.0
        df.loc[mask_no_text, "p_pos"] = 0.0
        df.loc[mask_no_text, "sentiment_label_teacher"] = "neu"
        df.loc[mask_no_text, "sentiment_conf"] = 1.0
        df.loc[mask_no_text, "accepted_by_teacher"] = 1

    # 3.5) Feats de texto (TF-IDF + LSA) con normalización de min_df / max_df
    if args.text_feats != "none":
        if args.text_feats == "tfidf_lsa":
            from neurocampus.features.tfidf_lsa import TfidfLSAFeaturizer

            _default_min_df = 3
            _default_max_df = None

            # --- Normalización segura de min_df / max_df ---
            min_df = args.tfidf_min_df if args.tfidf_min_df is not None else _default_min_df
            max_df = args.tfidf_max_df if args.tfidf_max_df is not None else _default_max_df

            # si viene 1.0 como float, conviértelo a entero 1 (evita fracción 100%)
            if isinstance(min_df, float) and min_df >= 1.0:
                min_df = int(round(min_df))
            # si min_df <= 0, usa default seguro
            if isinstance(min_df, (int, float)) and min_df <= 0:
                min_df = _default_min_df

            # max_df: fracciones válidas (0<max_df<=1.0). Si >1.0 o <=0 => ignóralo
            if isinstance(max_df, float) and (max_df > 1.0 or max_df <= 0.0):
                max_df = None
            if isinstance(max_df, int) and max_df <= 1:
                max_df = None
            # --- fin normalización ---

            feat = TfidfLSAFeaturizer(
                n_components=64,
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=max_df
            )
            texts = df["_texto_lemmas"].astype(str).fillna("").tolist()
            Z = feat.fit_transform(texts)
            if args.keep_empty_text and Z is not None and len(Z) == len(df):
                import numpy as _np
                Z = _np.asarray(Z)
                Z[mask_no_text.values, :] = 0.0
            if Z is not None:
                for i in range(Z.shape[1]):
                    df[f"feat_t_{i+1}"] = Z[:, i]
            if args.text_feats_out_dir:
                Path(args.text_feats_out_dir).mkdir(parents=True, exist_ok=True)
                feat.save(args.text_feats_out_dir)

    # 4) Asegurar columna 'comentario' (trazabilidad humana)
    if "comentario" not in df.columns:
        df["comentario"] = df["_texto_clean"]

    # 5) Guardado + meta
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if args.dst.lower().endswith(".parquet"):
        df.to_parquet(args.dst, index=False)
    else:
        df.to_csv(args.dst, index=False)

    meta = {
        "model": args.beto_model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(len(df)),
        "accepted_count": int(df["accepted_by_teacher"].fillna(0).astype(int).sum()),
        "threshold": float(args.threshold),
        "margin": float(args.margin),
        "neu_min": float(args.neu_min),
        "text_col": ",".join(text_cols) if text_cols else "",
        "text_coverage": float(df["has_text"].mean()),
        "keep_empty_text": bool(args.keep_empty_text),
        "text_feats": args.text_feats,
        "tfidf_min_df": args.tfidf_min_df,
        "tfidf_max_df": args.tfidf_max_df,
        "text_feats_out_dir": args.text_feats_out_dir
    }
    with open(args.dst + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print({
        "out": args.dst,
        "n_rows": meta["n_rows"],
        "accept_rate": float(df["accepted_by_teacher"].fillna(0).astype(int).mean()),
        "text_coverage": meta["text_coverage"],
        "text_col": meta["text_col"],
        "keep_empty_text": meta["keep_empty_text"],
        "text_feats": meta["text_feats"],
        "tfidf_min_df": meta["tfidf_min_df"],
        "tfidf_max_df": meta["tfidf_max_df"]
    })


if __name__ == "__main__":
    main()
