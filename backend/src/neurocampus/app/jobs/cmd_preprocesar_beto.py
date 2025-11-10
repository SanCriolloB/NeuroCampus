# backend/src/neurocampus/app/jobs/cmd_preprocesar_beto.py
# - Limpia + lematiza texto
# - Etiqueta sentimiento con BETO (dos modos):
#    * probs (recomendado): p_neg/p_neu/p_pos + gating (threshold, margin, neu_min)
#    * simple: pipeline tipo notebook (top1 + score) [import perezoso de transformers]
# - Deja 'comentario' estandarizado y columnas para el Student (RBM).
# - Trata el texto como opcional (has_text), ejecuta BETO sólo donde hay texto suficiente
#   y reporta cobertura de texto.
# - (Nuevo) Genera embeddings clásicos de texto (TF-IDF + LSA) como feat_t_1..feat_t_64 si se pide.
# - (Nuevo Nivel 2)
#   * --text-col admite múltiples columnas separadas por comas ("comentario,observaciones")
#   * --keep-empty-text: no descarta filas sin texto; etiqueta = neutral y embeddings = vector cero
#   * Overrides opcionales de TF-IDF: --tfidf-min-df / --tfidf-max-df

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
    """Detecta una sola columna de texto (compatibilidad hacia atrás)."""
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

def _cols_from_arg(text_col_arg: str | None, df: pd.DataFrame) -> list[str]:
    """
    Devuelve lista de columnas a usar. Soporta "a,b,c".
    Si es None, cae al detector de una sola columna.
    Filtra columnas inexistentes silenciosamente, pero si no queda ninguna, lanza error.
    """
    if text_col_arg is None:
        return [_pick_text_col(df, None)]
    cols = [c.strip() for c in str(text_col_arg).split(",") if c.strip()]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError(f"No se encontró ninguna de las columnas de texto indicadas: {cols}")
    return existing

def _concat_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Concatena columnas de texto existentes (ya crudas) con un separador.
    Convierte NaN a "" para evitar nulos.
    """
    parts = [df[c].astype(str).fillna("") for c in cols]
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" . ")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="CSV/Parquet estandarizado (calif_1..10 + texto).")
    ap.add_argument("--out", dest="dst", required=True, help="Salida etiquetada (parquet/csv).")
    ap.add_argument("--text-col", default=None,
                    help="Nombre(s) de columna de texto. Acepta múltiples separadas por coma p.ej. 'comentario,observaciones'.")
    ap.add_argument("--beto-model", default="finiteautomata/beto-sentiment-analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--neu-min", type=float, default=0.75)
    ap.add_argument("--beto-mode", choices=["probs", "simple"], default="probs",
                    help="probs: p_neg/p_neu/p_pos + gating; simple: pipeline tipo notebook (top1 + score)")
    ap.add_argument("--min-tokens", type=int, default=3,
                    help="Mínimo de tokens lematizados para considerar que hay texto. Default=3")

    # (Nuevo) Embeddings clásicos de texto
    ap.add_argument("--text-feats", choices=["none", "tfidf_lsa"], default="none",
                    help="Genera embeddings clásicos de texto (feat_t_*). Recomendado: tfidf_lsa")
    ap.add_argument("--text-feats-out-dir", default=None,
                    help="Directorio para guardar el featurizer (recomendado para reproducibilidad).")

    # (Nuevo Nivel 2) Comportamientos inclusivos y overrides TF-IDF
    ap.add_argument("--keep-empty-text", action="store_true",
                    help="Si se activa, no descarta filas con texto vacío: etiqueta neutral y embeddings cero.")
    ap.add_argument("--tfidf-min-df", type=float, default=None,
                    help="Override de min_df en TF-IDF. Si no se especifica, usa el default anterior del pipeline.")
    ap.add_argument("--tfidf-max-df", type=float, default=None,
                    help="Override de max_df en TF-IDF. Si no se especifica, usa el default anterior del pipeline.")

    args = ap.parse_args()

    # 1) Cargar datos
    df = pd.read_parquet(args.src) if args.src.lower().endswith(".parquet") else pd.read_csv(args.src)

    # 2) Seleccionar y preparar texto (multi-columna soportado)
    text_cols = _cols_from_arg(args.text_col, df)  # lista de columnas
    df["_texto_raw_concat"] = _concat_text_cols(df, text_cols)

    # 2.1) Limpiar + lematizar
    df["_texto_clean"]  = df["_texto_raw_concat"].astype(str).map(limpiar_texto)
    df["_texto_lemmas"] = tokenizar_y_lematizar_batch(df["_texto_clean"].tolist(), batch_size=512)

    # 2.2) Cobertura de texto: has_text si hay al menos min_tokens
    toklen = df["_texto_lemmas"].fillna("").str.split().map(len)
    df["has_text"] = (toklen >= args.min_tokens).astype(int)
    mask_has_text = df["has_text"] == 1
    mask_no_text  = ~mask_has_text

    # Inicializa columnas de salida comunes
    df["p_neg"], df["p_neu"], df["p_pos"] = np.nan, np.nan, np.nan
    df["sentiment_label_teacher"] = pd.NA
    df["sentiment_conf"] = np.nan
    df["accepted_by_teacher"] = 0

    # 3) BETO (dos modos) — sólo para filas con texto suficiente
    if args.beto_mode == "simple":
        # --- MODO SIMPLE (top1 + score) ---
        if mask_has_text.any():
            try:
                from transformers import pipeline  # import perezoso, sólo si se usa el modo simple
            except Exception as e:
                raise RuntimeError(
                    "Falta el paquete 'transformers' para usar --beto-mode simple. "
                    "Instala las dependencias (ver backend/requirements.txt) o usa --beto-mode probs."
                ) from e

            sentiment_analyzer = pipeline("sentiment-analysis", model=args.beto_model)
            df['Sugerencias_lemmatizadas'] = df['_texto_lemmas'].fillna('').astype(str)
            results = sentiment_analyzer(df.loc[mask_has_text, 'Sugerencias_lemmatizadas'].tolist())

            lbl_map = {"NEG": "neg", "NEU": "neu", "POS": "pos",
                       "negative": "neg", "neutral": "neu", "positive": "pos"}
            labels_simple = [lbl_map.get(r['label'], str(r['label']).lower()) for r in results]
            conf_simple   = [r.get("score", 1.0) for r in results]

            df.loc[mask_has_text, "sentiment_label_teacher"] = labels_simple
            df.loc[mask_has_text, "sentiment_conf"] = conf_simple
            df.loc[mask_has_text, "accepted_by_teacher"] = (df.loc[mask_has_text, "sentiment_conf"] >= args.threshold).astype(int)

    else:
        # --- MODO PROBS (recomendado) ---
        if mask_has_text.any():
            P_text = run_transformer(
                df.loc[mask_has_text, "_texto_lemmas"].astype(str).tolist(),
                args.beto_model,
                batch_size=args.batch_size
            )
            # asignar probs sólo a filas con texto
            df.loc[mask_has_text, "p_neg"] = P_text[:, 0]
            df.loc[mask_has_text, "p_neu"] = P_text[:, 1]
            df.loc[mask_has_text, "p_pos"] = P_text[:, 2]

            # top-1 + confianza
            idx = P_text.argmax(axis=1)
            labels = np.array(["neg", "neu", "pos"], dtype=object)[idx]
            conf   = P_text.max(axis=1)

            df.loc[mask_has_text, "sentiment_label_teacher"] = labels
            df.loc[mask_has_text, "sentiment_conf"] = conf

            # aceptación (gating)
            acc = accept_mask(P_text, labels, threshold=args.threshold, margin=args.margin, neu_min=args.neu_min)
            df.loc[mask_has_text, "accepted_by_teacher"] = acc.astype(int)

    # 3.1) (Nuevo) Mantener filas sin texto si se pide
    if args.keep_empty_text:
        df.loc[mask_no_text, "sentiment_label_teacher"] = "neu"
        df.loc[mask_no_text, "sentiment_conf"] = 1.0
        df.loc[mask_no_text, "accepted_by_teacher"] = 1

    # 3.5) (Nuevo) Feats de texto opcionales (con overrides TF-IDF)
    if args.text_feats != "none":
        if args.text_feats == "tfidf_lsa":
            from neurocampus.features.tfidf_lsa import TfidfLSAFeaturizer

            # Defaults anteriores del pipeline (compatibilidad)
            _default_min_df = 3
            _default_max_df = None

            min_df = args.tfidf_min_df if args.tfidf_min_df is not None else _default_min_df
            max_df = args.tfidf_max_df if args.tfidf_max_df is not None else _default_max_df

            feat_kwargs = dict(n_components=64, ngram_range=(1, 2), min_df=min_df)
            if max_df is not None:
                feat_kwargs["max_df"] = max_df

            feat = TfidfLSAFeaturizer(**feat_kwargs)

            texts = df["_texto_lemmas"].astype(str).fillna("").tolist()
            Z = feat.fit_transform(texts)  # shape: (N, 64)

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

    # 5) Guardar dataset + meta
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
        "text_col": ",".join(text_cols),
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
