# backend/src/neurocampus/app/jobs/validate_prep_dir.py
import argparse
from pathlib import Path
import sys
import json
import pandas as pd

REQ_DEFAULT = [
    "accepted_by_teacher",
    "sentiment_label_teacher",
    "sentiment_conf",
    "has_text",
]

def has_any_prefix(cols, prefixes):
    return any(any(c.startswith(p) for p in prefixes) for c in cols)

def validate_file(fp: Path, require_cols, any_prefixes):
    df = pd.read_parquet(fp) if fp.suffix.lower()==".parquet" else pd.read_csv(fp)
    cols = list(df.columns)
    ok = True
    missing = [c for c in require_cols if c not in cols]
    if missing:
        ok = False
    has_text_feats = True
    if any_prefixes:
        has_text_feats = has_any_prefix(cols, any_prefixes)
        if not has_text_feats:
            ok = False
    info = {
        "file": str(fp),
        "rows": len(df),
        "missing_cols": missing,
        "has_text_feats": has_text_feats,
    }
    return ok, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directorio con .parquet/.csv a validar")
    ap.add_argument("--must-exist-cols", default=",".join(REQ_DEFAULT),
                    help="Columnas obligatorias separadas por coma")
    ap.add_argument("--require-any-prefix", default="feat_t_",
                    help="Prefijos de columnas (cualquiera) separados por coma. Vacío para no exigir.")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"[validate] ERROR: no existe el directorio {d}", file=sys.stderr)
        sys.exit(2)

    require_cols = [c.strip() for c in args.must_exist_cols.split(",") if c.strip()]
    prefixes = [p.strip() for p in args.require_any_prefix.split(",") if p.strip()]

    files = sorted([*d.glob("*.parquet"), *d.glob("*.csv")])
    if not files:
        print(f"[validate] ERROR: no hay .parquet/.csv en {d}", file=sys.stderr)
        sys.exit(2)

    all_ok = True
    report = []
    for fp in files:
        ok, info = validate_file(fp, require_cols, prefixes)
        report.append(info)
        status = "OK" if ok else "FAIL"
        print(f"[validate] {status} {fp.name} -> rows={info['rows']} "
              f"missing={info['missing_cols']} text_feats={info['has_text_feats']}")
        if not ok:
            all_ok = False

    print("[validate] resumen json:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
