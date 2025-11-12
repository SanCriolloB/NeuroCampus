# backend/src/neurocampus/app/jobs/cmd_preprocesar_batch.py
import argparse, os, sys, glob, subprocess
from pathlib import Path

def _detect_project_python() -> str:
    """
    Prioridad del intérprete de Python:
      1) Entorno virtual ACTIVADO (VIRTUAL_ENV/Scripts|bin/python)
      2) Entorno conda ACTIVO (CONDA_PREFIX/python)
      3) .venv en la RAÍZ del repo (./.venv/Scripts|bin/python)
      4) .venv en backend/ (./backend/.venv/Scripts|bin/python)
      5) sys.executable (intérprete actual)
    """
    env = os.environ
    venv = env.get("VIRTUAL_ENV")
    if venv:
        cand = Path(venv) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if cand.exists():
            return str(cand)
    conda = env.get("CONDA_PREFIX")
    if conda:
        cand = Path(conda) / ("python.exe" if os.name == "nt" else "bin/python")
        if cand.exists():
            return str(cand)
    repo_root = Path(__file__).resolve().parents[4]
    cand = repo_root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if cand.exists():
        return str(cand)
    backend_dir = repo_root / "backend"
    cand = backend_dir / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if cand.exists():
        return str(cand)
    return sys.executable

def _build_env_with_src() -> dict:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[3]  # .../backend/src
    pypp = str(src_dir)
    env["PYTHONPATH"] = pypp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", default="examples,examples/synthetic",
                    help="Directorios separados por coma para buscar *.csv")
    ap.add_argument("--out-dir", default="data/prep_auto",
                    help="Directorio de salida para los .parquet generados")
    ap.add_argument("--text-cols", default="auto",
                    help="Columnas de texto (coma-separadas) o 'auto' para autodetección")
    ap.add_argument("--beto-mode", choices=["probs","simple"], default="simple")
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--keep-empty-text", action="store_true", default=True)
    ap.add_argument("--tfidf-min-df", type=float, default=1.0)
    ap.add_argument("--tfidf-max-df", type=float, default=1.0)
    ap.add_argument("--text-feats", choices=["none","tfidf_lsa"], default="tfidf_lsa")
    ap.add_argument("--text-feats-out-dir", default=None)
    ap.add_argument("--beto-model", default="finiteautomata/beto-sentiment-analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--neu-min", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = _detect_project_python()
    env = _build_env_with_src()

    in_dirs = [d.strip() for d in args.in_dirs.split(",") if d.strip()]
    csvs = []
    for d in in_dirs:
        csvs.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))
    if not csvs:
        print("[batch] No se encontraron CSV en:", in_dirs)
        sys.exit(0)

    feats_out = args.text_feats_out_dir
    if feats_out is None and args.text_feats != "none":
        feats_out = str(out_dir / "textfeats")

    ok = True
    for f in csvs:
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = str(out_dir / f"{base}.parquet")

        cmd = [
            py, "-m", "neurocampus.app.jobs.cmd_preprocesar_beto",
            "--in", f,
            "--out", out_path,
            "--beto-mode", args.beto_mode,
            "--min-tokens", str(args.min_tokens),
            "--text-feats", args.text_feats,
            "--beto-model", args.beto_model,
            "--batch-size", str(args.batch_size),
            "--threshold", str(args.threshold),
            "--margin", str(args.margin),
            "--neu-min", str(args.neu_min),
            "--tfidf-min-df", str(args.tfidf_min_df),
            "--tfidf-max-df", str(args.tfidf_max_df),
        ]
        # Solo pasamos --text-col si el usuario NO dejó 'auto'
        if args.text_cols and args.text_cols.strip().lower() != "auto":
            cmd.extend(["--text-col", args.text_cols])

        if args.keep_empty_text:
            cmd.append("--keep-empty-text")
        if feats_out is not None:
            cmd.extend(["--text-feats-out-dir", feats_out])

        print("[batch] Procesando:", f, "→", out_path)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            ok = False
            print("[batch] ERROR procesando:", f)

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
