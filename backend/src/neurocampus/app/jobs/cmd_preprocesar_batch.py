# backend/src/neurocampus/app/jobs/cmd_preprocesar_batch.py
import argparse, os, sys, glob, subprocess
from pathlib import Path
from shutil import which

def _detect_project_python() -> str:
    """
    Devuelve la ruta del intérprete de Python del proyecto con esta prioridad:
      1) VIRTUAL_ENV (venv activado)
      2) CONDA_PREFIX (entorno conda activo)
      3) ./.venv (venv local del repo) [Windows/Linux]
      4) sys.executable (intérprete actual)

    Garantiza una ruta ejecutable existente.
    """
    env = os.environ

    # 1) venv activo
    venv = env.get("VIRTUAL_ENV")
    if venv:
        cand = Path(venv) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if cand.exists():
            return str(cand)

    # 2) conda activo
    conda = env.get("CONDA_PREFIX")
    if conda:
        cand = Path(conda) / ("python.exe" if os.name == "nt" else "bin/python")
        if cand.exists():
            return str(cand)

    # 3) venv local en el repo
    repo_root = Path(__file__).resolve().parents[4]  # .../NeuroCampus (raíz)
    local_venv = repo_root / ".venv"
    if local_venv.exists():
        cand = local_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if cand.exists():
            return str(cand)

    # 4) intérprete actual
    return sys.executable

def _build_env_with_src() -> dict:
    """
    Construye un entorno de ejecución que añade backend/src al PYTHONPATH
    y preserva el resto de variables.
    """
    env = os.environ.copy()
    # Este archivo está en .../backend/src/neurocampus/app/jobs/cmd_preprocesar_batch.py
    src_dir = Path(__file__).resolve().parents[3]  # .../backend/src
    pypp = str(src_dir)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pypp + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pypp
    # Evita mezclar site-packages del usuario con el del entorno del proyecto
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", default="examples,examples/synthetic",
                    help="Directorios separados por coma para buscar *.csv")
    ap.add_argument("--out-dir", default="data/prep_auto",
                    help="Directorio de salida para los .parquet generados")
    ap.add_argument("--text-cols", default="comentario,observaciones",
                    help="Columnas de texto (coma-separadas) que se concatenarán")
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

    # Elegir el Python correcto del proyecto
    py = _detect_project_python()

    # Entorno con PYTHONPATH a backend/src
    env = _build_env_with_src()

    # Recolectar CSVs
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
            "--text-col", args.text_cols,
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
