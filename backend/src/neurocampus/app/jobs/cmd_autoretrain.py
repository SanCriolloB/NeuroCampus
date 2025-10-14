# backend/src/neurocampus/app/jobs/cmd_autoretrain.py
import argparse, json, os, sys, time, itertools
from pathlib import Path
import subprocess
import shutil
import tempfile
import textwrap

PY = sys.executable  # ruta a python.exe sin comillas raras

def run(args_list):
    """Ejecución segura multiplataforma (sin shell)."""
    print(">>", " ".join([str(a) for a in args_list]))
    subprocess.run(args_list, check=True)

def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def last_job_dir():
    jobs_root = Path("artifacts/jobs")
    if not jobs_root.exists():
        return None
    dirs = [d for d in jobs_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    # último por fecha de modificación
    dirs.sort(key=lambda d: d.stat().st_mtime)
    return str(dirs[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV nuevo (ej: examples/dataset_ejemplo.csv)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--family", default="with_text", choices=["with_text","numeric_only","distill"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=8, help="N° de combinaciones a probar")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    processed = f"data/processed/auto_{ts}.parquet"
    labeled   = f"data/labeled/auto_{ts}_beto.parquet"
    textonly  = f"data/labeled/auto_{ts}_beto_textonly.parquet"

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    Path("artifacts/jobs").mkdir(parents=True, exist_ok=True)
    Path("artifacts/champions").mkdir(parents=True, exist_ok=True)

    # 1) Estandarizar CSV -> parquet
    cmd = [
        PY, "-m", "neurocampus.app.jobs.cmd_cargar_dataset",
        "--in", args.csv,
        "--out", processed,
        "--meta-list", "codigo_materia,docente,grupo,periodo"
    ]
    if args.text_col:
        cmd += ["--text-col", args.text_col]
    run(cmd)

    # 2) Preprocesar + BETO
    run([
        PY, "-m", "neurocampus.app.jobs.cmd_preprocesar_beto",
        "--in", processed,
        "--out", labeled,
        "--beto-mode", "probs",
        "--threshold", "0.90",
        "--margin", "0.25",
        "--neu-min", "0.90",
        "--min-tokens", "1"
    ])

    # 3) Subset texto aceptado -> textonly (sin heredocs)
    code = textwrap.dedent("""
        import sys, pandas as pd
        inp, out = sys.argv[1], sys.argv[2]
        df = pd.read_parquet(inp)
        df = df[(df.get("has_text",0)==1) & (df.get("accepted_by_teacher",0)==1)].copy()
        df.to_parquet(out, index=False)
        print({"n": len(df)})
    """).strip()
    run([PY, "-c", code, labeled, textonly])

    # 4) Pequeña búsqueda de hiperparámetros con warm-start
    grid = dict(
        n_hidden=[64,128],
        cd_k=[1,2],
        epochs_rbm=[1,2],
        epochs=[80,100],
        lr_rbm=[5e-3],
        lr_head=[1e-2],
        scale_mode=["minmax"],
    )
    combos = list(itertools.product(*grid.values()))[:args.n_trials]
    data_path = textonly
    warm = "artifacts/champions/with_text/current"

    best = None
    for vals in combos:
        hp = dict(zip(grid.keys(), vals))
        train_cmd = [
            PY, "-m", "neurocampus.models.train_rbm",
            "--type", "general",
            "--data", data_path,
            "--job-id", "auto",
            "--seed", str(args.seed),
            "--epochs", str(hp["epochs"]),
            "--n-hidden", str(hp["n_hidden"]),
            "--cd-k", str(hp["cd_k"]),
            "--epochs-rbm", str(hp["epochs_rbm"]),
            "--batch-size", "128",
            "--lr-rbm", str(hp["lr_rbm"]),
            "--lr-head", str(hp["lr_head"]),
            "--scale-mode", hp["scale_mode"],
            "--warm-start-from", warm
        ]
        if args.family == "with_text":
            # NO añadir p_* como features (evita fuga)
            pass
        elif args.family == "numeric_only":
            pass
        elif args.family == "distill":
            train_cmd += ["--distill-soft"]

        run(train_cmd)

        job_dir = last_job_dir()
        if not job_dir:
            continue
        met_path = Path(job_dir) / "metrics.json"
        if not met_path.exists():
            continue
        m = read_json(str(met_path))
        score = (m.get("f1_macro") or 0.0, m.get("accuracy") or 0.0)
        if (best is None) or (score > best["score"]):
            best = dict(score=score, job_dir=job_dir, metrics=m)

    if not best:
        print({"promoted": False, "reason": "no candidate ran"})
        return

    # 5) Comparar con campeón y promover si mejora macro-F1
    champ = Path("artifacts/champions/with_text/current")
    champ_metrics = champ / "metrics.json"
    old_f1 = 0.0
    if champ_metrics.exists():
        old = read_json(str(champ_metrics))
        old_f1 = (old.get("f1_macro") or 0.0)

    new_f1 = best["metrics"].get("f1_macro") or 0.0
    print({"old_f1": old_f1, "new_f1": new_f1, "candidate": best["job_dir"]})

    if new_f1 > old_f1:
        ts2 = time.strftime("%Y%m%d_%H%M%S")
        archive = Path("artifacts/champions/with_text/archive") / ts2
        archive.parent.mkdir(parents=True, exist_ok=True)
        if champ.exists():
            # archiva campeón previo
            if archive.exists():
                shutil.rmtree(archive)
            shutil.copytree(champ, archive)
            shutil.rmtree(champ)
        champ.mkdir(parents=True, exist_ok=True)
        # Copia contenido del job ganador al current
        for item in Path(best["job_dir"]).iterdir():
            dest = champ / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        print({"promoted": True, "to": str(champ)})
    else:
        print({"promoted": False})

if __name__ == "__main__":
    main()
