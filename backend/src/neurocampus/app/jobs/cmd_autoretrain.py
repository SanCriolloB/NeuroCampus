# backend/src/neurocampus/app/jobs/cmd_autoretrain.py
import argparse, json, os, subprocess, sys, time, shlex, itertools
from pathlib import Path

PY = shlex.quote(sys.executable)

def run(cmd):
    print(">>", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def read_json(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def last_job_dir():
    jobs = [d for d in os.listdir("artifacts/jobs") if os.path.isdir(os.path.join("artifacts/jobs",d))]
    return os.path.join("artifacts","jobs", sorted(jobs)[-1]) if jobs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV nuevo (ej: examples/dataset_ejemplo.csv)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--family", default="with_text", choices=["with_text","numeric_only","distill"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=8, help="N° de combinaciones a probar (grid pequeño)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    processed = f"data/processed/auto_{ts}.parquet"
    labeled   = f"data/labeled/auto_{ts}_beto.parquet"
    textonly  = f"data/labeled/auto_{ts}_beto_textonly.parquet"

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    Path("artifacts/jobs").mkdir(parents=True, exist_ok=True)
    Path("artifacts/champions").mkdir(parents=True, exist_ok=True)

    # 1) Estandarizar
    cmd = f'{PY} -m neurocampus.app.jobs.cmd_cargar_dataset --in {shlex.quote(args.csv)} --out {processed} --meta-list "codigo_materia,docente,grupo,periodo"'
    if args.text_col: cmd += f" --text-col {shlex.quote(args.text_col)}"
    run(cmd)

    # 2) Preprocesar + BETO (ajusta si usas otros umbrales)
    run(f'{PY} -m neurocampus.app.jobs.cmd_preprocesar_beto --in {processed} --out {labeled} '
        f'--beto-mode probs --threshold 0.90 --margin 0.25 --neu-min 0.90 --min-tokens 1')

    # 3) Subset texto aceptado
    py = f"""
import pandas as pd
df = pd.read_parquet("{labeled}")
df = df[(df.get("has_text",0)==1) & (df.get("accepted_by_teacher",0)==1)].copy()
df.to_parquet("{textonly}", index=False)
print({{"n": len(df)}})
"""
    run(f'{PY} - <<"PY"\n{py}\nPY')

    # 4) Grid pequeño de HP (con warm-start)
    #    - Evitamos fuga: en "with_text" NO usamos --use-text-probs aquí (si quieres texto usa "distill")
    #    - "distill": usa --distill-soft (targets p_*), sin p_* en X.
    grid = dict(
        n_hidden=[64,128],
        cd_k=[1,2],
        epochs_rbm=[1,2],
        epochs=[80,100],
        lr_rbm=[5e-3],
        lr_head=[1e-2],
        scale_mode=["minmax"]
    )
    combos = list(itertools.product(*grid.values()))
    combos = combos[:args.n_trials]  # recortar
    data_path = textonly

    warm = "artifacts/champions/with_text/current"
    best = None

    for vals in combos:
        hp = dict(zip(grid.keys(), vals))
        job_id = "auto"
        base = (f'{PY} -m neurocampus.models.train_rbm --type general '
                f'--data {data_path} --job-id {job_id} --seed {args.seed} '
                f'--epochs {hp["epochs"]} --n-hidden {hp["n_hidden"]} '
                f'--cd-k {hp["cd_k"]} --epochs-rbm {hp["epochs_rbm"]} '
                f'--batch-size 128 --lr-rbm {hp["lr_rbm"]} --lr-head {hp["lr_head"]} '
                f'--scale-mode {hp["scale_mode"]} --warm-start-from {warm}')
        # modo de familia
        if args.family == "with_text":
            # sin p_* para evitar fuga; (si quieres texto real, integrar TF-IDF en otra iteración)
            pass
        elif args.family == "numeric_only":
            pass
        elif args.family == "distill":
            base += " --distill-soft"

        run(base)
        job_dir = last_job_dir()
        m = read_json(os.path.join(job_dir,"metrics.json"))
        score = (m.get("f1_macro") or 0.0, m.get("accuracy") or 0.0)
        if (best is None) or (score > best["score"]):
            best = dict(score=score, job_dir=job_dir, metrics=m)

    # 5) Comparar con campeón y promover si mejora macro-F1
    champ = "artifacts/champions/with_text/current"
    old = None
    if os.path.exists(os.path.join(champ,"metrics.json")):
        old = read_json(os.path.join(champ,"metrics.json"))
    old_f1 = (old or {}).get("f1_macro") or 0.0
    new_f1 = best["metrics"].get("f1_macro") or 0.0

    print({"old_f1": old_f1, "new_f1": new_f1, "candidate": best["job_dir"]})
    if new_f1 > old_f1:
        ts2 = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("artifacts/champions/with_text/archive", exist_ok=True)
        if os.path.isdir(champ):
            subprocess.run(f'cp -r {champ} artifacts/champions/with_text/archive/{ts2}', shell=True)
        subprocess.run(f'rm -rf {champ} && mkdir -p {champ}', shell=True)
        subprocess.run(f'cp -r {best["job_dir"]}/* {champ}/', shell=True)
        print({"promoted": True, "to": champ})
    else:
        print({"promoted": False})

if __name__ == "__main__":
    main()
