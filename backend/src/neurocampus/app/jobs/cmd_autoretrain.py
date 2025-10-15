# backend/src/neurocampus/app/jobs/cmd_autotrain.py
"""
Orquestador de auto-entrenamiento:
- Explora un espacio de hiperparámetros (RBM general/restringida, con/sin texto)
- Ejecuta train_rbm en modo silencioso (--quiet)
- Registra un leaderboard CSV
- Promueve el mejor job a artifacts/champions/<family>/
- Imprime solo el mejor resultado al final (F1 macro y Accuracy)

Uso típico:
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_autotrain \
  --data data/labeled/dataset_ejemplo_beto_text.parquet \
  --family with_text \
  --n-trials 16 \
  --seed 42 \
  --quiet
"""
from __future__ import annotations
import argparse, json, os, sys, time, random, shutil, subprocess
from pathlib import Path
import pandas as pd

ART_DIR = Path("artifacts")
JOBS_DIR = ART_DIR / "jobs"
CHAMPIONS_DIR = ART_DIR / "champions"
REPORTS_DIR = ART_DIR / "reports"

def _nowstamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _latest_job_meta() -> Path | None:
    metas = sorted(JOBS_DIR.glob("*/job_meta.json"), key=lambda p: p.stat().st_mtime)
    return metas[-1] if metas else None

def _has_text_embeds(data_path: str) -> bool:
    df = pd.read_parquet(data_path) if data_path.lower().endswith(".parquet") else pd.read_csv(data_path)
    cols = list(df.columns)
    # dos estilos soportados: 'x_text_*' (estandar actual) o 'feat_t_*' (TFIDF/LSA)
    return any(c.startswith("x_text_") for c in cols) or any(c.startswith("feat_t_") for c in cols)

def _build_space(with_text: bool):
    """Devuelve una lista de configuraciones candidatas (dicts)."""
    types = ["general", "restringida"]
    n_hidden = [64, 96, 128]
    cd_k = [1, 2]
    epochs_rbm = [1, 2]
    lr_rbm = [5e-3, 1e-3]
    lr_head = [1e-2, 5e-3]
    scale_mode = ["minmax", "scale_0_5"]
    use_text_embeds = [True, False] if with_text else [False]

    combos = []
    for t in types:
        for nh in n_hidden:
            for ck in cd_k:
                for er in epochs_rbm:
                    for lrr in lr_rbm:
                        for lrh in lr_head:
                            for sc in scale_mode:
                                for ute in use_text_embeds:
                                    combos.append(dict(
                                        type=t, n_hidden=nh, cd_k=ck, epochs_rbm=er,
                                        lr_rbm=lrr, lr_head=lrh, scale_mode=sc,
                                        use_text_embeds=ute
                                    ))
    return combos

def _run_train(args, trial_cfg: dict, seed: int, job_epochs: int, batch_size: int, data_path: str, quiet: bool):
    """Ejecuta train_rbm como subproceso. Devuelve (returncode, stdout, stderr)."""
    cmd = [
        sys.executable, "-m", "neurocampus.models.train_rbm",
        "--data", data_path,
        "--type", trial_cfg["type"],
        "--job-id", "auto",
        "--seed", str(seed),
        "--epochs", str(job_epochs),
        "--n-hidden", str(trial_cfg["n_hidden"]),
        "--cd-k", str(trial_cfg["cd_k"]),
        "--epochs-rbm", str(trial_cfg["epochs_rbm"]),
        "--batch-size", str(batch_size),
        "--lr-rbm", str(trial_cfg["lr_rbm"]),
        "--lr-head", str(trial_cfg["lr_head"]),
        "--scale-mode", trial_cfg["scale_mode"],
    ]
    if trial_cfg["use_text_embeds"]:
        cmd.append("--use-text-embeds")
    if args.accept_teacher:
        cmd.append("--accept-teacher")
        cmd.extend(["--accept-threshold", str(args.accept_threshold)])
    if args.warm_start_from:
        cmd.extend(["--warm-start-from", args.warm_start_from])
    if args.max_calif is not None:
        cmd.extend(["--max-calif", str(args.max_calif)])
    if args.use_cuda:
        cmd.append("--use-cuda")
    if quiet:
        cmd.append("--quiet")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def _read_last_job_meta():
    meta_path = _latest_job_meta()
    if meta_path is None:
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, meta_path.parent

def _promote_champion(best_meta: dict, best_dir: Path, family: str):
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    fam_dir = CHAMPIONS_DIR / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    dest = fam_dir / best_dir.name
    # limpia destino si ya existe
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(best_dir, dest)

    # marca latest
    (fam_dir / "latest.txt").write_text(str(dest).replace("\\","/"), encoding="utf-8")
    # guarda resumen
    summary = {
        "promoted_at": _nowstamp(),
        "job_dir": str(dest).replace("\\","/"),
        "f1_macro": float(best_meta.get("f1_macro", 0.0)),
        "accuracy": float(best_meta.get("accuracy", 0.0)),
        "classes": best_meta.get("classes", []),
        "hparams": best_meta.get("hparams", {}),
        "feature_cols": best_meta.get("feature_cols") or best_meta.get("feature_columns"),
        "data_path": best_meta.get("data_path"),
        "family": family,
    }
    (fam_dir / "best_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta parquet/csv ya preprocesado (con 'comentario' y labels del teacher).")
    ap.add_argument("--family", required=True, help="Nombre lógico de la familia (p.ej., with_text, numeric_only)")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--quiet", action="store_true")
    # opciones avanzadas que se pasan a train_rbm si se desean
    ap.add_argument("--accept-teacher", action="store_true")
    ap.add_argument("--accept-threshold", type=float, default=0.80)
    ap.add_argument("--warm-start-from", default=None)
    ap.add_argument("--max-calif", type=int, default=None)
    ap.add_argument("--use-cuda", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with_text = _has_text_embeds(args.data)
    space = _build_space(with_text=with_text)

    # muestreamos sin reemplazo
    if args.n_trials < len(space):
        space = random.sample(space, args.n_trials)

    leaderboard = []
    print(f">> Ejecutando {len(space)} trials (with_text={with_text}) sobre {args.data} ...")
    for i, cfg in enumerate(space, start=1):
        print(f"[{i}/{len(space)}] trial: {cfg}")
        rc, out, err = _run_train(args, cfg, seed=args.seed, job_epochs=args.epochs,
                                  batch_size=args.batch_size, data_path=args.data, quiet=args.quiet)
        if rc != 0:
            print(f"   !! fallo train_rbm (rc={rc})\nSTDERR:\n{err[:4000]}")
            leaderboard.append({
                "ok": 0, "trial_idx": i, "cfg": cfg, "f1_macro": float("nan"),
                "accuracy": float("nan"), "job_dir": None
            })
            continue

        meta, job_dir = _read_last_job_meta()
        if not meta:
            print("   !! no se encontró job_meta.json tras el entrenamiento")
            leaderboard.append({
                "ok": 0, "trial_idx": i, "cfg": cfg, "f1_macro": float("nan"),
                "accuracy": float("nan"), "job_dir": None
            })
            continue

        f1 = float(meta.get("f1_macro", 0.0))
        acc = float(meta.get("accuracy", 0.0))
        leaderboard.append({
            "ok": 1, "trial_idx": i, "cfg": cfg, "f1_macro": f1, "accuracy": acc,
            "job_dir": str(job_dir).replace("\\","/")
        })
        print(f"   -> f1_macro={f1:.4f} accuracy={acc:.4f} dir={job_dir.name}")

    # guardar leaderboard
    ts = _nowstamp()
    rows = []
    for row in leaderboard:
        flat = dict(
            ok=row["ok"], trial_idx=row["trial_idx"], job_dir=row["job_dir"],
            f1_macro=row["f1_macro"], accuracy=row["accuracy"],
            type=row["cfg"].get("type") if row["cfg"] else None,
            n_hidden=row["cfg"].get("n_hidden") if row["cfg"] else None,
            cd_k=row["cfg"].get("cd_k") if row["cfg"] else None,
            epochs_rbm=row["cfg"].get("epochs_rbm") if row["cfg"] else None,
            lr_rbm=row["cfg"].get("lr_rbm") if row["cfg"] else None,
            lr_head=row["cfg"].get("lr_head") if row["cfg"] else None,
            scale_mode=row["cfg"].get("scale_mode") if row["cfg"] else None,
            use_text_embeds=row["cfg"].get("use_text_embeds") if row["cfg"] else None,
        )
        rows.append(flat)
    lb_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lb_path = REPORTS_DIR / f"leaderboard_{args.family}_{ts}.csv"
    lb_df.to_csv(lb_path, index=False, encoding="utf-8")
    print(f"\n>> Leaderboard guardado en: {lb_path}")

    # elegir mejor (f1_macro, y desempate por accuracy)
    ok_df = lb_df[lb_df["ok"] == 1].copy()
    if ok_df.empty:
        print(">> No hubo entrenamientos válidos. Revisa errores arriba.")
        sys.exit(1)
    ok_df["_rank"] = ok_df[["f1_macro","accuracy"]].apply(tuple, axis=1)
    best_row = ok_df.sort_values(by=["f1_macro","accuracy"], ascending=[False, False]).iloc[0]
    best_meta, best_dir = _read_last_job_meta()  # ojo: _read_last_job_meta() da el último; mejor cargar desde best_row
    # Carga meta del directorio ganador concreto:
    if best_row["job_dir"]:
        best_meta_path = Path(str(best_row["job_dir"])) / "job_meta.json"
        if best_meta_path.exists():
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            best_dir = best_meta_path.parent

    promoted = _promote_champion(best_meta, best_dir, family=args.family)
    print("\n================= RESULTADO =================")
    print(f"Familia: {args.family}")
    print(f"Champion: {promoted}")
    print(f"Mejor F1 macro: {best_row['f1_macro']:.4f} | Accuracy: {best_row['accuracy']:.4f}")
    print(f"Config: type={best_row['type']}, n_hidden={best_row['n_hidden']}, cd_k={best_row['cd_k']}, "
          f"epochs_rbm={best_row['epochs_rbm']}, lr_rbm={best_row['lr_rbm']}, lr_head={best_row['lr_head']}, "
          f"scale_mode={best_row['scale_mode']}, use_text_embeds={bool(best_row['use_text_embeds'])}")
    print("============================================")

if __name__ == "__main__":
    main()
