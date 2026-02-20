# backend/src/neurocampus/app/jobs/cmd_train_dbm_manual.py
import argparse
from pathlib import Path
import pandas as pd
from neurocampus.models.strategies.dbm_manual_strategy import DBMManualStrategy

def build_argparser():
    p = argparse.ArgumentParser(description="Train manual DBM via greedy RBM pretraining")
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out-dir", dest="out_dir", required=True)
    p.add_argument("--n-hidden1", type=int, default=64)
    p.add_argument("--n-hidden2", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--cd-k", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    config = {
        "n_hidden1": args.n_hidden1,
        "n_hidden2": args.n_hidden2,
        "lr": args.lr,
        "cd_k": args.cd_k,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    strategy = DBMManualStrategy(config=config)
    strategy.fit(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Guardar DBM (dbm_state.npz + meta.json)
    if strategy.model is None:
        raise RuntimeError("DBMManualStrategy: modelo no entrenado (strategy.model es None)")

    # Meta extra mínima (cols numéricas)
    num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    strategy.model.save(str(model_dir), extra_meta={"feat_cols_": num_cols, "task_type": "unsupervised"})


if __name__ == "__main__":
    main()
