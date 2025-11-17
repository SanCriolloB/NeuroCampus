# backend/src/neurocampus/models/hparam_search.py
"""
Búsqueda de hiperparámetros (grid search) para RBM/BM usando el auditor k-fold
existente (run_kfold_audit) y seleccionando un modelo "champion".

Uso:
  PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.hparam_search \
      --config configs/rbm_search.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from neurocampus.models.audit_kfold import run_kfold_audit
from neurocampus.utils.metrics_io import (
    prepare_run_dir,
    save_config_snapshot,
    write_metrics,
    load_yaml,
)


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Recibe un diccionario {param: [val1, val2, ...]} y devuelve una lista
    de combinaciones [{param: valX, ...}, ...] para usar en grid search.
    """
    if not grid:
        return [{}]

    keys = sorted(grid.keys())
    values_list = [grid[k] for k in keys]
    combos: List[Dict[str, Any]] = []

    for values in itertools.product(*values_list):
        combos.append({k: v for k, v in zip(keys, values)})

    return combos


def _select_better(
    current_best: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
    metric_name: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Compara el champion actual con un nuevo candidato según metric_name y mode
    ('max' o 'min'). Devuelve el mejor.
    """
    if current_best is None:
        return candidate

    best_score = current_best["score"]
    cand_score = candidate["score"]

    if cand_score is None:
        return current_best

    if best_score is None:
        return candidate

    if mode == "min":
        return candidate if cand_score < best_score else current_best
    # por defecto 'max'
    return candidate if cand_score > best_score else current_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Ruta al YAML de configuración de búsqueda (e.g. configs/rbm_search.yaml)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Dataset y target
    df = pd.read_parquet(cfg["dataset"]["path"])
    target = cfg["dataset"].get("target")

    eval_cfg = cfg["evaluation"]
    search_cfg = cfg["search"]
    models_cfg = cfg["models"]
    globals_cfg = dict(cfg.get("globals", {}))

    metric_name: str = search_cfg.get("metric", "f1")
    mode: str = search_cfg.get("mode", "max")

    # Directorios de artefactos
    root = cfg["artifacts"]["root"]
    run_dir = prepare_run_dir(root, prefix="rbm_search")
    save_config_snapshot(run_dir, args.config)

    champion_dir = cfg["artifacts"].get("champion_dir", os.path.join(root, "champions"))
    os.makedirs(champion_dir, exist_ok=True)
    champion_path = os.path.join(champion_dir, "rbm_champion.json")

    all_runs: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    # Grid search por modelo
    for mm in models_cfg:
        model_name = mm["name"]
        grid = mm.get("param_grid", {})

        combos = _expand_grid(grid)
        if not combos:
            combos = [{}]

        print(f"[SEARCH] Modelo={model_name} con {len(combos)} combinaciones...")

        for combo in combos:
            # Mezclamos globals + combo específico
            params = dict(globals_cfg)
            params.update(combo)

            print(f"[SEARCH]  -> params={params}")

            res = run_kfold_audit(
                df=df,
                target=target,
                model_name=model_name,
                model_params=params,
                n_splits=eval_cfg["n_splits"],
                shuffle=eval_cfg["shuffle"],
                stratify=eval_cfg["stratify"],
                random_seed=eval_cfg["random_seed"],
                metrics=eval_cfg["metrics"],
            )

            # Métrica objetivo para selección del champion
            summary = res.get("summary", {})
            metric_info = summary.get(metric_name)
            score = None
            if isinstance(metric_info, dict):
                score = metric_info.get("mean")

            run_record = {
                "model": model_name,
                "params": params,
                "metrics": res,
                "score_metric": metric_name,
                "score": score,
            }
            all_runs.append(run_record)

            cand = {
                "model": model_name,
                "params": params,
                "metric": metric_name,
                "score": score,
                "summary": summary,
            }
            best = _select_better(best, cand, metric_name=metric_name, mode=mode)

    # Estructura de resultados completa
    results: Dict[str, Any] = {
        "dataset": cfg["dataset"],
        "evaluation": eval_cfg,
        "search": search_cfg,
        "runs": all_runs,
        "champion": best,
    }

    out = write_metrics(run_dir, results)
    print(f"[SEARCH] Resultados de búsqueda escritos en: {out}")
    if best is not None:
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        print(f"[SEARCH] Modelo champion guardado en: {champion_path}")
    else:
        print("[SEARCH] No se pudo determinar un modelo champion (sin métrica válida).")


if __name__ == "__main__":
    main()
