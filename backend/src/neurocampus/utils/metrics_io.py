# backend/src/neurocampus/utils/metrics_io.py
# Utilidades para crear el directorio de run, guardar snapshot de config y métricas.

from __future__ import annotations
import json, os, time, shutil
from typing import Dict, Any
import yaml

def prepare_run_dir(root: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"rbm_audit_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config_snapshot(run_dir: str, cfg_path: str):
    dst = os.path.join(run_dir, "config.snapshot.yaml")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(cfg_path, dst)

def write_metrics(run_dir: str, metrics: Dict[str, Any]) -> str:
    out = os.path.join(run_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
