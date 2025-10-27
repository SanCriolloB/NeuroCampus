"""
tools/cleanup.py — NeuroCampus
Día 1: Inventario y dry-run (no elimina archivos)

Uso:
  python -m tools.cleanup --inventory
  python -m tools.cleanup --dry-run --retention-days 90 --keep-last 3

Diseño:
- Escanea rutas conocidas (artifacts/, .tmp/, data/tmp/ y jobs/)
- Identifica candidatos a eliminación por:
  a) antigüedad (> retention_days)
  b) excedente respecto a keep_last por grupo/modelo
- Protege 'champions'
- Reporta totales y tamaño liberable, sin borrar (Día 1)

A partir del Día 2–4 se añadirá el modo 'real' de borrado.
"""

from __future__ import annotations
import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# --- Parámetros por defecto (pueden venir de .env) ---
DEFAULT_RETENTION_DAYS = int(os.getenv("NC_RETENTION_DAYS", "90"))
DEFAULT_KEEP_LAST = int(os.getenv("NC_KEEP_LAST", "3"))

# --- Rutas base (ajusta si el repo difiere) ---
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIRS = [
    BASE_DIR / "artifacts",
]
TMP_DIRS = [
    BASE_DIR / ".tmp",
    BASE_DIR / "data" / "tmp",
]
JOBS_DIR = BASE_DIR / "jobs"

CHAMPIONS_DIR = BASE_DIR / "artifacts" / "champions"  # protegidos

SECONDS_PER_DAY = 24 * 60 * 60


@dataclasses.dataclass
class FileInfo:
    path: Path
    size: int
    mtime: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.mtime) / SECONDS_PER_DAY


@dataclasses.dataclass
class InventoryReport:
    total_files: int
    total_size_bytes: int
    candidates_count: int
    candidates_size_bytes: int
    details: List[Tuple[str, int, float]]  # (str(path), size, age_days)


def human(nbytes: int) -> str:
    """Convierte bytes a una cadena legible (KB/MB/GB)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(nbytes)
    for u in units:
        if s < 1024.0:
            return f"{s:.2f} {u}"
        s /= 1024.0
    return f"{s:.2f} PB"


def iter_files(dirs: List[Path]) -> List[FileInfo]:
    files: List[FileInfo] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append(FileInfo(path=p, size=st.st_size, mtime=st.st_mtime))
                except FileNotFoundError:
                    # Archivos que desaparecen durante el recorrido
                    continue
    return files


def is_under(path: Path, parent: Path) -> bool:
    """Verifica que path esté bajo parent (seguridad)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def is_champion(path: Path) -> bool:
    """Marca archivos dentro de champions como protegidos."""
    return CHAMPIONS_DIR.exists() and is_under(path, CHAMPIONS_DIR)


def group_key(file: FileInfo) -> str:
    """
    Agrupa por modelo/experimento aproximado.
    Heurística:
      artifacts/<model>/<run_id>/...
    Devuelve 'model/run_id' o carpeta inmediata superior si no hay patrón claro.
    """
    parts = file.path.parts
    if "artifacts" in parts:
        i = parts.index("artifacts")
        # intenta tomar las dos carpetas siguientes como grupo
        group = parts[i + 1:i + 3]
        if group:
            return "/".join(group)
    # fallback: carpeta padre
    return str(file.path.parent)


def select_candidates(files: List[FileInfo], retention_days: int, keep_last: int) -> List[FileInfo]:
    """
    Selecciona candidatos por:
      1) Antigüedad > retention_days (excepto champions)
      2) Excedente de 'keep_last' por grupo (ordenado por mtime desc)
    """
    # 1) Por antigüedad
    oldies = [f for f in files if f.age_days > retention_days and not is_champion(f.path)]

    # 2) Por excedente de grupo
    by_group: Dict[str, List[FileInfo]] = {}
    for f in files:
        if is_champion(f.path):
            continue
        by_group.setdefault(group_key(f), []).append(f)

    surplus: List[FileInfo] = []
    for _, lst in by_group.items():
        lst_sorted = sorted(lst, key=lambda x: x.mtime, reverse=True)
        if len(lst_sorted) > keep_last:
            surplus.extend(lst_sorted[keep_last:])

    # Unir y de-duplicar
    uniq = {f.path: f for f in oldies + surplus}
    return list(uniq.values())


def inventory(retention_days: int, keep_last: int) -> InventoryReport:
    files = iter_files(ARTIFACTS_DIRS + TMP_DIRS + [JOBS_DIR])
    total_size = sum(f.size for f in files)

    candidates = select_candidates(files, retention_days, keep_last)
    cand_size = sum(f.size for f in candidates)
    details = [(str(f.path), f.size, f.age_days) for f in sorted(candidates, key=lambda x: x.mtime)]

    return InventoryReport(
        total_files=len(files),
        total_size_bytes=total_size,
        candidates_count=len(candidates),
        candidates_size_bytes=cand_size,
        details=details,
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="NeuroCampus cleanup tool (Día 1: inventario y dry-run)")
    parser.add_argument("--inventory", action="store_true", help="Sólo mostrar inventario resumido.")
    parser.add_argument("--dry-run", action="store_true", help="Simular eliminación sin borrar nada.")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS, help="Días de retención.")
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST, help="Cuántos artefactos recientes conservar por grupo.")
    args = parser.parse_args(argv)

    rep = inventory(args.retention_days, args.keep_last)

    print("== NeuroCampus :: Limpieza (Día 1) ==")
    print(f"Total archivos: {rep.total_files}")
    print(f"Total tamaño:   {human(rep.total_size_bytes)}")
    print(f"Candidatos:     {rep.candidates_count}")
    print(f"Tamaño elegible para liberar: {human(rep.candidates_size_bytes)}")
    print("")
    print("Top 50 candidatos (ruta, tamaño, edad_días):")
    for path, size, age in rep.details[:50]:
        print(f"  - {path} | {human(size)} | {age:.1f}d")

    if args.inventory:
        print("\nModo: INVENTORY (no se elimina nada).")
        return 0

    if args.dry_run:
        print("\nModo: DRY-RUN (no se elimina nada, sólo simulación).")
        # En Días 2–4 se implementará la eliminación real.
        # Aquí se podría imprimir el comando hipotético de borrado seguro:
        # print("Simular rm ...")
        return 0

    # Protección extra (día 1): impedir ejecución accidental de borrado real.
    print("\nBorrado real deshabilitado en Día 1. Usa --dry-run.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
