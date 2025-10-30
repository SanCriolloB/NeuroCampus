"""
tools/cleanup.py — NeuroCampus
Día 2: Borrado real seguro (mover a papelera), logs y exclusiones

Modos:
  - Inventario:      python -m tools.cleanup --inventory
  - Dry-run:         python -m tools.cleanup --dry-run --retention-days 90 --keep-last 3
  - Borrado real:    python -m tools.cleanup --force --retention-days 90 --keep-last 3

Seguridad:
  - Champions protegidos (artifacts/champions/**)
  - Exclusiones por glob (--exclude-globs)
  - Borrado reversible: mueve a .trash/<YYYYMMDD>/<ruta_relativa>
  - Logs CSV en logs/cleanup.log con: ts,action,path,size,age_days,reason

Siguientes días (3-4): endpoint FastAPI & UI de administración.
"""

from __future__ import annotations
import argparse
import csv
import dataclasses
import fnmatch
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# --- Defaults y entorno ---
DEFAULT_RETENTION_DAYS = int(os.getenv("NC_RETENTION_DAYS", "90"))
DEFAULT_KEEP_LAST = int(os.getenv("NC_KEEP_LAST", "3"))
DEFAULT_EXCLUDE_GLOBS = os.getenv("NC_EXCLUDE_GLOBS", "artifacts/champions/**")
DEFAULT_TRASH_DIR = os.getenv("NC_TRASH_DIR", ".trash")
DEFAULT_TRASH_RETENTION_DAYS = int(os.getenv("NC_TRASH_RETENTION_DAYS", "14"))

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIRS = [BASE_DIR / "artifacts"]
TMP_DIRS = [BASE_DIR / ".tmp", BASE_DIR / "data" / "tmp"]
JOBS_DIR = BASE_DIR / "jobs"
CHAMPIONS_DIR = BASE_DIR / "artifacts" / "champions"

LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "cleanup.log"
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
    details: List[Tuple[str, int, float, str]]  # (path, size, age_days, reason)


def human(nbytes: int) -> str:
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
                    continue
    return files


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def is_champion(path: Path) -> bool:
    return CHAMPIONS_DIR.exists() and is_under(path, CHAMPIONS_DIR)


def parse_exclusions(globs_str: str) -> List[str]:
    globs = [g.strip() for g in (globs_str or "").split(",") if g.strip()]
    if DEFAULT_EXCLUDE_GLOBS and not globs:
        globs = [g.strip() for g in DEFAULT_EXCLUDE_GLOBS.split(",") if g.strip()]
    return globs


# ---------------------------
# FIX Paso 3 (robustez rutas)
# ---------------------------
def _rel_from_base(path: Path) -> str:
    """
    Devuelve una ruta relativa tipo POSIX para comparar con patrones de exclusión.
    - Si el path no está dentro de BASE_DIR, intenta cortar desde 'artifacts'.
    - En última instancia usa el nombre de archivo.
    """
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except Exception:
        parts = path.resolve().parts
        if "artifacts" in parts:
            idx = parts.index("artifacts")
            return "/".join(parts[idx:])
        return path.name  # fallback mínimo


def is_excluded(path: Path, exclude_globs: List[str]) -> bool:
    """
    Devuelve True si el path coincide con alguno de los patrones de exclusión.
    Acepta patrones relativos tipo 'artifacts/champions/**'.
    Compara contra:
      - la ruta relativa robusta (_rel_from_base)
      - la ruta absoluta en POSIX, con un comodín **/ para mayor tolerancia
    """
    rel = _rel_from_base(path)
    abs_posix = path.resolve().as_posix()
    for pat in exclude_globs or []:
        pat_posix = pat.replace("\\", "/")
        if fnmatch.fnmatch(rel, pat_posix):
            return True
        if fnmatch.fnmatch(abs_posix, f"**/{pat_posix}"):
            return True
    return False


def group_key(file: FileInfo) -> str:
    parts = file.path.parts
    if "artifacts" in parts:
        i = parts.index("artifacts")
        group = parts[i + 1:i + 3]
        if group:
            return "/".join(group)
    return str(file.path.parent)


def select_candidates(files: List[FileInfo], retention_days: int, keep_last: int,
                      exclude_globs: List[str]) -> List[Tuple[FileInfo, str]]:
    """
    Devuelve lista de (FileInfo, reason)
    - reason ∈ {"age", "surplus", "age+surplus"}
    """
    # Filtrar champions y exclusiones
    eligible = [
        f for f in files
        if not is_champion(f.path) and not is_excluded(f.path, exclude_globs)
    ]

    # 1) Antigüedad
    oldies = {f.path: ("age", f) for f in eligible if f.age_days > retention_days}

    # 2) Excedente por grupo
    by_group: Dict[str, List[FileInfo]] = {}
    for f in eligible:
        by_group.setdefault(group_key(f), []).append(f)

    surplus_map: Dict[Path, Tuple[str, FileInfo]] = {}
    for _, lst in by_group.items():
        lst_sorted = sorted(lst, key=lambda x: x.mtime, reverse=True)
        if len(lst_sorted) > keep_last:
            for f in lst_sorted[keep_last:]:
                surplus_map[f.path] = ("surplus", f)

    # Unir razones
    combined: Dict[Path, Tuple[str, FileInfo]] = surplus_map.copy()
    for p, (reason, f) in oldies.items():
        if p in combined:
            combined[p] = ("age+surplus", f)
        else:
            combined[p] = (reason, f)

    # Salida ordenada por mtime ascendente (más viejos primero)
    items = list(combined.values())
    items.sort(key=lambda t: t[1].mtime)
    return [(f, reason) for reason, f in items]


def inventory(retention_days: int, keep_last: int, exclude_globs: List[str]) -> InventoryReport:
    files = iter_files(ARTIFACTS_DIRS + TMP_DIRS + [JOBS_DIR])
    total_size = sum(f.size for f in files)

    candidates = select_candidates(files, retention_days, keep_last, exclude_globs)
    cand_size = sum(f.size for f, _ in candidates)
    details = [(str(f.path), f.size, f.age_days, reason) for f, reason in candidates]

    return InventoryReport(
        total_files=len(files),
        total_size_bytes=total_size,
        candidates_count=len(candidates),
        candidates_size_bytes=cand_size,
        details=details,
    )


def ensure_dirs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_action(action: str, file: FileInfo, reason: str):
    ensure_dirs()
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["timestamp", "action", "path", "size", "age_days", "reason"])
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            action,
            str(file.path),
            file.size,
            f"{file.age_days:.2f}",
            reason,
        ])


def safe_move_to_trash(file: FileInfo, trash_dir: Path):
    """
    Mueve el archivo a .trash/YYYYMMDD/<ruta_relativa> para borrado reversible.
    """
    date_bucket = datetime.utcnow().strftime("%Y%m%d")
    rel = file.path.resolve().relative_to(BASE_DIR.resolve())
    dst = trash_dir / date_bucket / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Evita sobrescribir: si existe, agrega sufijo con epoch
    if dst.exists():
        dst = dst.with_name(dst.name + f".{int(time.time())}")
    shutil.move(str(file.path), str(dst))


def purge_old_trash(trash_dir: Path, trash_retention_days: int):
    if not trash_dir.exists():
        return
    cutoff = time.time() - trash_retention_days * SECONDS_PER_DAY
    for p in trash_dir.rglob("*"):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_mtime < cutoff:
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                # intentar limpiar directorios vacíos luego
                pass
    # limpiar directorios vacíos
    for d in sorted([d for d in trash_dir.rglob("*") if d.is_dir()], reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="NeuroCampus cleanup tool (Día 2: borrado real con papelera)")
    parser.add_argument("--inventory", action="store_true", help="Sólo mostrar inventario resumido.")
    parser.add_argument("--dry-run", action="store_true", help="Simular eliminación sin borrar/mover.")
    parser.add_argument("--force", action="store_true", help="Activar borrado real (mover a papelera).")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--exclude-globs", type=str, default=DEFAULT_EXCLUDE_GLOBS)
    parser.add_argument("--trash-dir", type=str, default=DEFAULT_TRASH_DIR)
    parser.add_argument("--trash-retention-days", type=int, default=DEFAULT_TRASH_RETENTION_DAYS)
    args = parser.parse_args(argv)

    exclude_globs = parse_exclusions(args.exclude_globs)
    rep = inventory(args.retention_days, args.keep_last, exclude_globs)

    print("== NeuroCampus :: Limpieza (Día 2) ==")
    print(f"Total archivos: {rep.total_files}")
    print(f"Total tamaño:   {human(rep.total_size_bytes)}")
    print(f"Candidatos:     {rep.candidates_count}")
    print(f"Tamaño elegible para liberar: {human(rep.candidates_size_bytes)}\n")

    print("Top 50 candidatos (ruta | tamaño | edad_días | razón):")
    for path, size, age, reason in rep.details[:50]:
        print(f"  - {path} | {human(size)} | {age:.1f}d | {reason}")

    if args.inventory:
        print("\nModo: INVENTORY (no se elimina nada).")
        return 0

    if args.dry_run or (not args.force):
        print("\nModo: DRY-RUN (no se elimina/mueve nada). Use --force para mover a papelera.")
        return 0

    # Borrado real: mover a papelera + logs
    trash_dir = (BASE_DIR / args.trash_dir).resolve()
    print(f"\nBorrado real ACTIVADO (--force). Papelera: {trash_dir}")
    moved_bytes = 0
    for path, size, age, reason in rep.details:
        fi = FileInfo(path=Path(path), size=size, mtime=time.time() - age * SECONDS_PER_DAY)
        if not fi.path.exists():
            continue
        if is_champion(fi.path) or is_excluded(fi.path, exclude_globs):
            continue
        if fi.path.is_symlink():
            # seguridad: no manipular symlinks
            log_action("skip_symlink", fi, reason)
            continue
        try:
            safe_move_to_trash(fi, trash_dir)
            log_action("moved_to_trash", fi, reason)
            moved_bytes += fi.size
        except Exception as e:
            log_action(f"error:{type(e).__name__}", fi, reason)

    print(f"\nResultado: movidos {human(moved_bytes)} a {trash_dir}")
    print("Log de acciones:", LOG_FILE)

    # Mantenimiento de papelera
    purge_old_trash(trash_dir, args.trash_retention_days)
    return 0

# --- fachada programática para uso interno (API/servicios) ---
def run_cleanup(*,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    keep_last: int = DEFAULT_KEEP_LAST,
    exclude_globs_str: str | None = None,
    dry_run: bool = True,
    force: bool = False,
    trash_dir: str = DEFAULT_TRASH_DIR,
    trash_retention_days: int = DEFAULT_TRASH_RETENTION_DAYS,
):
    """
    Ejecuta inventario y, opcionalmente, el movimiento a papelera.
    Devuelve un dict con resumen y candidatos.
    """
    exclude_globs = parse_exclusions(exclude_globs_str or DEFAULT_EXCLUDE_GLOBS)
    rep = inventory(retention_days, keep_last, exclude_globs)

    summary = {
        "total_files": rep.total_files,
        "total_size_bytes": rep.total_size_bytes,
        "candidates_count": rep.candidates_count,
        "candidates_size_bytes": rep.candidates_size_bytes,
    }
    candidates = [
        {"path": path, "size": size, "age_days": age, "reason": reason}
        for (path, size, age, reason) in rep.details
    ]

    moved_bytes = 0
    actions = []

    if not dry_run and force:
        # Borrado real: mover a papelera + log (igual a CLI)
        tdir = (BASE_DIR / trash_dir).resolve()
        for item in candidates:
            p = Path(item["path"])
            if not p.exists():
                continue
            fi = FileInfo(path=p, size=item["size"],
                          mtime=time.time() - item["age_days"] * SECONDS_PER_DAY)

            if is_champion(fi.path) or is_excluded(fi.path, exclude_globs) or fi.path.is_symlink():
                # se loggea un skip para trazabilidad
                log_action("skip", fi, item["reason"])
                continue

            try:
                safe_move_to_trash(fi, tdir)
                log_action("moved_to_trash", fi, item["reason"])
                moved_bytes += fi.size
                actions.append({"action": "moved_to_trash", "path": str(fi.path), "size": fi.size})
            except Exception as e:
                log_action(f"error:{type(e).__name__}", fi, item["reason"])
                actions.append({"action": f"error:{type(e).__name__}", "path": str(fi.path)})

        purge_old_trash(tdir, trash_retention_days)

    result = {
        "summary": summary,
        "candidates": candidates,
        "force": force,
        "dry_run": dry_run,
        "moved_bytes": moved_bytes,
        "actions": actions,
        "log_file": str(LOG_FILE),
        "trash_dir": str((BASE_DIR / trash_dir).resolve()),
    }
    return result

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
