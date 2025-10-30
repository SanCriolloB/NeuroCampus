import os
from pathlib import Path
import time

from tools.cleanup import (
    FileInfo, select_candidates, SECONDS_PER_DAY,
    parse_exclusions, is_excluded
)

def _mkfile(p: Path, size=64, age_days=0):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(os.urandom(size))
    if age_days > 0:
        mtime = time.time() - age_days * SECONDS_PER_DAY
        os.utime(p, (mtime, mtime))

def test_exclusions_and_group_selection(tmp_path: Path):
    # Estructura tipo artifacts/
    base = tmp_path
    a1 = base / "artifacts" / "modelX" / "runA" / "f.bin"
    a2 = base / "artifacts" / "modelX" / "runB" / "f.bin"
    ch = base / "artifacts" / "champions" / "modelX" / "best.bin"

    _mkfile(a1, age_days=120)  # viejo
    _mkfile(a2, age_days=10)   # reciente
    _mkfile(ch, age_days=200)  # champions -> excluido por glob

    # Simular lectura de FileInfo desde estas rutas
    files = []
    for p in [a1, a2, ch]:
        st = p.stat()
        files.append(FileInfo(path=p, size=st.st_size, mtime=st.st_mtime))

    exclude_globs = parse_exclusions("artifacts/champions/**")
    cands = select_candidates(files, retention_days=90, keep_last=1, exclude_globs=exclude_globs)

    paths = [f.path for f, _ in cands]
    assert a1 in paths            # viejo o excedente
    assert ch not in paths        # excluido por champions
    # como keep_last=1, el run más nuevo (a2) se mantiene

def test_is_excluded_glob(tmp_path: Path):
    p = tmp_path / "artifacts" / "champions" / "x" / "y.bin"
    _mkfile(p)
    assert is_excluded(p, ["artifacts/champions/**"])
