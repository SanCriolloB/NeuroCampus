import os
from pathlib import Path
import time

from tools.cleanup import FileInfo, select_candidates, SECONDS_PER_DAY

def test_select_candidates_respects_keep_last_and_age(tmp_path: Path):
    # Crear 5 archivos del mismo "grupo"
    now = time.time()
    files = []
    for idx in range(5):
        p = tmp_path / f"artifacts/modelX/run_{idx}/file.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(128))
        # simular mtimes de más viejo a más nuevo
        mtime = now - (idx * SECONDS_PER_DAY)  # idx=0 más nuevo
        os.utime(p, (mtime, mtime))
        files.append(FileInfo(path=p, size=128, mtime=mtime))

    # keep_last=3 -> 2 excedentes
    # retention_days=0 -> todos "viejos", pero el excedente ya debe atraparlos
    candidates = select_candidates(files, retention_days=0, keep_last=3)
    assert len(candidates) >= 2
