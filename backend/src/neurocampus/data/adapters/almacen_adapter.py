# backend/src/neurocampus/data/adapters/almacen_adapter.py
from __future__ import annotations
from pathlib import Path
from typing import List
from urllib.parse import urlparse

class AlmacenAdapter:
    """
    Adapter de almacenamiento mínimo para FS local.
    - base_uri: 'localfs://<ruta_base>'  (p.ej. 'localfs://.' o 'localfs://data-root')
    - Métodos usados por data/strategies: ls(), exists(), makedirs(), open()
    """
    def __init__(self, base_uri: str = "localfs://."):
        self.base_dir = self._parse_base(base_uri)

    def _parse_base(self, base_uri: str) -> Path:
        """
        Traduce 'localfs://<ruta>' a un Path absoluto.
        """
        parsed = urlparse(base_uri)
        if parsed.scheme not in ("localfs", "", None):
            raise ValueError(f"Esquema no soportado: {parsed.scheme}")
        # 'localfs://neurocampus' -> netloc='neurocampus', path=''
        # 'localfs:///abs'        -> netloc='', path='/abs'
        raw = (parsed.netloc or "") + (parsed.path or "")
        p = Path(raw if raw else ".").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _abs(self, rel: str | Path) -> Path:
        return (self.base_dir / Path(rel)).resolve()

    def ls(self, prefix: str) -> List[str]:
        """
        Lista entradas directas dentro de 'prefix' (carpetas/archivos de primer nivel).
        Devuelve rutas relativas a base_dir.
        """
        p = self._abs(prefix)
        if not p.exists():
            return []
        return [str(child.relative_to(self.base_dir)) for child in p.iterdir()]

    def exists(self, rel_path: str) -> bool:
        return self._abs(rel_path).exists()

    def makedirs(self, rel_path: str) -> None:
        self._abs(rel_path).mkdir(parents=True, exist_ok=True)

    def open(self, rel_path: str, mode: str = "r"):
        """
        Abre un handle estándar de archivo. Si el modo es de escritura,
        crea la carpeta padre si no existe.
        """
        abs_path = self._abs(rel_path)
        if any(m in mode for m in ("w", "a", "x", "+")):
            abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path.open(mode)
