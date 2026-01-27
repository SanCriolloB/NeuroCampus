# backend/src/neurocampus/data/strategies/unificacion.py
from __future__ import annotations

"""Estrategia de unificación histórica.

Este módulo pertenece al dominio **Datos** y su responsabilidad es crear
artefactos históricos reproducibles en disco, para consumo posterior por la
pestaña **Modelos**.

Soporta layouts:
- Nuevo (archivo plano):
  - datasets/<dataset_id>.parquet|csv|xlsx
  - data/labeled/<dataset_id>_beto.parquet
- Legacy (por carpeta):
  - datasets/<dataset_id>/data.parquet|csv|xlsx

Produce:
- historico/unificado.parquet
- historico/unificado_labeled.parquet
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path

import pandas as pd

from ..adapters.almacen_adapter import AlmacenAdapter
from ..adapters.formato_adapter import read_file
from ..adapters.dataframe_adapter import as_df
from ..utils.headers import normalizar_encabezados

DEDUP_KEYS = ["periodo", "codigo_materia", "grupo", "cedula_profesor"]


class UnificacionStrategy:
    """Unifica datasets históricos y genera artefactos en /historico."""

    def __init__(self, base_uri: str | None = None):
        """
        Inicializa el store para leer/escribir artefactos del pipeline.

        Problema que resuelve:
        - En jobs/background tasks el CWD puede NO ser la raíz del repo.
        - Si se pasa `base_uri="localfs://."`, el store apunta al CWD (p.ej. /backend),
        y `data/labeled/` queda “vacío” aunque exista en la raíz del repo.

        Solución:
        - Detectar la raíz del repo de forma robusta.
        - Si `base_uri` es None o es `localfs://.` → forzar raíz del repo.
        - Si `base_uri` apunta a un sitio que no tiene `data/` ni `datasets/`,
        aplicar fallback automático a la raíz del repo.
        """
        project_root = self._find_project_root()

        def _is_dot_uri(u: str | None) -> bool:
            u = (u or "").strip()
            return u in ("localfs://.", "localfs://./", "localfs://")

        if base_uri is None or _is_dot_uri(base_uri):
            base_uri = f"localfs://{project_root.as_posix()}"

        store = AlmacenAdapter(base_uri)

        # Fallback defensivo: si no vemos estructura del proyecto, usar raíz repo.
        if (not store.exists("data")) and (not store.exists("datasets")):
            store = AlmacenAdapter(f"localfs://{project_root.as_posix()}")

        self.store = store


    @staticmethod
    def _find_project_root() -> Path:
        """
        Encuentra la raíz del repo de NeuroCampus de forma robusta.

        Criterio:
        - Un directorio que contenga `data/` y `datasets/`.
        - Si no se encuentra, fallback al layout estándar (parents[5]).
        """
        here = Path(__file__).resolve()
        for p in here.parents:
            if (p / "data").exists() and (p / "datasets").exists():
                return p
        return here.parents[5]

    
    # ----------------------------
    # Listing helpers
    # ----------------------------

    def _ls_variants(self, prefix: str) -> list[str]:
        """
        Lista un prefix intentando variantes con y sin trailing slash.
        Evita inconsistencias entre adapters.
        """
        pref = prefix.strip()
        candidates = {pref, pref.rstrip("/"), pref.rstrip("\\")}
        candidates.add(pref.rstrip("/\\") + "/")

        out: list[str] = []
        for c in candidates:
            try:
                out.extend(self.store.ls(c))
            except Exception:
                continue

        # dedupe manteniendo orden
        seen = set()
        uniq: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq


    def listar_datasets_raw(self, prefix: str = "datasets/") -> list[str]:
        """
        Lista datasets disponibles en datasets/ soportando:
        - datasets/<id>.parquet|csv|xlsx
        - datasets/<id>/data.parquet|csv|xlsx (legacy)
        """
        items = self.store.ls(prefix)
        out: Set[str] = set()

        for it in items:
            name = Path(it).name
            p = Path(name)

            # Caso archivo plano
            if p.suffix.lower() in {".parquet", ".csv", ".xlsx"}:
                out.add(p.stem)
                continue

            # Caso carpeta legacy
            folder = f"{prefix.rstrip('/')}/{p.name}"
            for fname in ("data.parquet", "data.csv", "data.xlsx"):
                if self.store.exists(f"{folder}/{fname}"):
                    out.add(p.name)
                    break

        return sorted(out)

    def listar_datasets_labeled(self, prefix: str = "data/labeled/") -> list[str]:
        """
        Lista datasets etiquetados disponibles en data/labeled/:
        - <id>_beto.parquet
        - <id>_teacher.parquet (compat)
        - <id>_beto.csv
        - <id>_teacher.csv
        """
        items = self._ls_variants(prefix)
        out: Set[str] = set()

        for it in items:
            name = Path(it).name
            if name.endswith("_beto.parquet"):
                out.add(name[: -len("_beto.parquet")])
            elif name.endswith("_teacher.parquet"):
                out.add(name[: -len("_teacher.parquet")])
            elif name.endswith("_beto.csv"):
                out.add(name[: -len("_beto.csv")])
            elif name.endswith("_teacher.csv"):
                out.add(name[: -len("_teacher.csv")])

        return sorted(out)


    # ----------------------------
    # Resolve URIs
    # ----------------------------

    def listar_periodos_raw(self, datasets_prefix: str = "datasets") -> list[str]:
        """
        Lista datasets crudos disponibles en `datasets/` (layout nuevo + legacy).

        Soporta:
        - datasets/<id>.parquet|csv|xlsx
        - datasets/<id>/data.parquet|csv|xlsx (legacy)
        """
        out: set[str] = set()
        prefixes = (datasets_prefix.rstrip("/"), datasets_prefix.rstrip("/") + "/")

        for pref in prefixes:
            try:
                items = self.store.ls(pref)
            except Exception:
                items = []

            for it in items:
                name = Path(it).name
                p = Path(name)

                # Archivo plano
                if p.suffix.lower() in {".parquet", ".csv", ".xlsx"}:
                    out.add(p.stem)
                    continue

                # Posible carpeta legacy
                folder = f"{datasets_prefix.rstrip('/')}/{p.name}"
                for fname in ("data.parquet", "data.csv", "data.xlsx"):
                    if self.store.exists(f"{folder}/{fname}"):
                        out.add(p.name)
                        break

        return sorted(out)


    def _resolve_dataset_uri(self, dataset_id: str) -> str:
        """Resuelve la URI del dataset crudo (datasets/) para un dataset_id."""
        flat = [
            f"datasets/{dataset_id}.parquet",
            f"datasets/{dataset_id}.csv",
            f"datasets/{dataset_id}.xlsx",
        ]
        for uri in flat:
            if self.store.exists(uri):
                return uri

        folder = f"datasets/{dataset_id}"
        legacy = (f"{folder}/data.parquet", f"{folder}/data.csv", f"{folder}/data.xlsx")
        for uri in legacy:
            if self.store.exists(uri):
                return uri

        raise FileNotFoundError(f"No se encontró dataset {dataset_id} en datasets/")

    def _resolve_labeled_uri(self, dataset_id: str) -> str:
        """Resuelve la URI del dataset etiquetado (data/labeled) para un dataset_id."""
        candidates = [
            f"data/labeled/{dataset_id}_beto.parquet",
            f"data/labeled/{dataset_id}_teacher.parquet",
            f"data/labeled/{dataset_id}_beto.csv",
            f"data/labeled/{dataset_id}_teacher.csv",
        ]
        for uri in candidates:
            if self.store.exists(uri):
                return uri
        raise FileNotFoundError(f"No se encontró labeled para {dataset_id} en data/labeled/")

    # ----------------------------
    # Read/normalize
    # ----------------------------

    def _read_any(self, uri: str) -> pd.DataFrame:
        """Lee csv/xlsx/parquet usando adapters y lo convierte a pandas."""
        with self.store.open(uri, "rb") as fh:
            df_like = read_file(fh, uri)
        pdf = as_df(df_like)

        if hasattr(pdf, "to_pandas"):
            try:
                pdf = pdf.to_pandas()
            except Exception:
                pdf = pd.DataFrame(pdf)

        pdf.columns = normalizar_encabezados(list(pdf.columns))
        return pdf

    def _ensure_periodo(self, pdf: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        """
        Asegura que exista y sea válida la columna periodo por fila.
        Corrige None/NaN/"None"/vacíos.
        """
        if "periodo" not in pdf.columns:
            pdf["periodo"] = dataset_id
            return pdf

        s = pdf["periodo"].astype("string")
        s = s.fillna(dataset_id)
        s = s.replace({"None": dataset_id, "nan": dataset_id, "NaN": dataset_id, "<NA>": dataset_id})
        pdf["periodo"] = s
        pdf.loc[pdf["periodo"].astype(str).str.strip().eq(""), "periodo"] = dataset_id
        return pdf

    def _leer_raw(self, dataset_id: str) -> pd.DataFrame:
        """Lee dataset crudo/processed de un dataset_id y asegura periodo."""
        uri = self._resolve_dataset_uri(dataset_id)
        pdf = self._read_any(uri)
        return self._ensure_periodo(pdf, dataset_id)

    def _leer_labeled(self, dataset_id: str) -> pd.DataFrame:
        """Lee labeled de un dataset_id y asegura periodo."""
        uri = self._resolve_labeled_uri(dataset_id)
        pdf = self._read_any(uri)
        return self._ensure_periodo(pdf, dataset_id)

    # ----------------------------
    # Write helpers
    # ----------------------------

    def _to_text(self, x):
        """
        Convierte valores heterogéneos a texto de forma segura.

        - bytes/bytearray/memoryview -> decode utf-8 (replace)
        - NaN/<NA>/None -> None
        - otros -> str(...)
        """
        import pandas as pd

        if x is None:
            return None
        try:
            # cubre NaN y pandas NA
            if pd.isna(x):
                return None
        except Exception:
            pass

        if isinstance(x, (bytes, bytearray, memoryview)):
            return bytes(x).decode("utf-8", errors="replace")

        return str(x)


    def _sanitize_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza columnas para que PyArrow pueda convertir el DataFrame a Parquet.

        Problema típico:
        - columnas dtype=object con mezcla de tipos (ej: bytes + int) → ArrowTypeError

        Regla:
        - Forzar 'id' a texto si existe (es el caso más común del fallo).
        - Para otras columnas object: si detectamos bytes o tipos complejos (dict/list/tuple/set),
        convertimos toda la columna a texto.
        """
        import pandas as pd

        out = df.copy()

        # Caso crítico observado en tu error
        if "id" in out.columns:
            out["id"] = out["id"].map(self._to_text)

        for col in out.columns:
            if out[col].dtype != "object":
                continue

            sample = out[col].dropna().head(200).tolist()
            if not sample:
                continue

            has_bytes = any(isinstance(v, (bytes, bytearray, memoryview)) for v in sample)
            has_complex = any(isinstance(v, (dict, list, tuple, set)) for v in sample)

            # Si hay tipos que Arrow suele rechazar o mezclar, normalizamos a texto
            if has_bytes or has_complex:
                out[col] = out[col].map(self._to_text)

        return out


    def _dedupe_concat(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatena y elimina duplicados por claves canónicas cuando existan."""
        if not frames:
            raise ValueError("No hay frames para unificar")
        big = pd.concat(frames, ignore_index=True, copy=False)
        keys = [k for k in DEDUP_KEYS if k in big.columns]
        if keys:
            big = big.drop_duplicates(subset=keys)
        else:
            big = big.drop_duplicates()
        return big

    def _write_parquet(self, df: pd.DataFrame, out_uri: str) -> None:
        """
        Escribe parquet de forma segura y atómica.

        1) Sanitiza columnas object (ej: id bytes+int) para evitar ArrowTypeError.
        2) Escribe a un archivo temporal y luego hace replace atómico.
        Así evitamos archivos 0 bytes cuando hay error durante to_parquet.

        Nota:
        - Este strategy usa `localfs`, así que resolvemos paths con `self.store.base_dir`.
        """
        import os
        from pathlib import Path

        safe_df = self._sanitize_for_parquet(df)

        base_dir = Path(self.store.base_dir)
        abs_path = (base_dir / Path(out_uri)).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")

        # Escribir por path (mejor compatibilidad con pyarrow en Windows)
        safe_df.to_parquet(tmp_path, index=False)

        # Replace atómico
        os.replace(tmp_path, abs_path)


    # ----------------------------
    # Public API
    # ----------------------------

    def periodo_actual(self) -> Tuple[str, Dict[str, Any]]:
        """Último dataset_id raw lexicográfico → historico/periodo_actual/<id>.parquet"""
        ids = self.listar_datasets_raw()
        if not ids:
            raise RuntimeError("No hay datasets en datasets/")
        ultimo = ids[-1]
        pdf = self._leer_raw(ultimo)

        out_uri = f"historico/periodo_actual/{ultimo}.parquet"
        self._write_parquet(pdf, out_uri)
        return out_uri, {"dataset_id": ultimo, "rows": int(len(pdf))}

    def acumulado(self) -> Tuple[str, Dict[str, Any]]:
        """Concatena todos los datasets raw → historico/unificado.parquet"""
        ids = self.listar_datasets_raw()
        if not ids:
            raise RuntimeError("No hay datasets en datasets/")
        frames = [self._leer_raw(i) for i in ids]
        pdf = self._dedupe_concat(frames)

        out_uri = "historico/unificado.parquet"
        self._write_parquet(pdf, out_uri)
        return out_uri, {"datasets": ids, "rows": int(len(pdf))}

    def ventana(
        self,
        ultimos: Optional[int] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Unifica una ventana de datasets raw → historico/ventanas/unificado_<tag>.parquet"""
        ids = self.listar_datasets_raw()
        if not ids:
            raise RuntimeError("No hay datasets en datasets/")

        if ultimos:
            sel = ids[-ultimos:]
        else:
            if not (desde and hasta):
                raise ValueError("Se requiere 'ultimos' o bien ('desde' y 'hasta')")
            sel = [i for i in ids if desde <= i <= hasta]

        frames = [self._leer_raw(i) for i in sel]
        pdf = self._dedupe_concat(frames)

        tag = f"{sel[0]}_{sel[-1]}" if sel else "vacia"
        out_uri = f"historico/ventanas/unificado_{tag}.parquet"
        self._write_parquet(pdf, out_uri)
        return out_uri, {"datasets": sel, "rows": int(len(pdf))}

    def acumulado_labeled(self) -> Tuple[str, Dict[str, Any]]:
        """Unifica labeled disponibles → historico/unificado_labeled.parquet"""
        ids = self.listar_datasets_labeled()
        frames: List[pd.DataFrame] = []
        skipped: List[str] = []

        for i in ids:
            try:
                frames.append(self._leer_labeled(i))
            except FileNotFoundError:
                skipped.append(i)

        if not frames:
            raise RuntimeError(
                "No hay datasets etiquetados para unificar. "
                "Asegúrate de correr BETO al menos en un periodo."
            )

        pdf = self._dedupe_concat(frames)
        out_uri = "historico/unificado_labeled.parquet"
        self._write_parquet(pdf, out_uri)

        return out_uri, {"datasets": ids, "rows": int(len(pdf)), "skipped": skipped}
