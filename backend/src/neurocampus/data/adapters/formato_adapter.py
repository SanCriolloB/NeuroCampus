"""
FormatoAdapter: lectura robusta de CSV / XLSX / Parquet hacia el engine configurado.
- Detecta por extensión o parámetro explícito.
- Reintenta CSV con distintos encodings y separadores comunes.
- No guarda; solo lee a memoria para validar.
"""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, Optional
import csv

from .dataframe_adapter import _ENGINE


def infer_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt"):
        return "csv"
    if ext in (".xlsx", ".xls"):
        return "xlsx"
    if ext in (".parquet", ".pq"):
        return "parquet"
    # fallback seguro
    return "csv"


# ---------------------------------------------------------------------
# Helpers de lectura
# ---------------------------------------------------------------------

def _seek0(f: BinaryIO) -> None:
    try:
        f.seek(0)
    except Exception:
        pass


def _read_csv_pandas(fileobj: BinaryIO):
    import pandas as pd

    # 1) utf-8-sig + autodetección de separador (sep=None requiere engine='python')
    _seek0(fileobj)
    try:
        return pd.read_csv(fileobj, encoding="utf-8-sig", sep=None, engine="python")
    except UnicodeDecodeError:
        pass
    except Exception:
        # seguimos con reintentos
        pass

    # 2) latin-1 + autodetección
    _seek0(fileobj)
    try:
        return pd.read_csv(fileobj, encoding="latin-1", sep=None, engine="python")
    except Exception:
        pass

    # 3) Sniffer de separador (por si el caso anterior no detectó)
    _seek0(fileobj)
    sample = fileobj.read(4096)
    try:
        sample_text = sample.decode("utf-8", errors="ignore")
    except Exception:
        sample_text = ""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=";,|\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","  # fallback

    _seek0(fileobj)
    # ignoramos errores de encoding residuales
    return pd.read_csv(fileobj, sep=delimiter, encoding_errors="ignore")


def _read_csv_polars(fileobj: BinaryIO):
    import polars as pl

    # 1) intento por defecto (coma)
    _seek0(fileobj)
    try:
        return pl.read_csv(fileobj)
    except Exception:
        pass

    # 2) otros separadores comunes
    for sep in [";", "|", "\t"]:
        _seek0(fileobj)
        try:
            return pl.read_csv(fileobj, separator=sep)
        except Exception:
            continue

    # 3) encoding laxo (utf8-lossy)
    _seek0(fileobj)
    return pl.read_csv(fileobj, encoding="utf8-lossy")


# ---------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------

def read_file(fileobj: BinaryIO, filename: str, explicit: Optional[str] = None):
    """
    Lee un archivo a DataFrame del engine configurado:
      - CSV: reintentos con utf-8-sig / latin-1 / separadores (; | tab)
      - XLSX: pandas (y puente a polars cuando aplique)
      - Parquet: nativo en cada engine
    """
    fmt = (explicit or infer_format(filename)).lower()
    _seek0(fileobj)

    if _ENGINE == "polars":
        import polars as pl

        if fmt == "csv":
            return _read_csv_polars(fileobj)

        if fmt == "xlsx":
            # polars no lee XLSX nativamente → usar pandas y convertir
            import pandas as pd
            _seek0(fileobj)
            df_pd = pd.read_excel(fileobj)  # requiere openpyxl instalado
            return pl.from_pandas(df_pd)

        if fmt == "parquet":
            _seek0(fileobj)
            return pl.read_parquet(fileobj)

        raise ValueError(f"Formato no soportado: {fmt}")

    # Engine: pandas
    import pandas as pd

    if fmt == "csv":
        return _read_csv_pandas(fileobj)

    if fmt == "xlsx":
        _seek0(fileobj)
        return pd.read_excel(fileobj)  # engine=openpyxl si está disponible

    if fmt == "parquet":
        _seek0(fileobj)
        return pd.read_parquet(fileobj)

    raise ValueError(f"Formato no soportado: {fmt}")
