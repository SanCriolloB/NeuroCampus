"""
neurocampus.data.feature_pack_auto
=================================

Utilidades para construir el **Feature Pack** usado por los modelos RBM.

Motivación
----------

La **RBM Restringida** (y opcionalmente la RBM General) entrenan mejor y con
mayor estabilidad cuando consumen un dataset ya transformado a una matriz de
features lista para el modelo.

En NeuroCampus esto se materializa como:

- ``artifacts/features/<dataset_id>/train_matrix.parquet``

Este archivo se genera con :func:`neurocampus.data.features_prepare.prepare_feature_pack`
a partir de un parquet *labeled* (típicamente ``data/labeled/<dataset>_beto.parquet``)
o del histórico unificado labeled (``historico/unificado_labeled.parquet``).

Este módulo centraliza la lógica para:

- Resolver el path de salida.
- Evitar recomputación si el feature pack ya existe.
- Ejecutar el build de forma explícita (forzada o lazy).

.. note::
   Esta utilidad es "backend-only": el frontend no depende de ella, pero los
   endpoints pueden invocarla para automatizar la preparación de insumos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def ensure_feature_pack(
    *,
    base_dir: Path,
    dataset_id: str,
    input_uri: str,
    output_dir: Optional[str] = None,
    force: bool = False,
    text_feats_mode: str = 'none',
    text_col: Optional[str] = None,
    text_n_components: int = 64,
    text_min_df: int = 2,
    text_max_features: int = 20000,
    text_random_state: int = 42,
) -> Dict[str, str]:
    """
    Asegura que exista el feature pack ``train_matrix.parquet`` para un dataset.

    Si el archivo ya existe y ``force=False``, la función retorna sin recomputar.

    Parameters
    ----------
    base_dir:
        Raíz del repo/proyecto (donde existen ``artifacts/`` y ``data/``).
        En routers típicamente corresponde a ``BASE_DIR``.
    dataset_id:
        Identificador del dataset (ej. ``"2025-1"``).
    input_uri:
        Ruta/URI relativa dentro del proyecto hacia el dataset de entrada.
        Normalmente es el parquet *labeled* (ej. ``"data/labeled/2025-1_beto.parquet"``)
        o el histórico labeled (ej. ``"historico/unificado_labeled.parquet"``).
    output_dir:
        Directorio de salida. Si es ``None``, se usa::

            <base_dir>/artifacts/features/<dataset_id>

        Puede ser relativo o absoluto.
    force:
        Si ``True``, reconstruye el feature pack incluso si ya existe.

    Returns
    -------
    Dict[str, str]
        Diccionario con rutas a los artefactos generados (train_matrix, índices, meta, etc).
        El formato depende de :func:`neurocampus.data.features_prepare.prepare_feature_pack`.

    Raises
    ------
    ImportError
        Si no se puede importar :func:`prepare_feature_pack`.
    Exception
        Cualquier excepción que lance el builder (I/O, parquet, validación, etc).

    Examples
    --------
    >>> from pathlib import Path
    >>> from neurocampus.data.feature_pack_auto import ensure_feature_pack
    >>> ensure_feature_pack(
    ...     base_dir=Path(".").resolve(),
    ...     dataset_id="2025-1",
    ...     input_uri="data/labeled/2025-1_beto.parquet",
    ... )
    """
    ds = str(dataset_id or "").strip()
    if not ds:
        raise ValueError("dataset_id vacío")

    # Directorio de salida por defecto.
    out_dir = Path(output_dir) if output_dir else (base_dir / "artifacts" / "features" / ds)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()

    train_path = out_dir / "train_matrix.parquet"
    if train_path.exists() and not force:
        # Retornar un payload mínimo y consistente.
        return {"train_matrix": str(train_path)}

    # Import tardío para evitar costos y dependencias cuando no se usa.
    from neurocampus.data.features_prepare import prepare_feature_pack  # noqa: WPS433

    artifacts = prepare_feature_pack(
        base_dir=base_dir,
        dataset_id=ds,
        input_uri=str(input_uri),
        output_dir=str(out_dir),
        text_feats_mode=text_feats_mode,
        text_col=text_col,
        text_n_components=int(text_n_components),
        text_min_df=int(text_min_df),
        text_max_features=int(text_max_features),
        text_random_state=int(text_random_state),
    )
    return {str(k): str(v) for k, v in (artifacts or {}).items()}
