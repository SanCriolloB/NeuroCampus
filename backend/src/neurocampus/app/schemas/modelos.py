"""
neurocampus.app.schemas.modelos
================================

Esquemas (Pydantic) para la API de **Modelos**.

Este módulo define los request/response usados por los endpoints del router
``/modelos`` (entrenamiento, estado de jobs, listados de runs, champion, etc.).

Cambios principales (alineación con flujo actualizado)
------------------------------------------------------
- Se amplía :class:`EntrenarRequest` para soportar:

  - ``dataset_id`` (alias conveniente del dataset/periodo; compatible con ``periodo_actual``).
  - ``data_source``: ``feature_pack`` (recomendado), ``labeled`` (fallback), ``unified_labeled``.
  - ``target_mode``: por defecto ``sentiment_probs`` (usa ``p_neg/p_neu/p_pos``).
  - ``split_mode`` y ``val_ratio`` para evaluación real.
  - ``include_teacher_materia`` y **``teacher_materia_mode``** (evita que se pierda en el request).
  - ``auto_prepare`` para preparar artifacts cuando sea viable.

- Se amplía :class:`EstadoResponse` para devolver también:
  - ``model`` y ``params`` (para que UI pueda mostrar configuración real).
  - ``champion_promoted`` y ``time_total_ms`` (útiles para auditoría).

Compatibilidad hacia atrás
--------------------------
- Se mantiene ``periodo_actual`` y ``data_ref`` como campos legacy.
- ``dataset_id`` y ``periodo_actual`` se sincronizan automáticamente.
- Se conserva comportamiento tolerante ante campos extra (extra="ignore")
  para evitar romper clientes legacy.

Notas para Sphinx
-----------------
Los docstrings están escritos en reST para que Sphinx pueda renderizarlos con
``autodoc`` / ``napoleon``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, model_validator, ConfigDict


# ---------------------------------------------------------------------------
# Tipos comunes (enums via Literal)
# ---------------------------------------------------------------------------

ModeloName = Literal["rbm_general", "rbm_restringida", "dbm_manual"]
"""Nombre lógico del modelo.

- ``rbm_general``: RBM + head (general)
- ``rbm_restringida``: RBM variante restringida
- ``dbm_manual``: DBM (experimental / opcional)
"""

DataSource = Literal["feature_pack", "labeled", "unified_labeled"]
"""Fuente de datos para entrenamiento."""

TargetMode = Literal["sentiment_probs", "sentiment_label", "score_only"]
"""Modo del objetivo (target) para entrenamiento.

- ``sentiment_probs``: usa probabilidades soft ``p_neg/p_neu/p_pos`` (recomendado).
- ``sentiment_label``: usa etiqueta dura (si existe en dataset).
- ``score_only``: reservado / experimental.
"""

SplitMode = Literal["temporal", "stratified", "random"]
"""Estrategia de split train/val."""

Metodologia = Literal["periodo_actual", "acumulado", "ventana"]
"""Metodología de selección de datos (sobre el histórico)."""

TeacherMateriaMode = Literal["embed", "onehot", "none"]
"""Modo para incluir docente/materia como features.

- ``embed``: embeddings (hash-buckets + dim) (recomendado).
- ``onehot``: one-hot (solo viable si cardinalidad es pequeña).
- ``none``: desactiva explícitamente el uso de docente/materia.
"""

JobStatus = Literal["queued", "running", "completed", "failed", "unknown"]
"""Estados posibles reportados por el job de entrenamiento."""


# ---------------------------------------------------------------------------
# Request/Response: entrenamiento
# ---------------------------------------------------------------------------

class EntrenarRequest(BaseModel):
    """
    Request para lanzar el entrenamiento de un modelo.

    La fuente de entrenamiento puede ser un artifact reproducible (**feature-pack**)
    o dataframes derivados (**labeled / unified_labeled**).

    .. important::
       Para el flujo actualizado, lo recomendado es:
       ``data_source="feature_pack"`` y ``target_mode="sentiment_probs"``.

    :param modelo: Tipo de modelo a entrenar.
    :param dataset_id: Identificador del dataset (normalmente el periodo, p. ej. ``2024-2``).
        Se sincroniza con ``periodo_actual`` para compatibilidad.
    :param periodo_actual: Campo legacy (periodo de referencia). Si se omite pero hay
        ``dataset_id``, se copia automáticamente.
    :param metodologia: Estrategia de selección de datos (periodo actual / acumulado / ventana).
    :param ventana_n: Tamaño de la ventana si ``metodologia="ventana"``.
    :param data_source: Fuente de datos para entrenamiento.
    :param data_ref: Override manual (legacy/debug). Si se provee, el backend puede usarlo
        como ruta explícita.
    :param target_mode: Objetivo a entrenar (por defecto ``sentiment_probs``).
    :param include_teacher_materia: Si ``True``, incluir features de docente/materia.
    :param teacher_materia_mode: Modo para representar docente/materia (embed/onehot/none).
        Si ``include_teacher_materia=True`` y este campo se omite, el backend debería
        aplicar un default (normalmente ``embed``).
    :param auto_prepare: Si ``True``, el backend intentará generar artifacts faltantes
        (unificado/feature-pack) cuando sea posible.
    :param split_mode: Cómo hacer train/val.
    :param val_ratio: Proporción del set de validación (0..0.5 recomendado).
    :param epochs: Número de épocas de entrenamiento.
    :param hparams: Hiperparámetros específicos del modelo (dict flexible).
    """

    # Mantener tolerancia a campos extra (compatibilidad)
    model_config = ConfigDict(extra="ignore")

    modelo: ModeloName = Field(
        description="Tipo de modelo a entrenar (rbm_general | rbm_restringida | dbm_manual)."
    )

    # -----------------------------
    # Identidad del dataset (nuevo)
    # -----------------------------
    dataset_id: Optional[str] = Field(
        default=None,
        description="Identificador del dataset (recomendado). Usualmente coincide con el periodo (ej. '2024-2').",
    )

    # -----------------------------
    # Legacy (mantener compatibilidad)
    # -----------------------------
    periodo_actual: Optional[str] = Field(
        default=None,
        description="Campo legacy para periodo de referencia (ej. '2024-2'). Se sincroniza con dataset_id.",
    )

    data_ref: Optional[str] = Field(
        default=None,
        description=(
            "Override manual/legacy de la ruta de datos. "
            "Si no se provee, el backend resuelve la ruta según data_source + dataset_id."
        ),
    )

    # -----------------------------
    # Metodología de datos
    # -----------------------------
    metodologia: Metodologia = Field(
        default="periodo_actual",
        description=(
            "Estrategia de selección de datos: "
            "'periodo_actual' (solo dataset actual), "
            "'acumulado' (histórico <= periodo_actual), "
            "'ventana' (últimos N periodos)."
        ),
    )

    ventana_n: int = Field(
        default=4,
        ge=1,
        description="Tamaño de ventana para metodologia='ventana' (>=1).",
    )

    # -----------------------------
    # Flujo actualizado: datos/objetivo/split
    # -----------------------------
    data_source: DataSource = Field(
        default="feature_pack",
        description=(
            "Fuente de datos: "
            "'feature_pack' (artifacts/features/<dataset_id>/train_matrix.parquet), "
            "'labeled' (data/labeled/<dataset_id>_beto.parquet), "
            "'unified_labeled' (historico/unificado_labeled.parquet)."
        ),
    )

    target_mode: TargetMode = Field(
        default="sentiment_probs",
        description="Modo del target. Recomendado: sentiment_probs (p_neg/p_neu/p_pos).",
    )

    include_teacher_materia: bool = Field(
        default=True,
        description="Si True, incluye features de docente/materia en el entrenamiento.",
    )

    teacher_materia_mode: Optional[TeacherMateriaMode] = Field(
        default=None,
        description=(
            "Modo para representar docente/materia (embed/onehot/none). "
            "Si include_teacher_materia=True y se omite, el backend debería usar 'embed'."
        ),
    )

    auto_prepare: bool = Field(
        default=True,
        description=(
            "Si True, el backend puede intentar preparar artifacts faltantes (unificado/feature-pack) "
            "antes de entrenar."
        ),
    )

    split_mode: SplitMode = Field(
        default="temporal",
        description="Modo de split para train/val (temporal/stratified/random).",
    )

    val_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Proporción del set de validación (0..0.5).",
    )

    epochs: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Número de épocas de entrenamiento (1..500).",
    )

    # hparams se deja flexible (Any) para soportar floats/ints/bools/strings, etc.
    hparams: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_visible": None,  # si es None se infiere del dataset
            "n_hidden": 32,
            "lr": 0.01,
            "batch_size": 64,
            "cd_k": 1,
            "momentum": 0.5,
            "weight_decay": 0.0,
            "seed": 42,
            # Nota: teacher/materia hparams pueden vivir aquí si el strategy lo requiere:
            # "teacher_emb_buckets": 2048,
            # "materia_emb_buckets": 2048,
            # "tm_emb_dim": 16,
            # "tm_use_interaction": True,
        },
        description="Hiperparámetros del entrenamiento (dict flexible).",
    )

    @model_validator(mode="after")
    def _sync_dataset_id_and_periodo(self) -> "EntrenarRequest":
        """
        Sincroniza ``dataset_id`` y ``periodo_actual`` para compatibilidad.

        - Si viene ``dataset_id`` y no viene ``periodo_actual``, copia dataset_id -> periodo_actual.
        - Si viene ``periodo_actual`` y no viene ``dataset_id``, copia periodo_actual -> dataset_id.
        """
        if self.dataset_id and not self.periodo_actual:
            self.periodo_actual = self.dataset_id
        if self.periodo_actual and not self.dataset_id:
            self.dataset_id = self.periodo_actual
        return self


class EpochItem(BaseModel):
    """
    Métricas reportadas por época (para graficar en UI).

    Se recomienda que el backend reporte al menos:
    - loss (total o cls_loss)
    - recon_error (si aplica)
    - accuracy / val_accuracy
    - val_f1_macro
    - time_epoch_ms
    """

    model_config = ConfigDict(extra="ignore")

    epoch: int = Field(description="Época actual (1..N).")

    # Hacerlo opcional para robustez ante strategies que reporten solo recon_error u otros campos.
    loss: Optional[float] = Field(default=None, description="Pérdida (loss) de la época (opcional).")

    recon_error: Optional[float] = Field(default=None, description="Error de reconstrucción (opcional).")
    cls_loss: Optional[float] = Field(default=None, description="Loss de clasificación (opcional).")

    accuracy: Optional[float] = Field(default=None, description="Accuracy en train (opcional).")
    val_accuracy: Optional[float] = Field(default=None, description="Accuracy en validación (opcional).")
    val_f1_macro: Optional[float] = Field(default=None, description="F1 macro en validación (opcional).")

    grad_norm: Optional[float] = Field(default=None, description="Norma de gradiente (opcional).")
    time_epoch_ms: Optional[float] = Field(default=None, description="Tiempo por época en ms (opcional).")


class EntrenarResponse(BaseModel):
    """
    Respuesta inmediata al lanzar un entrenamiento (job async).
    """

    job_id: str
    status: Literal["queued", "running"] = Field(default="queued")
    message: str = Field(default="Entrenamiento lanzado")


class EstadoResponse(BaseModel):
    """
    Estado actual de un job de entrenamiento.

    Incluye métricas + trazas por época y metadatos de ejecución útiles para UI.
    """

    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: JobStatus = Field(description="Estado del job.")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progreso 0..1.")

    # NUEVO: para que la UI vea realmente qué modelo/cfg se ejecutó.
    model: Optional[str] = Field(default=None, description="Nombre lógico del modelo en ejecución.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parámetros/hparams efectivos del job.")

    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas globales (dict flexible).")
    history: List[EpochItem] = Field(default_factory=list, description="Historial por época.")

    run_id: Optional[str] = Field(default=None, description="run_id generado al completar (si aplica).")
    artifact_path: Optional[str] = Field(default=None, description="Ruta al directorio de artifacts del run.")
    champion_promoted: Optional[bool] = Field(default=None, description="True si el run fue promovido a champion.")
    time_total_ms: Optional[float] = Field(default=None, description="Tiempo total del job (ms).")

    error: Optional[str] = Field(default=None, description="Mensaje de error si falló.")


# ---------------------------------------------------------------------------
# Runs / Champion
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    """Resumen ligero de un run para listados."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    model_name: str
    dataset_id: Optional[str] = None
    created_at: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RunDetails(BaseModel):
    """Detalle completo de un run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    dataset_id: Optional[str] = None
    metrics: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None


class ChampionInfo(BaseModel):
    """Información del modelo champion."""

    model_config = ConfigDict(extra="ignore")

    model_name: str
    dataset_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    path: str


# ---------------------------------------------------------------------------
# Readiness / Promote (útiles para UI)
# ---------------------------------------------------------------------------

class ReadinessResponse(BaseModel):
    """Respuesta del endpoint ``GET /modelos/readiness``."""

    model_config = ConfigDict(extra="ignore")

    dataset_id: str
    labeled_exists: bool
    unified_labeled_exists: bool
    feature_pack_exists: bool
    paths: Dict[str, str] = Field(default_factory=dict)


class PromoteChampionRequest(BaseModel):
    """Request para promover un run existente a champion manualmente."""

    model_config = ConfigDict(extra="ignore")

    dataset_id: str
    run_id: str
    model_name: Optional[str] = None
