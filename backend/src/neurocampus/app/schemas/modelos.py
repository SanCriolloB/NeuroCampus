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

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
# ---------------------------------------------------------------------------
# Tipos comunes (enums via Literal)
# ---------------------------------------------------------------------------

ModeloName = Literal["rbm_general", "rbm_restringida", "dbm_manual"]
"""Nombre lógico del modelo.

- ``rbm_general``: RBM + head (general)
- ``rbm_restringida``: RBM variante restringida
- ``dbm_manual``: DBM (experimental / opcional)
"""

DataSource = Literal["feature_pack", "pair_matrix", "labeled", "unified_labeled"]
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
# ---------------------------------------------------------------------------
# Extensiones Ruta 2 (families)
# ---------------------------------------------------------------------------

Family = Literal["sentiment_desempeno", "score_docente"]
"""Familia de modelos.

- ``sentiment_desempeno``: clasificación (pipeline actual).
- ``score_docente``: regresión 0–50 por par docente–materia (Ruta 2).
"""

TaskType = Literal["classification", "regression"]
"""Tipo de tarea (derivable desde ``family``)."""

InputLevel = Literal["row", "pair"]
"""Nivel de entrada del modelo.

- ``row``: 1 fila = 1 registro (encuesta/comentario) (sentiment).
- ``pair``: 1 fila = 1 par (teacher_id, materia_id) (score_docente).
"""

DataPlan = Literal["dataset_only", "recent_window", "recent_window_plus_replay"]
"""Plan de datos incremental.

- ``dataset_only``: solo dataset actual.
- ``recent_window``: últimos K periodos.
- ``recent_window_plus_replay``: ventana reciente + replay histórico (muestra).
"""

ReplayStrategy = Literal["uniform", "by_period"]
"""Cómo muestrear el histórico para replay."""

WarmStartFrom = Literal["none", "champion", "run_id"]
"""Origen de warm-start."""


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
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    modelo: str = Field(
        ...,
        validation_alias=AliasChoices("modelo", "model_name", "model"),
        description=(
            "Nombre del modelo a entrenar. Backward compatible: acepta `modelo` "
            "y también `model_name`/`model` como alias para integraciones futuras."
        ),
    )

    # -----------------------------
    # Familia / tipo de tarea (Ruta 2)
    # -----------------------------
    family: Family = Field(
        default="sentiment_desempeno",
        description="Familia del entrenamiento (sentiment_desempeno | score_docente).",
    )

    # Opcionales: se derivan automáticamente desde family si se omiten.
    task_type: Optional[TaskType] = Field(
        default=None,
        description="Tipo de tarea (classification | regression). Se deriva desde family si se omite.",
    )

    input_level: Optional[InputLevel] = Field(
        default=None,
        description="Nivel de entrada (row | pair). Se deriva desde family si se omite.",
    )

    target_col: Optional[str] = Field(
        default=None,
        description=(
            "Columna target explícita (solo aplica a family=score_docente). "
            "Si se omite, el backend debe usar la indicada por pair_meta/feature meta (score_total_0_50 preferido)."
        ),
    )

    # -----------------------------
    # Config incremental (solo score_docente)
    # -----------------------------
    data_plan: Optional[DataPlan] = Field(
        default=None,
        description=(
            "Plan incremental de datos (dataset_only | recent_window | recent_window_plus_replay). "
            "Si se omite y family=score_docente, se asume recent_window_plus_replay."
        ),
    )

    window_k: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Tamaño de ventana reciente (K periodos) para score_docente. "
            "Si se omite, se infiere desde ventana_n o usa un default seguro."
        ),
    )

    replay_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Tamaño del replay histórico (n filas/pares muestreados). Solo score_docente.",
    )

    replay_strategy: ReplayStrategy = Field(
        default="uniform",
        description="Estrategia de muestreo para replay histórico (uniform | by_period).",
    )

    recency_lambda: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Factor de decaimiento por recencia (opcional; solo score_docente).",
    )

    warm_start_from: Optional[WarmStartFrom] = Field(
        default=None,
        description=(
            "Warm-start para score_docente (none | champion | run_id). "
            "Si se omite, el backend puede usar champion cuando sea compatible."
        ),
    )

    warm_start_run_id: Optional[str] = Field(
        default=None,
        description="run_id a usar como warm-start cuando warm_start_from='run_id'.",
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
        Sincroniza ``dataset_id`` y ``periodo_actual`` para compatibilidad y
        deriva defaults de Ruta 2 (families).

        - Si viene ``dataset_id`` y no viene ``periodo_actual``, copia dataset_id -> periodo_actual.
        - Si viene ``periodo_actual`` y no viene ``dataset_id``, copia periodo_actual -> dataset_id.
        - ``task_type`` e ``input_level`` se derivan desde ``family`` si se omiten.
        - Para ``score_docente``: define defaults razonables para plan incremental (window+replay+warm-start).
        """
        # Sync dataset_id <-> periodo_actual (legacy)
        if self.dataset_id and not self.periodo_actual:
            self.periodo_actual = self.dataset_id
        if self.periodo_actual and not self.dataset_id:
            self.dataset_id = self.periodo_actual

        # Derivar task_type / input_level desde family
        if self.family == "score_docente":
            if self.task_type is None:
                self.task_type = "regression"
            if self.input_level is None:
                self.input_level = "pair"

            if self.task_type != "regression":
                raise ValueError("family='score_docente' requiere task_type='regression'")
            if self.input_level != "pair":
                raise ValueError("family='score_docente' requiere input_level='pair'")

            # Defaults de plan incremental
            if self.data_plan is None:
                self.data_plan = "recent_window_plus_replay"

            # window_k: si viene metodologia='ventana' usar ventana_n; si no, usar un default seguro.
            if self.window_k is None:
                if self.metodologia == "ventana" and self.ventana_n:
                    self.window_k = int(self.ventana_n)
                else:
                    self.window_k = 4

            # replay_size: si el plan incluye replay y no viene, usar un default razonable.
            if self.data_plan == "recent_window_plus_replay":
                if self.replay_size is None:
                    self.replay_size = 5000
            else:
                if self.replay_size is None:
                    self.replay_size = 0

            # warm-start default para score_docente: champion (si es compatible)
            if self.warm_start_from is None:
                self.warm_start_from = "champion"

            if self.warm_start_from == "run_id" and not self.warm_start_run_id:
                raise ValueError("warm_start_run_id es requerido cuando warm_start_from='run_id'")

        else:
            # sentiment_desempeno (default)
            if self.task_type is None:
                self.task_type = "classification"
            if self.input_level is None:
                self.input_level = "row"

            if self.task_type != "classification":
                raise ValueError("family='sentiment_desempeno' requiere task_type='classification'")
            if self.input_level != "row":
                raise ValueError("family='sentiment_desempeno' requiere input_level='row'")

            if self.data_plan is None:
                self.data_plan = "dataset_only"
            if self.warm_start_from is None:
                self.warm_start_from = "none"
            if self.window_k is None:
                self.window_k = int(self.ventana_n) if (self.metodologia == "ventana" and self.ventana_n) else 1
            if self.replay_size is None:
                self.replay_size = 0

        # Coherencia replay
        if self.data_plan in ("recent_window_plus_replay",) and (self.replay_size or 0) <= 0:
            raise ValueError("data_plan incluye replay pero replay_size <= 0")

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

# ---------------------------------------------------------------------------
# Sweep (orquestación batch: modelo × hiperparámetros)
# ---------------------------------------------------------------------------

class SweepCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_name: ModeloName
    hparams: Dict[str, Any] = Field(default_factory=dict)

    status: JobStatus = Field(default="queued")
    child_job_id: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    score: Optional[List[Any]] = None  # serializable: [tier, score]
    error: Optional[str] = None


class SweepEntrenarRequest(BaseModel):
    """
    Lanza un sweep (barrido) entrenando múltiples modelos y múltiples hparams.

    - Usa SOLO modelos existentes (rbm_general, rbm_restringida, dbm_manual).
    - Reutiliza el mismo flujo de entrenamiento/evaluación que /modelos/entrenar.
    """
    model_config = ConfigDict(extra="ignore")

    dataset_id: str
    family: Family = "score_docente"
    task_type: Optional[TaskType] = None
    input_level: Optional[InputLevel] = None

    data_source: DataSource = "pair_matrix"
    epochs: int = 5

    # incremental (score_docente)
    data_plan: Optional[DataPlan] = None
    data_source: Optional[str] = None
    window_k: Optional[int] = None
    replay_size: Optional[int] = None
    replay_strategy: ReplayStrategy = "uniform"
    recency_lambda: Optional[float] = None

    warm_start_from: Optional[WarmStartFrom] = None
    warm_start_run_id: Optional[str] = None

    # selección
    modelos: List[ModeloName] = Field(min_length=1)

    # base hparams (se mezclan con cada combinación)
    base_hparams: Dict[str, Any] = Field(default_factory=dict)

    # grid global (aplica a todos los modelos si no hay override por modelo)
    hparams_grid: Optional[List[Dict[str, Any]]] = None

    # override por modelo: {"rbm_general":[{...},{...}], "dbm_manual":[{...}]}
    hparams_by_model: Optional[Dict[str, List[Dict[str, Any]]]] = None

    # comportamiento
    auto_promote_champion: bool = True
    max_total_runs: int = 50


class SweepEntrenarResponse(BaseModel):
    sweep_id: str
    status: Literal["queued", "running"] = "queued"
    message: str = "Sweep lanzado"


class SweepSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sweep_id: str
    dataset_id: str
    family: Family
    status: JobStatus

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    n_candidates: int = 0
    n_completed: int = 0
    n_failed: int = 0

    best_overall: Optional[SweepCandidate] = None
    best_by_model: Dict[str, SweepCandidate] = Field(default_factory=dict)

    champion_updated: Optional[bool] = None
    champion_run_id: Optional[str] = None

    summary_path: Optional[str] = None
    candidates: List[SweepCandidate] = Field(default_factory=list)


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

    # Sweep (opcional)
    job_type: Optional[Literal["train", "sweep"]] = Field(default=None)
    sweep_summary_path: Optional[str] = Field(default=None)
    sweep_best_overall: Optional[Dict[str, Any]] = Field(default=None)
    sweep_best_by_model: Optional[Dict[str, Any]] = Field(default=None)

    error: Optional[str] = Field(default=None, description="Mensaje de error si falló.")


# ---------------------------------------------------------------------------
# Runs / Champion
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    """Resumen ligero de un run para listados."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_name: str
    dataset_id: Optional[str] = None
    family: Optional[Family] = None
    task_type: Optional[TaskType] = None
    input_level: Optional[InputLevel] = None
    target_col: Optional[str] = None
    data_plan: Optional[DataPlan] = None
    data_source: Optional[str] = None
    created_at: str
    metrics: Dict[str, Any] = Field(default_factory=dict)



class RunDetails(BaseModel):
    """Detalle completo de un run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    dataset_id: Optional[str] = None
    family: Optional[Family] = None
    task_type: Optional[TaskType] = None
    input_level: Optional[InputLevel] = None
    target_col: Optional[str] = None
    data_plan: Optional[DataPlan] = None
    data_source: Optional[str] = None
    metrics: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None



class ChampionInfo(BaseModel):
    """Información del champion actual (por dataset y opcionalmente family)."""
    model_config = ConfigDict(protected_namespaces=())
    model_name: Optional[ModeloName] = None
    dataset_id: str

    # Nuevo layout
    family: Optional[Family] = None

    # Contexto del run
    task_type: Optional[TaskType] = None
    input_level: Optional[InputLevel] = None
    target_col: Optional[str] = None
    data_plan: Optional[DataPlan] = None
    data_source: Optional[DataSource] = None


    # ✅ Nuevo: run fuente del champion (debe venir de champion.json o fallback a metrics.run_id)
    source_run_id: Optional[str] = None

    metrics: Optional[Dict[str, Any]] = None
    path: str

    @model_validator(mode="after")
    def _fill_source_run_id(self):
        """
        Robustez:
        - Si champion.json no trae source_run_id (o fue creado en versiones viejas),
          lo inferimos desde metrics.run_id para no romper el contrato.
        - Esto evita nulls en API cuando el loader sí trae métricas.
        """
        if not self.source_run_id:
            m = self.metrics or {}
            if isinstance(m, dict):
                rid = m.get("run_id")
            else:
                rid = getattr(m, "run_id", None)
            if rid:
                self.source_run_id = str(rid)
        return self





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
    # Ruta 2: pair-level artifacts (opcionales para compatibilidad)
    pair_matrix_exists: Optional[bool] = Field(default=None, description="True si existe pair_matrix.parquet para el dataset.")
    score_col: Optional[str] = Field(default=None, description="Columna objetivo detectada desde meta del feature-pack/pair-meta.")
    pair_meta: Optional[Dict[str, Any]] = Field(default=None, description="Contenido (o resumen) de pair_meta.json si está disponible.")
    labeled_score_meta: Optional[Dict[str, Any]] = Field(default=None, description="Metadata del score_total (beta, delta_max, etc.) si está disponible.")
    paths: Dict[str, str] = Field(default_factory=dict)



class PromoteChampionRequest(BaseModel):
    """Request para promover un run existente a champion manualmente.

    Este request debe apuntar a un **run ya entrenado** (run_id) y opcionalmente
    permitir especificar family/task/input_level/target_col/data_plan.

    Nota: `metrics` y `path` NO pertenecen al request (son parte del response / champion.json).
    """

    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str = Field(description="Dataset/periodo al que pertenece el run (ej. '2025-1').")
    run_id: str = Field(description="ID del run a promover (carpeta en artifacts/runs/<run_id>).")
    model_name: str = Field(description="Nombre lógico del modelo (ej. 'rbm_restringida').")

    # Ruta 2 (families): opcional, pero se recomienda enviarlo para evitar ambigüedad
    family: Optional[Family] = Field(
        default=None,
        description="Familia del champion (sentiment_desempeno | score_docente). Si se omite, el backend puede inferirlo del run.",
    )
    task_type: Optional[TaskType] = Field(
        default=None,
        description="Tipo de tarea (classification | regression). Idealmente se infiere del run.",
    )
    input_level: Optional[InputLevel] = Field(
        default=None,
        description="Nivel de entrada (row | pair). Idealmente se infiere del run.",
    )
    target_col: Optional[str] = Field(
        default=None,
        description="Columna target efectiva del entrenamiento (p.ej. y_sentimiento o target_score). Idealmente se infiere del run.",
    )
    data_plan: Optional[DataPlan] = Field(
        default=None,
        description="Plan de datos usado (dataset_only | recent_window | recent_window_plus_replay). Idealmente se infiere del run.",
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        if v is None:
            raise ValueError("run_id es requerido")
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "nil"}:
            raise ValueError("run_id inválido")
        return s
