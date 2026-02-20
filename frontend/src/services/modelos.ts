// frontend/src/services/modelos.ts
// Cliente de API para /modelos (runs, champion, entrenamiento, sweep y diagnóstico)
//
// NOTA:
// - Este archivo es un thin-client HTTP. No contiene lógica de UI.
// - La traducción DTO->UI (paridad 1:1 con el prototipo) vive en:
//   `frontend/src/features/modelos/mappers.ts`.
// - Se mantienen endpoints legacy y fallbacks para minimizar reprocesos.

import api from "./apiClient";

export type Family = "sentiment_desempeno" | "score_docente";
export type ModeloName = "rbm_general" | "rbm_restringida" | "dbm_manual";

/**
 * Extrae status HTTP desde el error enriquecido de `apiClient`.
 *
 * `apiClient` lanza Error con `err.response = { status, body, json }`.
 */
function getHttpStatus(err: unknown): number | null {
  // El cliente HTTP adjunta `response.status` en runtime; usamos `any` de forma

  // explícita para evitar acoplar este módulo a un tipo de error específico.

  const status = (err as any)?.response?.status;

  return typeof status === "number" ? status : null;
}

/**
 * Heurística: decide si un error amerita intentar un endpoint legacy.
 */
function shouldFallback(err: unknown): boolean {
  const status = getHttpStatus(err);
  const msg = (err as any)?.message ? String((err as any).message) : "";
  return status === 404 || status === 405 || (status === 422 && /modelos?/i.test(msg));
}

export type EntrenarReq = {
  modelo: ModeloName;
  data_ref?: string;

  /** Dataset/periodo activo (ej: 2025-1). Preferido: dataset_id. */
  dataset_id?: string;

  /** Alias legacy (equivalente a dataset_id). */
  periodo_actual?: string;

  /** Familia (Ruta 2). */
  family?: Family;

  /** epochs del entrenamiento */
  epochs?: number;

  /** hparams del backend: lr, batch_size, etc */
  hparams?: Record<string, number | null>;

  /** Metodología de selección */
  metodologia?: "periodo_actual" | "acumulado" | "ventana";


  /** Ventana N si metodologia=ventana */
  ventana_n?: number;

  /**
   * Warm start:
   * - none: entrenamiento desde cero
   * - champion: carga desde champion actual
   * - run_id: carga desde un run específico
   */
  warm_start_from?: "none" | "champion" | "run_id";

  /** Obligatorio si warm_start_from="run_id". */
  warm_start_run_id?: string;

  /** Si true, backend intenta preparar artifacts faltantes (cuando sea viable). */
  auto_prepare?: boolean;

  /** Split train/val (si backend lo soporta). */
  split_mode?: "temporal" | "stratified" | "random";
  val_ratio?: number;
};

export type EntrenarResp = {
  job_id: string;
  status: string;
};

export type EstadoResp = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "unknown";
  progress?: number;

  /** run_id generado (si aplica). */
  run_id?: string | null;

  /** Ruta del directorio de artifacts (debug/diagnóstico). */
  artifact_path?: string | null;

  /** True si el run fue promovido automáticamente a champion. */
  champion_promoted?: boolean;

  /** Métricas globales (dict flexible). */
  metrics?: Record<string, any>;

  /** Historial de entrenamiento (si el backend lo expone). */
  history?: { epoch: number; loss?: number; recon_error?: number; time_epoch_ms?: number }[];

  /** Sweep (cuando job_type="sweep"). */
  job_type?: "train" | "sweep";
  sweep_summary_path?: string | null;

  /** Mensaje de error si falló. */
  error?: string | null;
};

export interface RunSummary {
  /**
   * Resumen de un run de entrenamiento/auditoría.
   *
   * - run_id: carpeta dentro de artifacts/runs/<run_id>
   * - model_name: nombre lógico del modelo (ej: "rbm_general")
   * - dataset_id: dataset asociado al run (si fue registrado o inferible)
   * - created_at: ISO8601 (UTC)
   * - metrics: subset de métricas principales
   */
  run_id: string;
  model_name: string;
  dataset_id?: string | null;

  /** Contexto (Ruta 2). */
  family?: Family | null;
  task_type?: "classification" | "regression" | null;
  input_level?: "row" | "pair" | null;
  target_col?: string | null;
  data_source?: string | null;

  created_at: string;
  metrics: {
    accuracy?: number;
    f1_macro?: number;
    f1?: number;
    f1_weighted?: number;
    loss?: number;
    precision?: number;
    recall?: number;
    time_sec?: number;
    train_time_sec?: number;
    [key: string]: number | undefined;
  };
}

export interface RunDetails {
  /**
   * Detalle completo de un run.
   *
   * - metrics: contenido completo de metrics.json
   * - config: snapshot de configuración si existe (config.snapshot.yaml / config.yaml)
   * - artifact_path: ruta del directorio del run (debug)
   */
  run_id: string;
  dataset_id?: string | null;

  /** Contexto (Ruta 2). */
  family?: Family | null;
  task_type?: "classification" | "regression" | null;
  input_level?: "row" | "pair" | null;
  target_col?: string | null;
  data_source?: string | null;

  metrics: any;
  config?: any;
  artifact_path?: string;
}

export interface ChampionInfo {
  /**
   * Champion actual (modelo “ganador”) para un dataset.
   *
   * - model_name: nombre lógico del champion
   * - dataset_id: dataset asociado (si aplica)
   * - metrics: métricas registradas
   * - path: ruta del directorio champion en artifacts/champions
   */
  model_name: string;
  dataset_id?: string | null;

  /** Contexto (Ruta 2). */
  family?: Family | null;
  task_type?: "classification" | "regression" | null;
  input_level?: "row" | "pair" | null;
  target_col?: string | null;
  data_source?: string | null;

  /** run origen del champion (cuando existe). */
  source_run_id?: string | null;

  metrics: any;
  path: string;
}

/** POST /modelos/entrenar */
export async function entrenar(req: EntrenarReq) {
  const { data } = await api.post<EntrenarResp>("/modelos/entrenar", req);
  return data;
}

/** GET /modelos/estado/:jobId */
export async function estado(jobId: string) {
  const { data } = await api.get<EstadoResp>(`/modelos/estado/${jobId}`);
  return data;
}

/**
 * GET /modelos/runs
 * Soporta filtros opcionales (si backend los implementa):
 * - model_name
 * - dataset_id
 * - periodo
 */
export function listRuns(filters?: { model_name?: string; dataset_id?: string; periodo?: string; family?: string }) {
  const params = new URLSearchParams();
  if (filters?.model_name) params.set("model_name", filters.model_name);
  if (filters?.dataset_id) params.set("dataset_id", filters.dataset_id);
  if (filters?.periodo) params.set("periodo", filters.periodo);
  if (filters?.family) params.set("family", filters.family);

  const qs = params.toString();
  const url = qs ? `/modelos/runs?${qs}` : "/modelos/runs";
  return api.get<RunSummary[]>(url).then((r) => r.data);
}

/** GET /modelos/runs/:runId */
export function getRunDetails(runId: string) {
  return api.get<RunDetails>(`/modelos/runs/${runId}`).then((r) => r.data);
}

/**
 * GET /modelos/champion
 * Soporta filtro opcional model_name (y potencialmente dataset/periodo si se define).
 */
export function getChampion(filters?: { model_name?: string; dataset_id?: string; periodo?: string; family?: string }) {
  const params = new URLSearchParams();
  if (filters?.model_name) params.set("model_name", filters.model_name);
  if (filters?.dataset_id) params.set("dataset_id", filters.dataset_id);
  if (filters?.periodo) params.set("periodo", filters.periodo);
  if (filters?.family) params.set("family", filters.family);
  if (filters?.family) params.set("family", filters.family);

  const qs = params.toString();
  const url = qs ? `/modelos/champion?${qs}` : "/modelos/champion";
  return api.get<ChampionInfo>(url).then((r) => r.data);
}

export type ReadinessResp = {
  dataset_id: string;
  labeled_exists: boolean;
  unified_labeled_exists: boolean;
  feature_pack_exists: boolean;
  pair_matrix_exists?: boolean | null;

  /** Columna objetivo detectada (score_docente). */
  score_col?: string | null;

  /** Rutas de artifacts (debug). */
  paths?: Record<string, string>;
};

export type PromoteChampionReq = {
  dataset_id: string;
  run_id: string;
  model_name: ModeloName;

  /** Contexto opcional (Ruta 2). */
  family?: Family | null;
};

export type SweepCandidateResult = {
  model_name: string;
  run_id?: string | null;
  status: string;
  primary_metric_value?: number | null;
  metrics?: Record<string, any> | null;
  error?: string | null;
};

export type SweepReq = {
  dataset_id: string;
  family?: Family;
  data_source?: string;
  seed?: number;
  epochs?: number;
  auto_prepare?: boolean;

  /** Modelos a probar (default backend: 3 modelos). */
  models?: ModeloName[];

  /** Warm start. */
  warm_start_from?: "none" | "champion" | "run_id";
  warm_start_run_id?: string;

  /** Auto-promover champion. */
  auto_promote_champion?: boolean;

  /** Overrides flexibles. */
  hparams_overrides?: Record<string, Record<string, unknown>>;
  base_hparams?: Record<string, unknown>;

  /** Campos incrementales (score_docente). */
  data_plan?: string | null;
  window_k?: number | null;
  replay_size?: number | null;
  replay_strategy?: string | null;
};

export type SweepResp = {
  sweep_id: string;
  status?: string;
  dataset_id: string;
  family: string;
  primary_metric: string;
  primary_metric_mode: string;
  candidates: SweepCandidateResult[];
  best?: SweepCandidateResult | null;
  champion_promoted?: boolean;
  champion_run_id?: string | null;
  summary_path?: string | null;
  elapsed_s?: number | null;
};

export type SweepSummaryResp = {
  sweep_id: string;
  dataset_id: string;
  family: string;
  status: string;
  summary_path?: string | null;
  best_overall?: Record<string, any> | null;
  best_by_model?: Record<string, any> | null;
};

/**
 * GET /modelos/readiness
 * - Diagnóstico de disponibilidad de datasets/artifacts (labeled, feature-pack, pair-matrix).
 */
export async function getReadiness(filters?: { dataset_id?: string }) {
  const params = new URLSearchParams();
  if (filters?.dataset_id) params.set("dataset_id", filters.dataset_id);
  const qs = params.toString();
  const url = qs ? `/modelos/readiness?${qs}` : "/modelos/readiness";
  const { data } = await api.get<ReadinessResp>(url);
  return data;
}

/**
 * POST /modelos/promote (legacy) o /modelos/champion/promote (actual)
 * - Promueve un run existente a champion de forma manual.
 */
export async function promoteChampion(req: PromoteChampionReq) {
  try {
    const { data } = await api.post<ChampionInfo>("/modelos/promote", req as any);
    return data;
  } catch (err) {
    if (!shouldFallback(err)) throw err;
    const { data } = await api.post<ChampionInfo>("/modelos/champion/promote", req as any);
    return data;
  }
}

/**
 * POST /modelos/sweep (P2 Parte 5) con fallback a /modelos/entrenar/sweep (legacy).
 */
export async function sweep(req: SweepReq) {
  try {
    const { data } = await api.post<SweepResp>("/modelos/sweep", req as any);
    return data;
  } catch (err) {
    if (!shouldFallback(err)) throw err;

    // Payload legacy: espera `modelos` en vez de `models`.
    const legacy: any = { ...req };
    legacy.modelos = legacy.models ?? ["rbm_general", "rbm_restringida", "dbm_manual"];
    delete legacy.models;

    const { data } = await api.post<any>("/modelos/entrenar/sweep", legacy);
    return data as SweepResp;
  }
}

/** GET /modelos/sweeps/:sweepId */
export async function getSweepSummary(sweepId: string) {
  const { data } = await api.get<SweepSummaryResp>(`/modelos/sweeps/${encodeURIComponent(sweepId)}`);
  return data;
}
