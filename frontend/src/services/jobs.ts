// frontend/src/services/jobs.ts
/**
 * servicios/jobs — Flujo 2: preprocesamiento BETO desde el frontend.
 *
 * Aquí centralizamos las llamadas a /jobs/preproc/beto* para:
 *  - Lanzar un job de preprocesamiento.
 *  - Consultar el estado de un job concreto.
 *  - Listar jobs recientes.
 */

import api from "./apiClient";

export type JobStatus = "created" | "running" | "done" | "failed";

export interface BetoPreprocMeta {
  model: string;
  created_at: string;
  n_rows: number;
  accepted_count: number;
  threshold: number;
  margin: number;
  neu_min: number;
  text_col: string;
  text_coverage: number;
  keep_empty_text: boolean;

  /** Opcionales (pipeline actualizado Datos). */
  text_feats?: string | null;
  text_feats_out_dir?: string | null;
  empty_text_policy?: string | null;
}

export interface BetoPreprocJob {
  id: string;
  dataset: string;
  src: string;
  dst: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  meta?: BetoPreprocMeta | null;
  error?: string | null;

  // NUEVO (opcionales, no es necesario usarlos en la UI aún)
  raw_src?: string | null;
  needs_cargar_dataset?: boolean;
}

/**
 * Lanza un job de preprocesamiento BETO.
 *
 * Compatibilidad:
 * - Si no envías `text_feats` / `empty_text_policy`, el backend mantendrá el comportamiento previo.
 *
 * Flags nuevos (pipeline Datos):
 * - text_feats: "tfidf_lsa" genera feat_t_1..feat_t_64.
 * - empty_text_policy: "zero" marca NO_TEXT (evita sesgo neutral).
 */
export function launchBetoPreproc(params: {
  dataset: string;
  text_col?: string | null;
  keep_empty_text?: boolean;
  min_tokens?: number;
  text_feats?: "none" | "tfidf_lsa" | null;
  text_feats_out_dir?: string | null;
  empty_text_policy?: "neutral" | "zero" | null;
  force_cargar_dataset?: boolean | null;
}) {
  const body = {
    dataset: params.dataset,
    text_col: params.text_col ?? null,
    keep_empty_text: params.keep_empty_text ?? true,
    min_tokens: params.min_tokens ?? 1,
    text_feats: params.text_feats ?? null,
    text_feats_out_dir: params.text_feats_out_dir ?? null,
    empty_text_policy: params.empty_text_policy ?? null,
    force_cargar_dataset: Boolean(params.force_cargar_dataset ?? false),
  };

  return api.post<BetoPreprocJob>("/jobs/preproc/beto/run", body).then((r) => r.data);
}

/** Obtiene el estado de un job BETO concreto. */
export function getBetoJob(jobId: string) {
  return api.get<BetoPreprocJob>(`/jobs/preproc/beto/${jobId}`).then((r) => r.data);
}

/** Lista jobs BETO recientes (por defecto 20). */
export function listBetoJobs(limit = 20) {
  const query = new URLSearchParams({ limit: String(limit) }).toString();
  return api.get<BetoPreprocJob[]>(`/jobs/preproc/beto?${query}`).then((r) => r.data);
}

// ---------------------------------------------------------------------------
// Jobs del dominio Datos: Unificación histórica + Feature-pack
// ---------------------------------------------------------------------------

export type DataUnifyMode = "acumulado" | "acumulado_labeled" | "periodo_actual" | "ventana";

export interface DataUnifyRequest {
  mode?: DataUnifyMode;
  ultimos?: number | null;
  desde?: string | null;
  hasta?: string | null;
}

export interface DataUnifyJob {
  id: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;

  mode: string;
  out_uri?: string | null;
  meta?: Record<string, unknown> | null;
  error?: string | null;
}

/** Lanza un job de unificación histórica (historico/*). */
export function launchDataUnify(params: DataUnifyRequest = {}) {
  const body = {
    mode: params.mode ?? "acumulado",
    ultimos: params.ultimos ?? null,
    desde: params.desde ?? null,
    hasta: params.hasta ?? null,
  };
  return api.post<DataUnifyJob>("/jobs/data/unify/run", body).then((r) => r.data);
}

/** Obtiene el estado de un job de unificación. */
export function getDataUnifyJob(jobId: string) {
  return api.get<DataUnifyJob>(`/jobs/data/unify/${jobId}`).then((r) => r.data);
}

/** Lista jobs de unificación recientes. */
export function listDataUnifyJobs(limit = 20) {
  const qs = new URLSearchParams({ limit: String(limit) }).toString();
  return api.get<DataUnifyJob[]>(`/jobs/data/unify?${qs}`).then((r) => r.data);
}

// ----------------------------- Feature-pack -------------------------------

export interface FeaturesPrepareRequest {
  dataset_id: string;
  input_uri?: string | null;
  output_dir?: string | null;
}

export interface FeaturesPrepareJob {
  id: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;

  dataset_id: string;
  input_uri?: string | null;
  output_dir?: string | null;
  artifacts?: Record<string, unknown> | null;
  error?: string | null;
}

/** Lanza un job para crear artifacts/features/<dataset_id>/train_matrix.parquet. */
export function launchFeaturesPrepare(params: FeaturesPrepareRequest) {
  const body = {
    dataset_id: params.dataset_id,
    input_uri: params.input_uri ?? null,
    output_dir: params.output_dir ?? null,
  };
  return api.post<FeaturesPrepareJob>("/jobs/data/features/prepare/run", body).then((r) => r.data);
}

/** Obtiene el estado de un job de feature-pack. */
export function getFeaturesPrepareJob(jobId: string) {
  return api.get<FeaturesPrepareJob>(`/jobs/data/features/prepare/${jobId}`).then((r) => r.data);
}

/** Lista jobs de feature-pack recientes. */
export function listFeaturesPrepareJobs(limit = 20) {
  const qs = new URLSearchParams({ limit: String(limit) }).toString();
  return api.get<FeaturesPrepareJob[]>(`/jobs/data/features/prepare?${qs}`).then((r) => r.data);
}


export interface RbmSearchJob {
  id: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  config_path: string;
  last_run_id?: string | null;
}

export function launchRbmSearch(configPath?: string) {
  const body = configPath ? { config: configPath } : {};
  return api.post<RbmSearchJob>("/jobs/training/rbm-search", body).then((r) => r.data);
}

export function getRbmSearchJob(jobId: string) {
  return api.get<RbmSearchJob>(`/jobs/training/rbm-search/${jobId}`).then((r) => r.data);
}

export function listRbmSearchJobs(limit = 20) {
  const qs = new URLSearchParams({ limit: String(limit) }).toString();
  return api.get<RbmSearchJob[]>(`/jobs/training/rbm-search?${qs}`).then((r) => r.data);
}