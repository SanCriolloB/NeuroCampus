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
  text_feats?: string | null;
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
 * - dataset: nombre base en data/processed (ej: "evaluaciones_2025").
 * - text_col: columna preferida de texto o null → auto.
 * - keep_empty_text: mantener filas sin texto como neutrales.
 */
export function launchBetoPreproc(params: {
  dataset: string;
  text_col?: string | null;
  keep_empty_text?: boolean;
}) {
  const body = {
    dataset: params.dataset,
    text_col: params.text_col ?? null,
    keep_empty_text: params.keep_empty_text ?? true,
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