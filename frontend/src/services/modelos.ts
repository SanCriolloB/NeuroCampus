// frontend/src/services/modelos.ts
// Cliente de API para /modelos (runs, champion, entrenamiento y estado)

import api from "./apiClient";

export type EntrenarReq = {
  modelo: "rbm_general" | "rbm_restringida";
  data_ref?: string;
  epochs?: number;
  hparams?: Record<string, number | null>;
};

export type EntrenarResp = {
  job_id: string;
  status: string;
};

export type EstadoResp = {
  job_id: string;
  status: "running" | "completed" | "failed" | "unknown";
  metrics?: {
    accuracy?: number;
    f1_macro?: number;
    f1?: number;
  };
  history: { epoch: number; loss: number; recon_error?: number }[];
};

export interface RunSummary {
  run_id: string;
  model_name: string;
  created_at: string;
  metrics: {
    accuracy?: number;
    f1_macro?: number;
    f1_weighted?: number;
    loss?: number;
    [key: string]: number | undefined;
  };
}

export interface RunDetails {
  run_id: string;
  metrics: any;
}

export interface ChampionInfo {
  model_name: string;
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
export function listRuns(filters?: { model_name?: string; dataset_id?: string; periodo?: string }) {
  const params = new URLSearchParams();
  if (filters?.model_name) params.set("model_name", filters.model_name);
  if (filters?.dataset_id) params.set("dataset_id", filters.dataset_id);
  if (filters?.periodo) params.set("periodo", filters.periodo);

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
export function getChampion(filters?: { model_name?: string; dataset_id?: string; periodo?: string }) {
  const params = new URLSearchParams();
  if (filters?.model_name) params.set("model_name", filters.model_name);
  if (filters?.dataset_id) params.set("dataset_id", filters.dataset_id);
  if (filters?.periodo) params.set("periodo", filters.periodo);

  const qs = params.toString();
  const url = qs ? `/modelos/champion?${qs}` : "/modelos/champion";
  return api.get<ChampionInfo>(url).then((r) => r.data);
}
