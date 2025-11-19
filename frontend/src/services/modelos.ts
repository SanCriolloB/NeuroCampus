// frontend/src/services/modelos.ts
// Día 4 (B) — servicio axios-like basado en apiClient para /modelos
import api from "./apiClient";

export type EntrenarReq = {
  modelo: "rbm_general" | "rbm_restringida";
  data_ref?: string;
  epochs?: number;
  hparams?: Record<string, number | null>;
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

export async function entrenar(req: EntrenarReq) {
  const { data } = await api.post("/modelos/entrenar", req);
  return data as { job_id: string; status: string };
}

export async function estado(jobId: string) {
  const { data } = await api.get(`/modelos/estado/${jobId}`);
  return data as EstadoResp;
}

// frontend/src/services/modelos.ts
import api from "./apiClient";

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
  metrics: any; // aquí puede venir histórico por época, etc.
}

export interface ChampionInfo {
  model_name: string;
  metrics: any;
  path: string;
}

export function listRuns(modelName?: string) {
  const params = new URLSearchParams();
  if (modelName) params.set("model_name", modelName);
  const qs = params.toString();
  const url = qs ? `/modelos/runs?${qs}` : "/modelos/runs";
  return api.get<RunSummary[]>(url).then((r) => r.data);
}

export function getRunDetails(runId: string) {
  return api.get<RunDetails>(`/modelos/runs/${runId}`).then((r) => r.data);
}

export function getChampion(modelName?: string) {
  const params = new URLSearchParams();
  if (modelName) params.set("model_name", modelName);
  const qs = params.toString();
  const url = qs ? `/modelos/champion?${qs}` : "/modelos/champion";
  return api.get<ChampionInfo>(url).then((r) => r.data);
}
