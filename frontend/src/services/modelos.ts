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
  metrics: Record<string, number>;
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
