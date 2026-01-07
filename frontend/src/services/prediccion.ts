// frontend/src/services/prediccion.ts
// Endpoints reales del router /prediccion

import api from "./apiClient";

export type OnlineInput = {
  calificaciones: Record<string, number>;
  comentario: string;
};

export type PrediccionOnlineRequest = {
  job_id?: string | null;
  family?: string; // default backend: "sentiment_desempeno"
  input: OnlineInput;
};

export type PrediccionOnlineResponse = {
  label_top: string;
  scores: Record<string, number>;
  sentiment?: string;
  confidence?: number;
  decision_rule?: Record<string, number> | null;
  latency_ms: number;
  correlation_id: string;
};

export type PrediccionBatchResponse = {
  batch_id: string;
  summary: Record<string, any>;
  sample: Array<Record<string, any>>;
  artifact: string; // URL o path devuelto por backend
  correlation_id: string;
};

export async function online(req: PrediccionOnlineRequest) {
  const { data } = await api.post<PrediccionOnlineResponse>("/prediccion/online", req);
  return data;
}

/**
 * Batch: el backend actual recibe multipart file.
 * Si en el futuro se soporta dataset_id por query, queda listo:
 *   POST /prediccion/batch?dataset_id=...
 */
export async function batch(file: File, opts?: { dataset_id?: string }) {
  const fd = new FormData();
  fd.append("file", file);

  const qs = opts?.dataset_id ? `?dataset_id=${encodeURIComponent(opts.dataset_id)}` : "";
  const { data } = await api.post<PrediccionBatchResponse>(`/prediccion/batch${qs}`, fd);

  return data;
}
