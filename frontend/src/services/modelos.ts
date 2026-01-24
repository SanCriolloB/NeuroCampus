// frontend/src/services/modelos.ts
// Cliente de API para /modelos (runs, champion, entrenamiento y estado)

import api from "./apiClient";

export type EntrenarReq = {
  modelo: "rbm_general" | "rbm_restringida";
  data_ref?: string;

  /** epochs del entrenamiento */
  epochs?: number;

  /** hparams del backend: lr, batch_size, etc */
  hparams?: Record<string, number | null>;

  /** Metodología de selección */
  metodologia?: "periodo_actual" | "acumulado" | "ventana";

  /** Periodo/dataset activo (ej: 2025-1) */
  periodo_actual?: string;

  /** Ventana N si metodologia=ventana */
  ventana_n?: number;
};

export type EntrenarResp = {
  job_id: string;
  status: string;
};

export type EstadoResp = {
  job_id: string;
  status: "running" | "completed" | "failed" | "unknown";
  metrics?: Record<string, number>;
  history: { epoch: number; loss: number; recon_error?: number; time_epoch_ms?: number }[];
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
