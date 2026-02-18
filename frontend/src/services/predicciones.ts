/**
 * @file predicciones.ts
 * @description Cliente de API para la pestaña **Predicciones** (score_docente).
 *
 * Este módulo es el punto único de acceso para todos los endpoints del router
 * `/predicciones` relacionados con el flujo de predicción docente–materia.
 *
 * Coexiste con `prediccion.ts` que gestiona el flujo legacy de clasificación
 * fila a fila (`/prediccion/online`, `/prediccion/batch`).
 */

import api, { qs } from "./apiClient";

// ---------------------------------------------------------------------------
// Tipos de respuesta del backend
// ---------------------------------------------------------------------------

/** Información de un dataset disponible para predicción. */
export type DatasetInfo = {
  dataset_id: string;
  n_pairs: number;
  n_docentes: number;
  n_materias: number;
  has_champion: boolean;
  created_at: string | null;
};

/** Información de un docente disponible en el dataset. */
export type TeacherInfo = {
  teacher_key: string;
  teacher_id: number;
  n_encuestas: number;
};

/** Información de una materia disponible en el dataset. */
export type MateriaInfo = {
  materia_key: string;
  materia_id: number;
  n_encuestas: number;
};

/** Un punto del radar de indicadores (una dimensión). */
export type RadarPoint = {
  indicator: string;
  actual: number;
  prediccion: number;
};

/** Un punto del bar chart comparativo (una dimensión). */
export type ComparisonPoint = {
  dimension: string;
  docente: number;
  cohorte: number;
};

/** Un punto de la serie temporal (un período). */
export type TimelinePoint = {
  semester: string;
  real: number | null;
  predicted?: number;
};

/** Estadísticas de evidencia del par. */
export type EvidenceInfo = {
  n_par: number;
  n_docente: number;
  n_materia: number;
};

/** Estadísticas históricas del par en el dataset actual. */
export type HistoricalStats = {
  mean_score: number;
  std_score: number;
};

/** Respuesta completa de predicción individual (alimenta los charts de la UI). */
export type IndividualPredictionResponse = {
  dataset_id: string;
  teacher_key: string;
  materia_key: string;

  score_total_pred: number;
  risk: "low" | "medium" | "high";
  confidence: number;
  cold_pair: boolean;

  evidence: EvidenceInfo;
  historical: HistoricalStats;

  radar: RadarPoint[];
  comparison: ComparisonPoint[];
  timeline: TimelinePoint[];

  champion_run_id: string;
  model_name: string;
};

/** Estado de un job de predicción por lote. */
export type BatchJobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;

  pred_run_id: string | null;
  dataset_id: string;
  n_pairs: number | null;
  predictions_uri: string | null;
  champion_run_id: string | null;
  error?: string;
};

/** Vista previa del parquet persistido. */
export type PredictionsPreviewResponse = {
  predictions_uri: string;
  rows: Array<Record<string, any>>;
  columns: string[];
  schema?: Record<string, any> | null;
  note: string;
};

// ---------------------------------------------------------------------------
// Funciones de API
// ---------------------------------------------------------------------------

export const listDatasets = (): Promise<DatasetInfo[]> =>
  api.get<DatasetInfo[]>("/predicciones/datasets").then((r) => r.data);

export const listTeachers = (dataset_id: string): Promise<TeacherInfo[]> =>
  api
    .get<TeacherInfo[]>(`/predicciones/teachers${qs({ dataset_id })}`)
    .then((r) => r.data);

export const listMaterias = (dataset_id: string): Promise<MateriaInfo[]> =>
  api
    .get<MateriaInfo[]>(`/predicciones/materias${qs({ dataset_id })}`)
    .then((r) => r.data);

export const predictIndividual = (payload: {
  dataset_id: string;
  teacher_key: string;
  materia_key: string;
}): Promise<IndividualPredictionResponse> =>
  api
    .post<IndividualPredictionResponse>("/predicciones/individual", payload)
    .then((r) => r.data);

export const runBatch = (dataset_id: string): Promise<BatchJobStatus> =>
  api.post<BatchJobStatus>("/predicciones/batch/run", { dataset_id }).then((r) => r.data);

export const getBatchJob = (job_id: string): Promise<BatchJobStatus> =>
  api
    .get<BatchJobStatus>(`/predicciones/batch/${encodeURIComponent(job_id)}`)
    .then((r) => r.data);

export const getOutputsPreview = (args: {
  predictions_uri: string;
  limit?: number;
  offset?: number;
}): Promise<PredictionsPreviewResponse> => {
  const { predictions_uri, limit = 50, offset = 0 } = args;
  return api
    .get<PredictionsPreviewResponse>(
      `/predicciones/outputs/preview${qs({ predictions_uri, limit, offset })}`
    )
    .then((r) => r.data);
};

/**
 * URL directa para descargar el parquet (para usar como href).
 * Construida con la misma BASE que usa apiClient.
 */
const RAW_BASE =
  (import.meta as any).env?.VITE_API_BASE ??
  (import.meta as any).env?.VITE_API_URL ??
  "http://127.0.0.1:8000";
const BASE = String(RAW_BASE).replace(/\/+$/, "");

export const getArtifactDownloadUrl = (predictions_uri: string): string =>
  `${BASE}/predicciones/outputs/file${qs({ predictions_uri })}`;
