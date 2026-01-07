// frontend/src/types/neurocampus.ts
// Tipos compartidos del dominio NeuroCampus (frontend).
// Nota: No acoplar estos tipos a la UI (Tabs). Estos tipos son para capa Domain/Features.

export type DatasetId = string;
export type PeriodoId = string;

/** Estado del dataset (backend) */
export type DatasetStatus = "uploaded" | "processing" | "ready" | "failed" | "unknown";

/** Metadata mínima del dataset (para listado/selector) */
export interface DatasetMeta {
  dataset_id: DatasetId;
  periodo?: PeriodoId;
  nombre?: string;
  created_at?: string; // ISO
  status?: DatasetStatus;
  rows?: number;
  cols?: number;
  source?: string;
}

/** Esquema esperado de ingesta */
export interface EsquemaField {
  name: string;
  dtype?: string | null;
  required?: boolean;
  desc?: string | null;
  domain?: unknown;
  range?: unknown;
  min_len?: number | null;
  max_len?: number | null;
}

export interface EsquemaResp {
  version?: string;
  required?: string[];
  optional?: string[];
  fields?: EsquemaField[];
  examples?: Record<string, unknown>;
}

/** Validación previa */
export type ValidarIssueLevel = "info" | "warning" | "error";

export interface ValidarIssue {
  level: ValidarIssueLevel;
  code?: string;
  msg: string;
  col?: string | null;
}

export interface ValidarResp {
  ok: boolean;
  dataset_id?: DatasetId;
  missing?: string[];
  extra?: string[];
  sample?: Array<Record<string, unknown>>;
  message?: string;
  n_rows?: number;
  n_cols?: number;
  issues?: ValidarIssue[];
  stats?: Record<string, unknown>;
}

/** Resultado de upload */
export interface UploadResp {
  ok: boolean;
  dataset_id?: DatasetId;
  stored_as?: string;
  message?: string;
  status?: DatasetStatus;
}

/** Resumen del dataset */
export interface ColumnaResumen {
  name: string;
  dtype: string;
  non_nulls: number;
  sample_values: string[];
}

export interface DatasetResumen {
  dataset_id: DatasetId;
  n_rows: number;
  n_cols: number;
  periodos: string[];
  fecha_min?: string | null;
  fecha_max?: string | null;
  n_docentes?: number | null;
  n_asignaturas?: number | null;
  columns: ColumnaResumen[];
}

/** Sentimientos (BETO) */
export type SentimentLabel = "neg" | "neu" | "pos";

export interface SentimentBreakdown {
  label: SentimentLabel;
  count: number;
  proportion: number; // [0,1]
}

export interface SentimentByGroup {
  group: string;
  counts: SentimentBreakdown[];
}

export interface DatasetSentimientos {
  dataset_id: DatasetId;
  total_comentarios: number;
  global_counts: SentimentBreakdown[];
  por_docente: SentimentByGroup[];
  por_asignatura: SentimentByGroup[];
}

/** Jobs */
export type JobStatus = "created" | "running" | "done" | "failed";

export interface BetoPreprocJob {
  id: string;
  dataset: string;
  src: string;
  dst: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  meta?: Record<string, unknown> | null;
}

/** Modelos */
export interface RunSummary {
  run_id: string;
  model_name: string;
  created_at: string;
  metrics: Record<string, number | undefined>;
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

/** Predicciones */
export interface OnlineInput {
  calificaciones: Record<string, number>;
  comentario: string;
}

export interface PrediccionOnlineRequest {
  job_id?: string | null;
  family?: string;
  input: OnlineInput;
}

export interface PrediccionOnlineResponse {
  label_top: string;
  scores: Record<string, number>;
  sentiment?: string;
  confidence?: number;
  decision_rule?: Record<string, number> | null;
  latency_ms: number;
  correlation_id: string;
}

export interface PrediccionBatchResponse {
  batch_id: string;
  summary: Record<string, any>;
  sample: Array<Record<string, any>>;
  artifact: string;
  correlation_id: string;
}

/** Dashboard (placeholder tipado — se completa cuando se definan payloads reales) */
export interface DashboardKpis {
  [key: string]: number | string | null | undefined;
}
