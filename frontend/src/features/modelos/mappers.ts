// frontend/src/features/modelos/mappers.ts
// =============================================================================
// NeuroCampus — Feature Modelos: mappers (DTO backend -> tipos UI)
// =============================================================================
//
// Este módulo contiene funciones puras para transformar respuestas del backend
// (`/modelos`) en estructuras consumidas por la UI del prototipo (Modelos Tab).
//
// Principio clave:
// - La UI no debería conocer detalles del backend ni de compatibilidad legacy.
// - Cualquier "fallback" o heurística para campos faltantes vive aquí.
//
// Documentación:
// - Comentarios estilo JSDoc para facilitar documentación futura.
// - Cada mapper declara supuestos y fallbacks explícitamente.

import {
  type ChampionInfoDto,
  type Family,
  type ModeloName,
  type RunDetailsDto,
  type RunSummaryDto,
  type ReadinessResponseDto,
} from "./types";

import {
  FAMILY_CONFIGS,
  type BundleChecklist,
  type BundleStatus,
  type ChampionRecord,
  type EpochData,
  type ResolvedModel,
  type RunMetrics,
  type RunRecord,
} from "@/components/models/mockData";

/**
 * Convierte un valor desconocido en número (o retorna `null`).
 */
function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/**
 * Devuelve string seguro (o `null`).
 */
function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string" && value.trim().length > 0) return value;
  return null;
}

/**
 * Normaliza family cuando venga incompleta o legacy.
 *
 * - Si `family` está ausente: default "sentiment_desempeno".
 * - Si viene un valor desconocido: default "sentiment_desempeno".
 */
export function normalizeFamily(family: unknown): Family {
  if (family === "sentiment_desempeno" || family === "score_docente") return family;
  return "sentiment_desempeno";
}

/**
 * Determina el status del bundle con heurística:
 * - Si el backend provee `artifact_path` y el run está completed => "complete".
 * - Si el backend no provee nada => "incomplete" (conservador).
 *
 * Nota:
 * - Esto se reemplazará cuando exista `GET /modelos/runs/{run_id}/bundle`.
 */
export function computeBundleStatus(details: RunDetailsDto | null | undefined): BundleStatus {
  if (!details) return "incomplete";
  const hasArtifactPath = Boolean(details.artifact_path && details.artifact_path.length > 0);
  return hasArtifactPath ? "complete" : "incomplete";
}

/**
 * Construye un checklist de bundle (placeholder) a partir de señales mínimas.
 *
 * Importante:
 * - La UI del prototipo renderiza un checklist; hoy no existe endpoint para
 *   conocer estado real de cada artefacto.
 * - Por eso devolvemos un checklist "optimista" cuando hay artifact_path.
 */
export function computeBundleChecklist(details: RunDetailsDto | null | undefined): BundleChecklist {
  const complete = computeBundleStatus(details) === "complete";
  return {
    "predictor.json": complete,
    "metrics.json": complete,
    "job_meta.json": complete,
    "preprocess.json": complete,
    "model/": complete,
  };
}

/**
 * Extrae "primary metric value" desde un dict genérico de métricas.
 *
 * Fallback:
 * - Si no existe `primaryMetric` en el dict, intenta claves comunes
 *   ("val_f1_macro", "val_rmse", "val_accuracy", etc).
 */
export function extractPrimaryMetricValue(
  metrics: Record<string, unknown> | null | undefined,
  primaryMetric: string
): number {
  const m = metrics ?? {};
  const direct = toNumber(m[primaryMetric]);
  if (direct != null) return direct;

  // Fallbacks comunes
  const candidates = [
    "primary_metric_value",
    "val_f1_macro",
    "val_accuracy",
    "val_rmse",
    "val_mae",
    "val_r2",
  ];
  for (const k of candidates) {
    const v = toNumber(m[k]);
    if (v != null) return v;
  }
  return 0;
}

/**
 * Convierte un dict de métricas del backend a `RunMetrics` del prototipo.
 *
 * Nota:
 * - Se conserva el dict completo en la medida de lo posible.
 * - Se mapea un subconjunto esperado por la UI a campos tipados.
 */
export function mapMetricsToRunMetrics(metrics: Record<string, unknown> | null | undefined): RunMetrics {
  const m = metrics ?? {};
  const out: RunMetrics = {};

  // Campos comunes UI (si existen)
  const val_f1_macro = toNumber(m["val_f1_macro"]);
  const val_accuracy = toNumber(m["val_accuracy"]);
  const val_rmse = toNumber(m["val_rmse"]);
  const val_mae = toNumber(m["val_mae"]);
  const val_r2 = toNumber(m["val_r2"]);

  if (val_f1_macro != null) out.val_f1_macro = val_f1_macro;
  if (val_accuracy != null) out.val_accuracy = val_accuracy;
  if (val_rmse != null) out.val_rmse = val_rmse;
  if (val_mae != null) out.val_mae = val_mae;
  if (val_r2 != null) out.val_r2 = val_r2;

  // Copiar numéricos adicionales
  for (const [k, v] of Object.entries(m)) {
    const num = toNumber(v);
    if (num != null) out[k] = num;
  }

  return out;
}

/**
 * Mapea `RunSummaryDto` (listado) -> `RunRecord` (UI prototipo).
 *
 * Importante:
 * - `RunRecord` tiene MUCHOS campos; aquí poblamos lo esencial.
 * - Campos no disponibles se inicializan con defaults seguros sin romper la UI.
 *
 * Nota:
 * - Cuando exista endpoint estable de `run_details`, se completarán más campos.
 */
export function mapRunSummaryToRunRecord(summary: RunSummaryDto): RunRecord {
  const family = normalizeFamily(summary.family);
  const fc = FAMILY_CONFIGS[family];

  const metrics = mapMetricsToRunMetrics(summary.metrics);
  const primaryMetricValue = extractPrimaryMetricValue(summary.metrics, fc.primaryMetric);

  const createdAt = summary.created_at ?? new Date(0).toISOString();

  return {
    run_id: summary.run_id,
    dataset_id: summary.dataset_id ?? "unknown",
    family,
    model_name: (summary.model_name as ModeloName) ?? "rbm_general",
    task_type: summary.task_type ?? fc.taskType,
    input_level: summary.input_level ?? fc.inputLevel,
    data_source: (summary.data_source as any) ?? fc.dataSource,
    target_col: summary.target_col ?? (family === "sentiment_desempeno" ? "sentiment_label" : "score_final"),
    primary_metric: fc.primaryMetric,
    metric_mode: fc.metricMode,
    primary_metric_value: primaryMetricValue,
    metrics,

    // Estado:
    status: "completed",

    // Bundle (placeholder)
    bundle_version: "2.1.0",
    bundle_status: "incomplete",
    bundle_checklist: {
      "predictor.json": false,
      "metrics.json": false,
      "job_meta.json": false,
      "preprocess.json": false,
      "model/": false,
    },

    // Warm-start (placeholder)
    warm_started: false,
    warm_start_from: "none",
    warm_start_source_run_id: null,
    warm_start_path: null,
    warm_start_result: null,

    // Features (placeholder)
    n_feat_total: 0,
    n_feat_text: 0,
    text_feat_cols: [],

    // Training curve (placeholder)
    epochs_data: [] as EpochData[],

    // metadata
    created_at: createdAt,
    duration_seconds: 0,
    seed: 0,
    epochs: 0,
  };
}

/**
 * Enriquecer un `RunRecord` con señales del detalle del run.
 *
 * Nota:
 * - Este mapper NO cambia estructura ni estética; solo llena datos.
 */
export function mergeRunDetails(record: RunRecord, details: RunDetailsDto): RunRecord {
  const family = normalizeFamily(details.family ?? record.family);
  const fc = FAMILY_CONFIGS[family];

  const bundleStatus = computeBundleStatus(details);
  const bundleChecklist = computeBundleChecklist(details);

  return {
    ...record,
    family,
    task_type: details.task_type ?? record.task_type ?? fc.taskType,
    input_level: details.input_level ?? record.input_level ?? fc.inputLevel,
    target_col: details.target_col ?? record.target_col,
    data_source: (details.data_source as any) ?? record.data_source,
    metrics: mapMetricsToRunMetrics(details.metrics ?? record.metrics),

    primary_metric: fc.primaryMetric,
    metric_mode: fc.metricMode,
    primary_metric_value: extractPrimaryMetricValue(details.metrics, fc.primaryMetric),

    bundle_status: bundleStatus,
    bundle_checklist: bundleChecklist,
  };
}

/**
 * Mapea Champion del backend a `ResolvedModel` (UI).
 *
 * Fallbacks:
 * - `source_run_id` puede venir `null` -> usar `metrics.run_id` si existe.
 * - `model_name` puede venir ausente -> fallback "rbm_general".
 */
export function mapChampionToResolvedModel(champion: ChampionInfoDto): ResolvedModel {
  const family = normalizeFamily(champion.family);
  const fc = FAMILY_CONFIGS[family];

  const metrics = champion.metrics ?? {};
  const sourceRunId =
    champion.source_run_id ??
    toStringOrNull((metrics as any)["run_id"]) ??
    "unknown";

  const modelName = (champion.model_name as ModeloName) ?? "rbm_general";

  const pmv = extractPrimaryMetricValue(metrics as any, fc.primaryMetric);

  return {
    resolved_run_id: sourceRunId,
    source: "champion",
    bundle_status: champion.path ? "complete" : "incomplete",
    primary_metric: fc.primaryMetric,
    primary_metric_value: pmv,
    model_name: modelName,
    family,
    dataset_id: champion.dataset_id ?? "unknown",
  };
}

/**
 * Mapea Champion del backend a `ChampionRecord` (UI prototipo).
 */
export function mapChampionToChampionRecord(champion: ChampionInfoDto): ChampionRecord {
  const resolved = mapChampionToResolvedModel(champion);
  const fc = FAMILY_CONFIGS[resolved.family];

  return {
    run_id: resolved.resolved_run_id,
    model_name: resolved.model_name,
    primary_metric_value: resolved.primary_metric_value,
    primary_metric: fc.primaryMetric,
    metric_mode: fc.metricMode,
    family: resolved.family,
    dataset_id: resolved.dataset_id,
    promoted_at: new Date().toISOString(),
  };
}

/**
 * Mapea readiness del backend a un set de checks simples.
 *
 * Nota:
 * - El prototipo incluye un diagnóstico más amplio; aquí retornamos señales base.
 * - Los checks finales de `DiagnosticSubTab` se alimentarán con este readiness.
 */
export function mapReadinessToFlags(readiness: ReadinessResponseDto): {
  hasLabeled: boolean;
  hasUnifiedLabeled: boolean;
  hasFeaturePack: boolean;
  hasPairMatrix: boolean;
} {
  return {
    hasLabeled: Boolean(readiness.labeled_exists),
    hasUnifiedLabeled: Boolean(readiness.unified_labeled_exists),
    hasFeaturePack: Boolean(readiness.feature_pack_exists),
    hasPairMatrix: Boolean(readiness.pair_matrix_exists),
  };
}
