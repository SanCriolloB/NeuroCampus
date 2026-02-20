// ============================================================
// NeuroCampus — Models Tab: Types & Mock Data
// ============================================================

// ---------- ENUMS / LITERALS ----------
export type Family = 'sentiment_desempeno' | 'score_docente';
export type ModelStrategy = 'rbm_general' | 'rbm_restringida' | 'dbm_manual';
export type TaskType = 'classification' | 'regression';
export type InputLevel = 'row' | 'pair';
export type DataSource = 'feature_pack' | 'pair_matrix';
export type MetricMode = 'max' | 'min';
export type RunStatus = 'queued' | 'running' | 'completed' | 'failed';
export type BundleStatus = 'complete' | 'incomplete';
export type WarmStartFrom = 'champion' | 'run_id' | 'none';
export type ModelResolveSource = 'champion' | 'run_id';

// ---------- INTERFACES ----------
export interface FamilyConfig {
  family: Family;
  label: string;
  taskType: TaskType;
  inputLevel: InputLevel;
  dataSource: DataSource;
  dataset: string;
  primaryMetric: string;
  metricMode: MetricMode;
  secondaryMetrics: string[];
}

export interface RunMetrics {
  val_f1_macro?: number;
  val_accuracy?: number;
  val_rmse?: number;
  val_mae?: number;
  val_r2?: number;
  [key: string]: number | undefined;
}

export interface BundleChecklist {
  'predictor.json': boolean;
  'metrics.json': boolean;
  'job_meta.json': boolean;
  'preprocess.json': boolean;
  'model/': boolean;
}

export interface EpochData {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_metric: number;
  val_metric: number;
}

export interface RunRecord {
  run_id: string;
  dataset_id: string;
  family: Family;
  model_name: ModelStrategy;
  task_type: TaskType;
  input_level: InputLevel;
  data_source: DataSource;
  target_col: string;
  primary_metric: string;
  metric_mode: MetricMode;
  primary_metric_value: number;
  metrics: RunMetrics;
  status: RunStatus;
  bundle_version: string;
  bundle_status: BundleStatus;
  bundle_checklist: BundleChecklist;
  warm_started: boolean;
  warm_start_from: WarmStartFrom;
  warm_start_source_run_id: string | null;
  warm_start_path: string | null;
  warm_start_result: 'ok' | 'skipped' | 'error' | null;
  n_feat_total: number;
  n_feat_text: number;
  text_feat_cols: string[];
  epochs_data: EpochData[];
  created_at: string;
  duration_seconds: number;
  seed: number;
  epochs: number;
  confusion_matrix?: number[][];
  residuals?: { y_true: number; y_pred: number }[];
}

export interface ChampionRecord {
  run_id: string;
  model_name: ModelStrategy;
  primary_metric_value: number;
  primary_metric: string;
  metric_mode: MetricMode;
  family: Family;
  dataset_id: string;
  promoted_at: string;
}

export interface DatasetOption {
  id: string;
  label: string;
  period: string;
  rows: number;
}

export interface ResolvedModel {
  resolved_run_id: string;
  source: ModelResolveSource;
  bundle_status: BundleStatus;
  primary_metric: string;
  primary_metric_value: number;
  model_name: ModelStrategy;
  family: Family;
  dataset_id: string;
}

export interface SweepResult {
  candidates: RunRecord[];
  winner_run_id: string;
  winner_reason: string;
  auto_promoted: boolean;
}

export interface DiagnosticCheck {
  name: string;
  status: 'pass' | 'warn' | 'fail';
  message: string;
}

// ---------- FAMILY CONFIGS ----------
export const FAMILY_CONFIGS: Record<Family, FamilyConfig> = {
  sentiment_desempeno: {
    family: 'sentiment_desempeno',
    label: 'Sentiment Desempeño',
    taskType: 'classification',
    inputLevel: 'row',
    dataSource: 'feature_pack',
    dataset: 'train_matrix.parquet',
    primaryMetric: 'val_f1_macro',
    metricMode: 'max',
    secondaryMetrics: ['val_accuracy'],
  },
  score_docente: {
    family: 'score_docente',
    label: 'Score Docente',
    taskType: 'regression',
    inputLevel: 'pair',
    dataSource: 'pair_matrix',
    dataset: 'pair_matrix.parquet',
    primaryMetric: 'val_rmse',
    metricMode: 'min',
    secondaryMetrics: ['val_mae', 'val_r2'],
  },
};

export const MODEL_STRATEGIES: { value: ModelStrategy; label: string }[] = [
  { value: 'rbm_general', label: 'RBM General' },
  { value: 'rbm_restringida', label: 'RBM Restringida' },
  { value: 'dbm_manual', label: 'DBM Manual' },
];

// ---------- DATASETS ----------
export const DATASETS: DatasetOption[] = [
  { id: 'ds_2025_1', label: 'Evaluaciones 2025-1', period: '2025-1', rows: 1320 },
  { id: 'ds_2024_2', label: 'Evaluaciones 2024-2', period: '2024-2', rows: 1250 },
  { id: 'ds_2024_1', label: 'Evaluaciones 2024-1', period: '2024-1', rows: 1180 },
  { id: 'ds_2023_2', label: 'Evaluaciones 2023-2', period: '2023-2', rows: 1100 },
  { id: 'ds_full_2024', label: 'Dataset Completo 2023-2024', period: 'Multi', rows: 4500 },
];

// ---------- HELPERS ----------
const genId = () => {
  const chars = 'abcdef0123456789';
  let id = '';
  for (let i = 0; i < 8; i++) id += chars[Math.floor(Math.random() * chars.length)];
  return id;
};

function generateEpochs(n: number, family: Family): EpochData[] {
  const data: EpochData[] = [];
  const isCls = family === 'sentiment_desempeno';
  for (let i = 1; i <= n; i++) {
    const progress = i / n;
    if (isCls) {
      data.push({
        epoch: i,
        train_loss: +(0.7 - 0.5 * progress + Math.random() * 0.02).toFixed(4),
        val_loss: +(0.75 - 0.45 * progress + Math.random() * 0.03).toFixed(4),
        train_metric: +(0.55 + 0.35 * progress + Math.random() * 0.02).toFixed(4),
        val_metric: +(0.50 + 0.33 * progress + Math.random() * 0.03).toFixed(4),
      });
    } else {
      data.push({
        epoch: i,
        train_loss: +(0.6 - 0.4 * progress + Math.random() * 0.02).toFixed(4),
        val_loss: +(0.65 - 0.38 * progress + Math.random() * 0.03).toFixed(4),
        train_metric: +(0.5 - 0.35 * progress + Math.random() * 0.02).toFixed(4),
        val_metric: +(0.55 - 0.33 * progress + Math.random() * 0.03).toFixed(4),
      });
    }
  }
  return data;
}

const TEXT_FEAT_COLS = [
  'tfidf_claridad', 'tfidf_metodologia', 'tfidf_evaluacion', 'tfidf_apoyo',
  'tfidf_recursos', 'tfidf_dinamico', 'tfidf_innovador', 'tfidf_puntual',
  'tfidf_disponible', 'tfidf_retroalimentacion', 'tfidf_excelente', 'tfidf_confuso',
];

function buildRun(
  overrides: Partial<RunRecord> & { run_id: string; model_name: ModelStrategy; family: Family; dataset_id: string }
): RunRecord {
  const fc = FAMILY_CONFIGS[overrides.family];
  const epochs = overrides.epochs ?? 10;
  const epochsData = generateEpochs(epochs, overrides.family);
  const lastEpoch = epochsData[epochsData.length - 1];

  const isCls = overrides.family === 'sentiment_desempeno';
  const pmv = overrides.primary_metric_value ?? (isCls ? lastEpoch.val_metric : lastEpoch.val_metric);

  const nText = overrides.n_feat_text ?? (Math.random() > 0.3 ? Math.floor(Math.random() * 8) + 4 : 0);

  return {
    run_id: overrides.run_id,
    dataset_id: overrides.dataset_id,
    family: overrides.family,
    model_name: overrides.model_name,
    task_type: fc.taskType,
    input_level: fc.inputLevel,
    data_source: fc.dataSource,
    target_col: isCls ? 'sentiment_label' : 'score_final',
    primary_metric: fc.primaryMetric,
    metric_mode: fc.metricMode,
    primary_metric_value: +pmv.toFixed(4),
    metrics: isCls
      ? { val_f1_macro: +pmv.toFixed(4), val_accuracy: +(pmv + 0.02 + Math.random() * 0.03).toFixed(4) }
      : { val_rmse: +pmv.toFixed(4), val_mae: +(pmv * 0.85 + Math.random() * 0.02).toFixed(4), val_r2: +(0.75 + Math.random() * 0.2).toFixed(4) },
    status: overrides.status ?? 'completed',
    bundle_version: overrides.bundle_version ?? '2.1.0',
    bundle_status: overrides.bundle_status ?? 'complete',
    bundle_checklist: overrides.bundle_checklist ?? {
      'predictor.json': true,
      'metrics.json': true,
      'job_meta.json': true,
      'preprocess.json': true,
      'model/': true,
    },
    warm_started: overrides.warm_started ?? false,
    warm_start_from: overrides.warm_start_from ?? 'none',
    warm_start_source_run_id: overrides.warm_start_source_run_id ?? null,
    warm_start_path: overrides.warm_start_path ?? null,
    warm_start_result: overrides.warm_start_result ?? null,
    n_feat_total: overrides.n_feat_total ?? (45 + nText),
    n_feat_text: nText,
    text_feat_cols: TEXT_FEAT_COLS.slice(0, nText),
    epochs_data: epochsData,
    created_at: overrides.created_at ?? '2025-02-18T14:30:00Z',
    duration_seconds: overrides.duration_seconds ?? Math.floor(120 + Math.random() * 180),
    seed: overrides.seed ?? 42,
    epochs,
    confusion_matrix: isCls ? [[145, 22], [18, 135]] : undefined,
    residuals: !isCls
      ? Array.from({ length: 50 }, () => {
          const yt = 2 + Math.random() * 3;
          return { y_true: +yt.toFixed(2), y_pred: +(yt + (Math.random() - 0.5) * 0.8).toFixed(2) };
        })
      : undefined,
  };
}

// ---------- MOCK RUNS ----------
export const MOCK_RUNS: RunRecord[] = [
  // sentiment_desempeno runs
  buildRun({
    run_id: 'run_a1b2c3d4',
    model_name: 'dbm_manual',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.8742,
    created_at: '2025-02-18T14:30:00Z',
    warm_started: true,
    warm_start_from: 'champion',
    warm_start_source_run_id: 'run_prev0001',
    warm_start_path: 'artifacts/runs/run_prev0001/model/',
    warm_start_result: 'ok',
    n_feat_text: 8,
    epochs: 15,
    seed: 42,
    duration_seconds: 245,
  }),
  buildRun({
    run_id: 'run_e5f6g7h8',
    model_name: 'rbm_general',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.8534,
    created_at: '2025-02-17T10:15:00Z',
    warm_started: false,
    warm_start_from: 'none',
    n_feat_text: 6,
    epochs: 12,
    seed: 123,
    duration_seconds: 198,
  }),
  buildRun({
    run_id: 'run_i9j0k1l2',
    model_name: 'rbm_restringida',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.8651,
    created_at: '2025-02-17T11:45:00Z',
    warm_started: true,
    warm_start_from: 'run_id',
    warm_start_source_run_id: 'run_e5f6g7h8',
    warm_start_path: 'artifacts/runs/run_e5f6g7h8/model/',
    warm_start_result: 'ok',
    n_feat_text: 6,
    epochs: 12,
    seed: 456,
    duration_seconds: 187,
  }),
  buildRun({
    run_id: 'run_m3n4o5p6',
    model_name: 'dbm_manual',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2024_2',
    primary_metric_value: 0.8398,
    created_at: '2025-01-20T09:00:00Z',
    warm_started: false,
    n_feat_text: 5,
    epochs: 10,
    seed: 42,
    duration_seconds: 168,
  }),
  buildRun({
    run_id: 'run_q7r8s9t0',
    model_name: 'rbm_general',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2024_2',
    primary_metric_value: 0.8215,
    created_at: '2025-01-19T16:30:00Z',
    status: 'completed',
    bundle_status: 'incomplete',
    bundle_checklist: {
      'predictor.json': true,
      'metrics.json': true,
      'job_meta.json': true,
      'preprocess.json': false,
      'model/': true,
    },
    n_feat_text: 0,
    epochs: 8,
    seed: 789,
    duration_seconds: 138,
    bundle_version: '1.0.0',
  }),
  // Failed run
  buildRun({
    run_id: 'run_fail0001',
    model_name: 'dbm_manual',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0,
    status: 'failed',
    created_at: '2025-02-16T08:00:00Z',
    bundle_status: 'incomplete',
    bundle_checklist: {
      'predictor.json': false,
      'metrics.json': false,
      'job_meta.json': true,
      'preprocess.json': false,
      'model/': false,
    },
    n_feat_text: 0,
    epochs: 10,
    seed: 999,
    duration_seconds: 45,
  }),

  // score_docente runs
  buildRun({
    run_id: 'run_sd_001',
    model_name: 'rbm_general',
    family: 'score_docente',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.1823,
    created_at: '2025-02-18T08:00:00Z',
    warm_started: true,
    warm_start_from: 'champion',
    warm_start_source_run_id: 'run_sd_old1',
    warm_start_path: 'artifacts/runs/run_sd_old1/model/',
    warm_start_result: 'ok',
    n_feat_text: 4,
    epochs: 20,
    seed: 42,
    duration_seconds: 320,
  }),
  buildRun({
    run_id: 'run_sd_002',
    model_name: 'dbm_manual',
    family: 'score_docente',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.1645,
    created_at: '2025-02-18T12:00:00Z',
    warm_started: false,
    n_feat_text: 7,
    epochs: 20,
    seed: 123,
    duration_seconds: 380,
  }),
  buildRun({
    run_id: 'run_sd_003',
    model_name: 'rbm_restringida',
    family: 'score_docente',
    dataset_id: 'ds_2025_1',
    primary_metric_value: 0.1756,
    created_at: '2025-02-17T15:00:00Z',
    warm_started: true,
    warm_start_from: 'run_id',
    warm_start_source_run_id: 'run_sd_001',
    warm_start_path: 'artifacts/runs/run_sd_001/model/',
    warm_start_result: 'ok',
    n_feat_text: 4,
    epochs: 15,
    seed: 456,
    duration_seconds: 265,
  }),
];

// ---------- MOCK CHAMPIONS ----------
export const MOCK_CHAMPIONS: Record<string, ChampionRecord> = {
  'sentiment_desempeno__ds_2025_1': {
    run_id: 'run_a1b2c3d4',
    model_name: 'dbm_manual',
    primary_metric_value: 0.8742,
    primary_metric: 'val_f1_macro',
    metric_mode: 'max',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2025_1',
    promoted_at: '2025-02-18T15:00:00Z',
  },
  'sentiment_desempeno__ds_2024_2': {
    run_id: 'run_m3n4o5p6',
    model_name: 'dbm_manual',
    primary_metric_value: 0.8398,
    primary_metric: 'val_f1_macro',
    metric_mode: 'max',
    family: 'sentiment_desempeno',
    dataset_id: 'ds_2024_2',
    promoted_at: '2025-01-20T10:00:00Z',
  },
  'score_docente__ds_2025_1': {
    run_id: 'run_sd_002',
    model_name: 'dbm_manual',
    primary_metric_value: 0.1645,
    primary_metric: 'val_rmse',
    metric_mode: 'min',
    family: 'score_docente',
    dataset_id: 'ds_2025_1',
    promoted_at: '2025-02-18T13:00:00Z',
  },
};

// ---------- MOCK SWEEP ----------
export function generateMockSweep(family: Family, datasetId: string): SweepResult {
  const fc = FAMILY_CONFIGS[family];
  const candidates = MODEL_STRATEGIES.map(ms => {
    const isCls = family === 'sentiment_desempeno';
    const pmv = isCls
      ? 0.82 + Math.random() * 0.07
      : 0.15 + Math.random() * 0.06;
    return buildRun({
      run_id: `sweep_${ms.value}_${genId()}`,
      model_name: ms.value,
      family,
      dataset_id: datasetId,
      primary_metric_value: pmv,
      created_at: new Date().toISOString(),
      warm_started: true,
      warm_start_from: 'champion',
      warm_start_source_run_id: MOCK_CHAMPIONS[`${family}__${datasetId}`]?.run_id ?? null,
      warm_start_result: 'ok',
    });
  });

  // Determine winner
  const sorted = [...candidates].sort((a, b) => {
    if (fc.metricMode === 'max') return b.primary_metric_value - a.primary_metric_value;
    return a.primary_metric_value - b.primary_metric_value;
  });
  const winner = sorted[0];

  return {
    candidates,
    winner_run_id: winner.run_id,
    winner_reason: `Seleccionado por ${fc.primaryMetric} (${fc.metricMode}): ${winner.primary_metric_value.toFixed(4)}`,
    auto_promoted: false,
  };
}

// ---------- DIAGNOSTICS ----------
export function generateDiagnostics(family: Family, datasetId: string): DiagnosticCheck[] {
  const champKey = `${family}__${datasetId}`;
  const champ = MOCK_CHAMPIONS[champKey];
  const checks: DiagnosticCheck[] = [];

  if (champ) {
    checks.push({ name: 'Champion exists', status: 'pass', message: `Champion ${champ.run_id} found` });
    const run = MOCK_RUNS.find(r => r.run_id === champ.run_id);
    if (run) {
      checks.push({
        name: 'Bundle completeness',
        status: run.bundle_status === 'complete' ? 'pass' : 'warn',
        message: run.bundle_status === 'complete' ? 'All bundle artifacts present' : 'Some bundle artifacts missing',
      });
      checks.push({
        name: 'Primary metric available',
        status: run.primary_metric_value > 0 ? 'pass' : 'fail',
        message: run.primary_metric_value > 0 ? `${run.primary_metric}: ${run.primary_metric_value}` : 'No primary metric value',
      });
      checks.push({
        name: 'Text features',
        status: run.n_feat_text > 0 ? 'pass' : 'warn',
        message: run.n_feat_text > 0 ? `${run.n_feat_text} text features detected` : 'No text features — TF-IDF may be missing',
      });
    }
  } else {
    checks.push({ name: 'Champion exists', status: 'fail', message: 'No champion for this dataset/family' });
  }

  // Legacy runs check
  const legacyRuns = MOCK_RUNS.filter(r => r.family === family && r.dataset_id === datasetId && r.bundle_version === '1.0.0');
  checks.push({
    name: 'Legacy runs',
    status: legacyRuns.length > 0 ? 'warn' : 'pass',
    message: legacyRuns.length > 0 ? `${legacyRuns.length} runs with legacy bundle_version` : 'No legacy runs',
  });

  checks.push({
    name: 'Contract stability',
    status: 'pass',
    message: 'predictor.json contract v2.1 verified',
  });

  return checks;
}

// ---------- PREDICTOR.JSON MOCK ----------
export const MOCK_PREDICTOR_JSON = {
  contract_version: '2.1.0',
  family: 'sentiment_desempeno',
  task_type: 'classification',
  input_level: 'row',
  target_col: 'sentiment_label',
  primary_metric: 'val_f1_macro',
  metric_mode: 'max',
  feature_columns: [
    'planificacion', 'metodologia', 'claridad', 'evaluacion', 'materiales',
    'interaccion', 'retroalimentacion', 'innovacion', 'puntualidad', 'disponibilidad',
    'tfidf_claridad', 'tfidf_metodologia', 'tfidf_evaluacion', 'tfidf_apoyo',
  ],
  model_class: 'DeepBoltzmannMachine',
  hidden_layers: [128, 64],
  created_at: '2025-02-18T14:30:00Z',
};

export const MOCK_METRICS_JSON = {
  run_id: 'run_a1b2c3d4',
  primary_metric: 'val_f1_macro',
  primary_metric_value: 0.8742,
  metrics: {
    val_f1_macro: 0.8742,
    val_accuracy: 0.8965,
    train_f1_macro: 0.9012,
    train_accuracy: 0.9234,
  },
  best_epoch: 13,
  total_epochs: 15,
};

export const MOCK_JOB_META_JSON = {
  run_id: 'run_a1b2c3d4',
  dataset_id: 'ds_2025_1',
  family: 'sentiment_desempeno',
  model_name: 'dbm_manual',
  seed: 42,
  epochs: 15,
  warm_start: true,
  warm_start_from: 'champion',
  warm_start_source_run_id: 'run_prev0001',
  auto_prepare: true,
  hparams_overrides: {},
  started_at: '2025-02-18T14:30:00Z',
  finished_at: '2025-02-18T14:34:05Z',
  duration_seconds: 245,
  status: 'completed',
};

// ---------- UTIL: Format date ----------
export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('es-ES', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}
