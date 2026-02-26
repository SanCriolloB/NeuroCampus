// ============================================================
// NeuroCampus — Runs Sub-Tab (Table + Detail Panel)
// ============================================================
import { useEffect, useMemo, useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import {
  LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import {
  Eye, Award, ExternalLink, ArrowLeft, ChevronDown, ChevronUp,
  CheckCircle2, XCircle,
} from 'lucide-react';
import { motion } from 'motion/react';
import { modelosApi } from '@/features/modelos/api';
import {
  MOCK_RUNS, DATASETS, MODEL_STRATEGIES, FAMILY_CONFIGS, formatDate, formatDuration,
  type Family, type RunRecord,
} from './mockData';
import {
  RunStatusBadge, WarmStartBadge, TextFeaturesBadge, BundleStatusBadge, CopyButton,
} from './SharedBadges';

interface RunsSubTabProps {
  family: Family;
  datasetId: string;
  extraRuns: RunRecord[];
  initialRunId?: string | null;
  onUsePredictions: (runId: string) => void;
}

export function RunsSubTab({
  family, datasetId, extraRuns, initialRunId, onUsePredictions,
}: RunsSubTabProps) {
  const fc = FAMILY_CONFIGS[family];
  const [selectedRunId, setSelectedRunId] = useState<string | null>(initialRunId ?? null);

  // ---------------------------------------------------------------------------
  // Backend integration (tolerante): list runs con fallback a mocks.
  // ---------------------------------------------------------------------------
  const [remoteRuns, setRemoteRuns] = useState<RunRecord[] | null>(null);

  useEffect(() => {
    let cancelled = false;

    // El UI usa IDs tipo "ds_2025_1"; el backend suele usar "2025-1".
    // Si el dataset no existe en DATASETS, usamos el valor tal cual.
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    (async () => {
      try {
        const runs = await modelosApi.listRunsUI({ datasetId: backendDatasetId, family });

        // Mantener paridad visual del prototipo:
        // - `dataset_id` se conserva como el ID UI seleccionado para que los filtros existentes funcionen.
        const uiRuns = runs.map(r => ({ ...r, dataset_id: datasetId }));
        if (!cancelled) setRemoteRuns(uiRuns);
      } catch {
        // Si el backend aún no está listo, dejamos `remoteRuns` en null y la UI usa MOCK_RUNS.
        if (!cancelled) setRemoteRuns(null);
      }
    })();

    return () => { cancelled = true; };
  }, [family, datasetId]);

  // Filters
  const [filterModel, setFilterModel] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterWarmStart, setFilterWarmStart] = useState<string>('all');
  const [filterText, setFilterText] = useState<string>('all');

  // All runs (mock + training extras)
  const allRuns = useMemo(() => {
    const baseRuns = remoteRuns ?? MOCK_RUNS;
    const combined = [...baseRuns, ...extraRuns];
    return combined
      .filter(r => r.family === family && r.dataset_id === datasetId)
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }, [family, datasetId, extraRuns, remoteRuns]);

  // Filtered
  const filteredRuns = useMemo(() => {
    return allRuns.filter(r => {
      if (filterModel !== 'all' && r.model_name !== filterModel) return false;
      if (filterStatus !== 'all' && r.status !== filterStatus) return false;
      if (filterWarmStart === 'yes' && !r.warm_started) return false;
      if (filterWarmStart === 'no' && r.warm_started) return false;
      if (filterText === 'yes' && r.n_feat_text === 0) return false;
      if (filterText === 'no' && r.n_feat_text > 0) return false;
      return true;
    });
  }, [allRuns, filterModel, filterStatus, filterWarmStart, filterText]);

  // ---------------------------------------------------------------------------
  // Selección + enriquecimiento: si el run viene del backend, intentamos cargar
  // el detalle para completar bundle_status, metrics, etc. Mantiene UI 1:1.
  // ---------------------------------------------------------------------------
  const selectedBaseRun = useMemo(() => {
    if (!selectedRunId) return null;
    return allRuns.find(r => r.run_id === selectedRunId) ?? null;
  }, [selectedRunId, allRuns]);

  const [resolvedRun, setResolvedRun] = useState<RunRecord | null>(null);

  useEffect(() => {
    let cancelled = false;

    // Base (siempre disponible)
    setResolvedRun(selectedBaseRun);

    if (!selectedBaseRun) return () => { cancelled = true; };

    // Solo intentamos detalle si el run proviene del backend (está en remoteRuns).
    const shouldFetchDetails = Boolean(remoteRuns?.some(r => r.run_id === selectedBaseRun.run_id));
    if (!shouldFetchDetails) return () => { cancelled = true; };

    (async () => {
      try {
        const detailed = await modelosApi.getRunDetailsUI(selectedBaseRun.run_id, selectedBaseRun);
        if (cancelled) return;

        // Paridad visual: mantener dataset_id como el ID UI.
        setResolvedRun({ ...detailed, dataset_id: datasetId });
      } catch {
        // Si falla detalle, mantenemos el baseRun para no romper la UI.
      }
    })();

    return () => { cancelled = true; };
  }, [selectedBaseRun?.run_id, remoteRuns, datasetId]);

  if (resolvedRun) {
    return (
      <RunDetail
        run={resolvedRun}
        family={family}
        fc={fc}
        onBack={() => { setSelectedRunId(null); setResolvedRun(null); }}
        onUsePredictions={onUsePredictions}
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Modelo</label>
            <Select value={filterModel} onValueChange={setFilterModel}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-8 text-xs w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="all">Todos</SelectItem>
                {MODEL_STRATEGIES.map(ms => (
                  <SelectItem key={ms.value} value={ms.value}>{ms.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Estado</label>
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-8 text-xs w-[130px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="all">Todos</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="queued">Queued</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Warm Start</label>
            <Select value={filterWarmStart} onValueChange={setFilterWarmStart}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-8 text-xs w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="all">Todos</SelectItem>
                <SelectItem value="yes">Sí</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Texto</label>
            <Select value={filterText} onValueChange={setFilterText}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-8 text-xs w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="all">Todos</SelectItem>
                <SelectItem value="yes">Sí</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <p className="text-xs text-gray-500 self-end ml-auto">{filteredRuns.length} runs</p>
        </div>
      </Card>

      {/* Runs Table */}
      <Card className="bg-[#1a1f2e] border-gray-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left text-gray-400 text-xs py-3 px-4">Fecha</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Run ID</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Modelo</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">{fc.primaryMetric}</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Estado</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Bundle</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">WS</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Texto</th>
                <th className="text-left text-gray-400 text-xs py-3 px-4">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map(run => (
                <tr
                  key={run.run_id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                >
                  <td className="py-2.5 px-4 text-gray-400 text-xs whitespace-nowrap">
                    {formatDate(run.created_at)}
                  </td>
                  <td className="py-2.5 px-4">
                    <div className="flex items-center gap-1">
                      <span className="text-cyan-400 font-mono text-xs">{run.run_id}</span>
                      <CopyButton text={run.run_id} />
                    </div>
                  </td>
                  <td className="py-2.5 px-4 text-gray-300 text-xs capitalize">
                    {run.model_name.replace(/_/g, ' ')}
                  </td>
                  <td className="py-2.5 px-4 text-white">
                    {run.status === 'completed' ? run.primary_metric_value.toFixed(4) : '—'}
                  </td>
                  <td className="py-2.5 px-4">
                    <RunStatusBadge status={run.status} />
                  </td>
                  <td className="py-2.5 px-4">
                    <BundleStatusBadge status={run.bundle_status} />
                  </td>
                  <td className="py-2.5 px-4">
                    <WarmStartBadge warmed={run.warm_started} resolved={run.warm_start_resolved ?? Boolean(run.warm_start_path)} from={run.warm_start_from} result={run.warm_start_result} reason={run.warm_start_reason} />
                  </td>
                  <td className="py-2.5 px-4">
                    <TextFeaturesBadge count={run.n_feat_text} />
                  </td>
                  <td className="py-2.5 px-4">
                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 w-7 p-0 text-gray-400 hover:text-white"
                        onClick={() => setSelectedRunId(run.run_id)}
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 w-7 p-0 text-gray-400 hover:text-cyan-400"
                        onClick={() => onUsePredictions(run.run_id)}
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {filteredRuns.length === 0 && (
                <tr>
                  <td colSpan={9} className="text-center py-8 text-gray-500 text-sm">
                    No se encontraron runs con los filtros seleccionados.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

// ============================================================
// Run Detail Panel
// ============================================================
function RunDetail({
  run, family, fc, onBack, onUsePredictions,
}: {
  run: RunRecord;
  family: Family;
  fc: (typeof FAMILY_CONFIGS)[Family];
  onBack: () => void;
  onUsePredictions: (runId: string) => void;
}) {
  const [showTextFeatures, setShowTextFeatures] = useState(false);
  const [isPromoting, setIsPromoting] = useState(false);
  const isCls = family === 'sentiment_desempeno';

  /**
   * Promueve el run actual a champion usando el backend.
   *
   * Mantiene la UI 1:1: el botón existe ya en el prototipo; aquí reemplazamos
   * el `alert()` inmediato por una llamada real y luego mostramos el mismo alert.
   */
  const handlePromote = async () => {
    if (isPromoting) return;
    setIsPromoting(true);

    try {
      // Map UI dataset id (ds_2025_1) -> backend id (2025-1) cuando aplica.
      const backendDatasetId = DATASETS.find(d => d.id === run.dataset_id)?.period ?? run.dataset_id;

      await modelosApi.promote({
        dataset_id: backendDatasetId,
        family: run.family,
        model_name: run.model_name,
        run_id: run.run_id,
      });

      alert(`Champion actualizado: ${run.run_id}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      alert(`No se pudo promover a champion: ${msg}`);
    } finally {
      setIsPromoting(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-5">
      {/* Back + header */}
      <div className="flex items-center gap-3">
        <Button
          size="sm"
          variant="ghost"
          className="text-gray-400 hover:text-white gap-1"
          onClick={onBack}
        >
          <ArrowLeft className="w-4 h-4" /> Volver
        </Button>
        <h4 className="text-white">Detalle de Run</h4>
        <RunStatusBadge status={run.status} />
      </div>

      {/* A) Identity & Contract */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Identidad y Contrato</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-xs text-gray-400">Run ID</p>
            <div className="flex items-center gap-1">
              <span className="text-cyan-400 font-mono">{run.run_id}</span>
              <CopyButton text={run.run_id} />
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-400">Dataset</p>
            <p className="text-gray-300">{run.dataset_id}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Family</p>
            <p className="text-gray-300">{run.family}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Modelo</p>
            <p className="text-gray-300 capitalize">{run.model_name.replace(/_/g, ' ')}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">task_type</p>
            <p className="text-gray-300">{run.task_type}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">input_level</p>
            <p className="text-gray-300">{run.input_level}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">data_source</p>
            <p className="text-gray-300">{run.data_source}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">target_col</p>
            <p className="text-gray-300">{run.target_col}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">bundle_version</p>
            <p className="text-gray-300">{run.bundle_version}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Seed</p>
            <p className="text-gray-300">{run.seed}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Epochs</p>
            <p className="text-gray-300">{run.epochs}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Duración</p>
            <p className="text-gray-300">{formatDuration(run.duration_seconds)}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 mt-3">
          <Button
            size="sm"
            className="bg-cyan-600 hover:bg-cyan-700 gap-1 text-xs"
            onClick={() => onUsePredictions(run.run_id)}
          >
            <ExternalLink className="w-3 h-3" /> Usar en Predictions
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="border-yellow-600 text-yellow-400 hover:bg-yellow-600/20 gap-1 text-xs"
            onClick={() => void handlePromote()}
            disabled={isPromoting}
          >
            <Award className="w-3 h-3" /> Promover a Champion
          </Button>
        </div>
      </Card>

      {/* B) Warm Start */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Warm Start</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-xs text-gray-400">warm_started</p>
            <WarmStartBadge warmed={run.warm_started} resolved={run.warm_start_resolved ?? Boolean(run.warm_start_path)} from={run.warm_start_from} result={run.warm_start_result} reason={run.warm_start_reason} />
          </div>
          <div>
            <p className="text-xs text-gray-400">warm_start_from</p>
            <p className="text-gray-300">{run.warm_start_from}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">source_run_id</p>
            <p className="text-cyan-400 font-mono text-xs">{run.warm_start_source_run_id ?? '—'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">path</p>
            <p className="text-gray-500 text-xs truncate">{run.warm_start_path ?? '—'}</p>
          </div>
        </div>
      </Card>

      {/* C) Metrics */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Métricas</h4>
        <div className="flex items-end gap-6 mb-4">
          <div>
            <p className="text-xs text-gray-400">{run.primary_metric} (principal)</p>
            <p className="text-white text-3xl">{run.primary_metric_value.toFixed(4)}</p>
            <p className="text-xs text-gray-500">{run.metric_mode}</p>
          </div>
          {Object.entries(run.metrics)
            .filter(([k]) => k !== run.primary_metric)
            .map(([k, v]) => (
              <div key={k} className="bg-[#0f1419] border border-gray-700 rounded-lg px-4 py-3">
                <p className="text-xs text-gray-400">{k}</p>
                <p className="text-white text-xl">{v?.toFixed(4)}</p>
              </div>
            ))}
        </div>
      </Card>

      {/* D) Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss curve */}
        <Card className="bg-[#1a1f2e] border-gray-800 p-5">
          <h4 className="text-white mb-3">Loss por Época</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={run.epochs_data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_loss" stroke="#3B82F6" strokeWidth={2} name="Train Loss" dot={false} />
              <Line type="monotone" dataKey="val_loss" stroke="#F59E0B" strokeWidth={2} name="Val Loss" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Metric curve */}
        <Card className="bg-[#1a1f2e] border-gray-800 p-5">
          <h4 className="text-white mb-3">{fc.primaryMetric} por Época</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={run.epochs_data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_metric" stroke="#10B981" strokeWidth={2} name="Train" dot={false} />
              <Line type="monotone" dataKey="val_metric" stroke="#06B6D4" strokeWidth={2} name="Validation" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Classification: Confusion Matrix */}
        {isCls && run.confusion_matrix && (
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <h4 className="text-white mb-3">Matriz de Confusión</h4>
            <div className="grid grid-cols-2 gap-3 max-w-[300px]">
              <div className="bg-green-500/20 border border-green-500/40 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-400">TP</p>
                <p className="text-white text-2xl">{run.confusion_matrix[0][0]}</p>
              </div>
              <div className="bg-red-500/20 border border-red-500/40 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-400">FP</p>
                <p className="text-white text-2xl">{run.confusion_matrix[0][1]}</p>
              </div>
              <div className="bg-red-500/20 border border-red-500/40 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-400">FN</p>
                <p className="text-white text-2xl">{run.confusion_matrix[1][0]}</p>
              </div>
              <div className="bg-green-500/20 border border-green-500/40 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-400">TN</p>
                <p className="text-white text-2xl">{run.confusion_matrix[1][1]}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Regression: Scatter y_true vs y_pred */}
        {!isCls && run.residuals && (
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <h4 className="text-white mb-3">y_true vs y_pred</h4>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="y_true" name="y_true" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                <YAxis dataKey="y_pred" name="y_pred" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151', fontSize: 12 }}
                  cursor={{ strokeDasharray: '3 3' }}
                />
                <Scatter data={run.residuals} fill="#06B6D4" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>

      {/* E) Features */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Features</h4>
        <div className="grid grid-cols-3 gap-4 text-sm mb-3">
          <div>
            <p className="text-xs text-gray-400">n_feat_total</p>
            <p className="text-white">{run.n_feat_total}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">n_feat_text</p>
            <p className="text-purple-400">{run.n_feat_text}</p>
          </div>
          <div>
            <TextFeaturesBadge count={run.n_feat_text} />
          </div>
        </div>
        {run.n_feat_text > 0 && (
          <>
            <button
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300 transition-colors"
              onClick={() => setShowTextFeatures(!showTextFeatures)}
            >
              {showTextFeatures ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              text_feat_cols ({run.text_feat_cols.length})
            </button>
            {showTextFeatures && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-2 flex flex-wrap gap-1.5">
                {run.text_feat_cols.map(col => (
                  <Badge key={col} className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs font-mono">
                    {col}
                  </Badge>
                ))}
              </motion.div>
            )}
          </>
        )}
      </Card>

      {/* F) Bundle / Artifacts */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Artefactos / Bundle</h4>
        <div className="flex items-center gap-3 mb-3">
          <BundleStatusBadge status={run.bundle_status} />
          <span className="text-xs text-gray-500">v{run.bundle_version}</span>
        </div>
        <div className="space-y-1.5">
          {Object.entries(run.bundle_checklist).map(([key, ok]) => (
            <div key={key} className="flex items-center gap-2 text-sm">
              {ok ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <XCircle className="w-4 h-4 text-red-400" />}
              <span className={`font-mono text-xs ${ok ? 'text-gray-300' : 'text-red-400'}`}>{key}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-3 font-mono">
          artifacts/runs/{run.run_id}/
        </p>
      </Card>
    </motion.div>
  );
}
