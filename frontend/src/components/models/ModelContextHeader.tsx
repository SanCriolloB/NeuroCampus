// ============================================================
// NeuroCampus — Global Model Context Header (always visible)
// ============================================================
import { useEffect, useMemo, useState } from 'react';
import { Card } from '../ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { Search, Zap } from 'lucide-react';
import { modelosApi } from '@/features/modelos/api';
import {
  DATASETS, FAMILY_CONFIGS, MOCK_CHAMPIONS, MOCK_RUNS,
  type Family, type ModelResolveSource, type ResolvedModel,
} from './mockData';
import { BundleStatusBadge, MetricChip } from './SharedBadges';

interface ModelContextHeaderProps {
  datasetId: string;
  family: Family;
  onDatasetChange: (id: string) => void;
  onFamilyChange: (f: Family) => void;
  resolvedModel: ResolvedModel | null;
  onResolve: (model: ResolvedModel | null) => void;
}

export function ModelContextHeader({
  datasetId, family, onDatasetChange, onFamilyChange,
  resolvedModel, onResolve,
}: ModelContextHeaderProps) {
  const [resolveSource, setResolveSource] = useState<ModelResolveSource>('champion');
  const [resolveRunId, setResolveRunId] = useState('');
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [backendDatasets, setBackendDatasets] = useState<Array<{
    dataset_id: string;
    n_rows?: number | null;
    n_pairs?: number | null;
    has_pair_matrix: boolean;
    has_train_matrix: boolean;
    has_champion_sentiment?: boolean;
    has_champion_score?: boolean;
  }> | null>(null);


  const fc = FAMILY_CONFIGS[family];

  // ---------------------------------------------------------------------------
  // Dataset listing (backend-first).
  // - Evita desalineamiento UI vs backend (ds_2025_1 vs 2025-1, etc.).
  // - Si el backend no está disponible, mantenemos el prototipo (DATASETS).
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function loadDatasets() {
      try {
        const list = await modelosApi.listDatasets();
        if (cancelled) return;

        const normalized = (list || []).map((d) => ({
          dataset_id: d.dataset_id,
          n_rows: d.n_rows ?? null,
          n_pairs: d.n_pairs ?? null,
          has_pair_matrix: Boolean(d.has_pair_matrix),
          has_train_matrix: Boolean(d.has_train_matrix),
          has_champion_sentiment: (d as any).has_champion_sentiment,
          has_champion_score: (d as any).has_champion_score,
        }));

        setBackendDatasets(normalized);

        // Si el dataset seleccionado no existe en backend, selecciona el primero disponible.
        if (normalized.length > 0) {
          const values = new Set(normalized.map((x) => x.dataset_id));
          if (!values.has(datasetId)) {
            onDatasetChange(normalized[0].dataset_id);
          }
        }
      } catch {
        // Backend offline/incompleto: nos quedamos con DATASETS (prototipo).
        if (!cancelled) setBackendDatasets(null);
      }
    }

    void loadDatasets();
    return () => {
      cancelled = true;
    };
  }, [datasetId, onDatasetChange]);

  const datasetOptions = useMemo(() => {
    if (backendDatasets && backendDatasets.length > 0) {
      return backendDatasets.map((d) => {
        const bits: string[] = [];
        if (typeof d.n_rows === 'number') bits.push(`${d.n_rows} filas`);
        if (typeof d.n_pairs === 'number') bits.push(`${d.n_pairs} pares`);
        if (d.has_train_matrix) bits.push('train_matrix');
        if (d.has_pair_matrix) bits.push('pair_matrix');
        return {
          value: d.dataset_id,
          label: d.dataset_id,
          detail: bits.length ? bits.join(' · ') : null,
        };
      });
    }

    return DATASETS.map((d) => ({
      value: d.id,
      label: `${d.label} (${d.rows} filas)`,
      detail: d.period,
    }));
  }, [backendDatasets]);

  /**
   * Resuelve el "modelo activo" para inferencia.
   *
   * Estrategia:
   * 1) Intentar contra backend (`/modelos`) usando `modelosApi` (adapter).
   * 2) Si falla (backend incompleto/offline), caer a mocks del prototipo para
   *    mantener la UI funcional y 1:1 con el diseño esperado.
   *
   * Nota:
   * - `datasetId` en UI usa IDs tipo "ds_2025_1". Para backend, convertimos a
   *   periodo con `DATASETS[].period` (ej. "2025-1") cuando está disponible.
   */
  const handleResolve = async () => {
    setResolveError(null);

    // Mapea dataset UI -> dataset backend (periodo). Si no hay match, usa el id tal cual.
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    try {
      if (resolveSource === 'champion') {
        const { resolved } = await modelosApi.getChampionUI({
          datasetId: backendDatasetId,
          family,
        });

        // Mantener paridad visual: dataset_id se conserva como el ID UI seleccionado.
        onResolve({ ...resolved, dataset_id: datasetId, family });
        return;
      }

      // resolveSource === 'run_id'
      const run = await modelosApi.getRunDetailsUI(resolveRunId);
      if (run.bundle_status === 'incomplete') {
        setResolveError('422: Bundle incompleto para inferencia.');
      }

      onResolve({
        resolved_run_id: run.run_id,
        source: 'run_id',
        bundle_status: run.bundle_status,
        primary_metric: run.primary_metric,
        primary_metric_value: run.primary_metric_value,
        model_name: run.model_name,
        family: run.family,
        dataset_id: datasetId, // paridad visual (ID UI)
      });
      return;
    } catch (err) {
      // Fallback: comportamiento 100% prototipo (mocks), para evitar bloquear la UI.
    }

    // ---------------------------------------------------------------------
    // Fallback (mocks) — exactamente como el prototipo.
    // ---------------------------------------------------------------------
    if (resolveSource === 'champion') {
      const key = `${family}__${datasetId}`;
      const champ = MOCK_CHAMPIONS[key];
      if (!champ) {
        setResolveError('404: No existe champion para este dataset/family.');
        onResolve(null);
        return;
      }
      const run = MOCK_RUNS.find(r => r.run_id === champ.run_id);
      onResolve({
        resolved_run_id: champ.run_id,
        source: 'champion',
        bundle_status: run?.bundle_status ?? 'incomplete',
        primary_metric: champ.primary_metric,
        primary_metric_value: champ.primary_metric_value,
        model_name: champ.model_name,
        family: champ.family,
        dataset_id: champ.dataset_id,
      });
    } else {
      const run = MOCK_RUNS.find(r => r.run_id === resolveRunId);
      if (!run) {
        setResolveError(`404: Run "${resolveRunId}" no encontrado.`);
        onResolve(null);
        return;
      }
      if (run.bundle_status === 'incomplete') {
        setResolveError('422: Bundle incompleto para inferencia.');
      }
      onResolve({
        resolved_run_id: run.run_id,
        source: 'run_id',
        bundle_status: run.bundle_status,
        primary_metric: run.primary_metric,
        primary_metric_value: run.primary_metric_value,
        model_name: run.model_name,
        family: run.family,
        dataset_id: run.dataset_id,
      });
    }
  };

  return (
    <Card className="bg-[#1a1f2e] border-gray-800 p-4">
      <div className="flex flex-wrap items-end gap-4">
        {/* Dataset */}
        <div className="min-w-[200px]">
          <label className="block text-xs text-gray-400 mb-1">Dataset</label>
          <Select value={datasetId} onValueChange={onDatasetChange}>
            <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1f2e] border-gray-700">
              {datasetOptions.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}{d.detail ? ` — ${d.detail}` : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Family */}
        <div className="min-w-[180px]">
          <label className="block text-xs text-gray-400 mb-1">Family</label>
          <Select value={family} onValueChange={(v) => onFamilyChange(v as Family)}>
            <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1f2e] border-gray-700">
              <SelectItem value="sentiment_desempeno">Sentiment Desempeño</SelectItem>
              <SelectItem value="score_docente">Score Docente</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Data Source (read-only derived) */}
        <div className="min-w-[130px]">
          <label className="block text-xs text-gray-400 mb-1">Data Source</label>
          <div className="bg-[#0f1419] border border-gray-700 rounded-md h-9 px-3 flex items-center text-sm text-gray-300">
            {fc.dataSource}
          </div>
        </div>

        {/* Separator */}
        <div className="w-px h-9 bg-gray-700 hidden lg:block" />

        {/* Model Resolver */}
        <div className="flex items-end gap-2 flex-1 min-w-[300px]">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Modelo activo</label>
            <Select value={resolveSource} onValueChange={(v) => setResolveSource(v as ModelResolveSource)}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm w-[130px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="champion">Champion</SelectItem>
                <SelectItem value="run_id">Run ID</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {resolveSource === 'run_id' && (
            <div className="flex-1">
              <label className="block text-xs text-gray-400 mb-1">Run ID</label>
              <Input
                value={resolveRunId}
                onChange={e => setResolveRunId(e.target.value)}
                placeholder="run_xxxxxxxx"
                className="bg-[#0f1419] border-gray-700 h-9 text-sm"
              />
            </div>
          )}
          <Button
            onClick={() => void handleResolve()}
            size="sm"
            className="bg-cyan-600 hover:bg-cyan-700 h-9 gap-1"
          >
            <Search className="w-3.5 h-3.5" />
            Resolver
          </Button>
        </div>
      </div>

      {/* Indicator chips */}
      <div className="flex flex-wrap items-center gap-2 mt-3">
        <MetricChip label="task_type" value={fc.taskType} />
        <MetricChip label="input_level" value={fc.inputLevel} />
        <MetricChip label="primary_metric" value={fc.primaryMetric} mode={fc.metricMode} />
        {resolvedModel && (
          <>
            <MetricChip label="bundle_version" value={MOCK_RUNS.find(r => r.run_id === resolvedModel.resolved_run_id)?.bundle_version ?? '—'} />
            <BundleStatusBadge status={resolvedModel.bundle_status} />
            <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/40 gap-1 text-xs">
              <Zap className="w-3 h-3" />
              {resolvedModel.source === 'champion' ? 'Champion' : 'Run'}: {resolvedModel.resolved_run_id}
            </Badge>
          </>
        )}
      </div>

      {/* Error */}
      {resolveError && (
        <div className="mt-2 bg-red-500/10 border border-red-500/30 rounded-md px-3 py-2 text-sm text-red-400">
          {resolveError}
        </div>
      )}
    </Card>
  );
}
