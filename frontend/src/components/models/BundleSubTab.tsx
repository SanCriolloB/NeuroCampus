// ============================================================
// NeuroCampus — Bundle / Artefactos Sub-Tab
// ============================================================
import { useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Search, CheckCircle2, XCircle, FileJson, Download,
} from 'lucide-react';
import { motion } from 'motion/react';
import {
  MOCK_RUNS, MOCK_CHAMPIONS, MOCK_PREDICTOR_JSON, MOCK_METRICS_JSON, MOCK_JOB_META_JSON,
  FAMILY_CONFIGS,
  type Family, type ModelResolveSource,
} from './mockData';
import { BundleStatusBadge } from './SharedBadges';

interface BundleSubTabProps {
  family: Family;
  datasetId: string;
}

export function BundleSubTab({ family, datasetId }: BundleSubTabProps) {
  const fc = FAMILY_CONFIGS[family];

  const [resolveSource, setResolveSource] = useState<ModelResolveSource>('champion');
  const [resolveRunId, setResolveRunId] = useState('');
  const [resolved, setResolved] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [activeJsonTab, setActiveJsonTab] = useState('predictor');

  const champKey = `${family}__${datasetId}`;
  const champion = MOCK_CHAMPIONS[champKey];

  const resolvedRun = resolved
    ? resolveSource === 'champion'
      ? champion
        ? MOCK_RUNS.find(r => r.run_id === champion.run_id)
        : undefined
      : MOCK_RUNS.find(r => r.run_id === resolveRunId)
    : undefined;

  const handleResolve = () => {
    setResolveError(null);
    if (resolveSource === 'champion') {
      if (!champion) {
        setResolveError('404: No existe champion para este dataset/family.');
        setResolved(false);
        return;
      }
      setResolved(true);
    } else {
      const run = MOCK_RUNS.find(r => r.run_id === resolveRunId);
      if (!run) {
        setResolveError(`404: Run "${resolveRunId}" no encontrado.`);
        setResolved(false);
        return;
      }
      if (run.bundle_status === 'incomplete') {
        setResolveError('422: Bundle incompleto — falta artefactos para inferencia completa.');
      }
      setResolved(true);
    }
  };

  // Customize JSON based on resolved family
  const predictorJson = {
    ...MOCK_PREDICTOR_JSON,
    family,
    task_type: fc.taskType,
    primary_metric: fc.primaryMetric,
    metric_mode: fc.metricMode,
  };

  const metricsJson = {
    ...MOCK_METRICS_JSON,
    primary_metric: fc.primaryMetric,
    run_id: resolvedRun?.run_id ?? MOCK_METRICS_JSON.run_id,
    primary_metric_value: resolvedRun?.primary_metric_value ?? MOCK_METRICS_JSON.primary_metric_value,
  };

  const jobMetaJson = {
    ...MOCK_JOB_META_JSON,
    family,
    run_id: resolvedRun?.run_id ?? MOCK_JOB_META_JSON.run_id,
    dataset_id: datasetId,
  };

  const preprocessJson = {
    scaler: 'StandardScaler',
    feature_columns: predictorJson.feature_columns,
    n_features: predictorJson.feature_columns.length,
    text_pipeline: {
      vectorizer: 'TfidfVectorizer',
      max_features: 500,
      ngram_range: [1, 2],
    },
    created_at: resolvedRun?.created_at ?? new Date().toISOString(),
  };

  const jsonTabs: Record<string, object> = {
    predictor: predictorJson,
    metrics: metricsJson,
    job_meta: jobMetaJson,
    preprocess: preprocessJson,
  };

  return (
    <div className="space-y-6">
      {/* Resolver */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Seleccionar Bundle</h4>
        <div className="flex items-end gap-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Fuente</label>
            <Select value={resolveSource} onValueChange={(v) => { setResolveSource(v as ModelResolveSource); setResolved(false); }}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm w-[140px]">
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
                onChange={e => { setResolveRunId(e.target.value); setResolved(false); }}
                placeholder="run_xxxxxxxx"
                className="bg-[#0f1419] border-gray-700 h-9 text-sm"
              />
            </div>
          )}
          <Button
            onClick={handleResolve}
            size="sm"
            className="bg-cyan-600 hover:bg-cyan-700 h-9 gap-1"
          >
            <Search className="w-3.5 h-3.5" /> Resolver
          </Button>
        </div>
        {resolveError && (
          <div className="mt-2 bg-red-500/10 border border-red-500/30 rounded-md px-3 py-2 text-sm text-red-400">
            {resolveError}
          </div>
        )}
      </Card>

      {/* Bundle info + JSON viewer */}
      {resolved && resolvedRun && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {/* Checklist */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-white">Artefactos del Bundle</h4>
              <BundleStatusBadge status={resolvedRun.bundle_status} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.entries(resolvedRun.bundle_checklist).map(([key, ok]) => (
                <div
                  key={key}
                  className={`border rounded-lg p-3 text-center ${ok ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'}`}
                >
                  {ok ? <CheckCircle2 className="w-5 h-5 text-green-400 mx-auto mb-1" /> : <XCircle className="w-5 h-5 text-red-400 mx-auto mb-1" />}
                  <p className={`text-xs font-mono ${ok ? 'text-green-400' : 'text-red-400'}`}>{key}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-3 font-mono">
              artifacts/runs/{resolvedRun.run_id}/
            </p>
          </Card>

          {/* JSON Viewer */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-white flex items-center gap-2">
                <FileJson className="w-4 h-4 text-cyan-400" /> Viewer JSON
              </h4>
              <Button
                size="sm"
                variant="outline"
                className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-1 text-xs"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(jsonTabs[activeJsonTab], null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `${activeJsonTab}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <Download className="w-3 h-3" /> Descargar
              </Button>
            </div>
            <Tabs value={activeJsonTab} onValueChange={setActiveJsonTab}>
              <TabsList className="bg-[#0f1419] border border-gray-700 mb-3">
                <TabsTrigger value="predictor" className="text-xs">predictor.json</TabsTrigger>
                <TabsTrigger value="metrics" className="text-xs">metrics.json</TabsTrigger>
                <TabsTrigger value="job_meta" className="text-xs">job_meta.json</TabsTrigger>
                <TabsTrigger value="preprocess" className="text-xs">preprocess.json</TabsTrigger>
              </TabsList>
              {Object.entries(jsonTabs).map(([key, json]) => (
                <TabsContent key={key} value={key}>
                  <pre className="bg-[#0f1419] border border-gray-700 rounded-lg p-4 text-xs text-gray-300 font-mono overflow-auto max-h-[400px] whitespace-pre-wrap">
                    {JSON.stringify(json, null, 2)}
                  </pre>
                </TabsContent>
              ))}
            </Tabs>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
