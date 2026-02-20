// ============================================================
// NeuroCampus — Champion Sub-Tab
// ============================================================
import { useEffect, useMemo, useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import {
  Award, Eye, ExternalLink, RefreshCw, AlertTriangle,
} from 'lucide-react';
import { motion } from 'motion/react';
import { modelosApi } from '@/features/modelos/api';
import {
  MOCK_CHAMPIONS, MOCK_RUNS, DATASETS, FAMILY_CONFIGS, formatDate,
  type Family, type ChampionRecord, type RunRecord,
} from './mockData';
import { BundleStatusBadge, WarmStartBadge, TextFeaturesBadge, CopyButton } from './SharedBadges';

interface ChampionSubTabProps {
  family: Family;
  datasetId: string;
  extraRuns: RunRecord[];
  onNavigateToRun: (runId: string) => void;
  onUsePredictions: (runId: string) => void;
}

export function ChampionSubTab({
  family, datasetId, extraRuns, onNavigateToRun, onUsePredictions,
}: ChampionSubTabProps) {
  const fc = FAMILY_CONFIGS[family];
  const champKey = `${family}__${datasetId}`;

  // ---------------------------------------------------------------------------
  // Backend integration (tolerante): champion + runs con fallback a mocks.
  // ---------------------------------------------------------------------------
  const [remoteChampion, setRemoteChampion] = useState<ChampionRecord | undefined>(undefined);
  const [remoteRuns, setRemoteRuns] = useState<RunRecord[] | null>(null);

  useEffect(() => {
    let cancelled = false;

    // El UI usa IDs tipo "ds_2025_1"; el backend suele usar "2025-1".
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    (async () => {
      try {
        const { record } = await modelosApi.getChampionUI({
          datasetId: backendDatasetId,
          family,
        });

        // Paridad visual: mantener dataset_id como el ID UI seleccionado.
        const uiRecord: ChampionRecord = { ...record, dataset_id: datasetId };
        if (!cancelled) setRemoteChampion(uiRecord);
      } catch {
        // Si el backend no está listo/offline, dejamos `remoteChampion` indefinido
        // y la UI cae a los mocks del prototipo.
        if (!cancelled) setRemoteChampion(undefined);
      }
    })();

    return () => { cancelled = true; };
  }, [family, datasetId]);

  useEffect(() => {
    let cancelled = false;
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    (async () => {
      try {
        const runs = await modelosApi.listRunsUI({ datasetId: backendDatasetId, family });
        const uiRuns = runs.map(r => ({ ...r, dataset_id: datasetId }));
        if (!cancelled) setRemoteRuns(uiRuns);
      } catch {
        if (!cancelled) setRemoteRuns(null);
      }
    })();

    return () => { cancelled = true; };
  }, [family, datasetId]);

  const champion = remoteChampion ?? (MOCK_CHAMPIONS[champKey] as ChampionRecord | undefined);

  // Para mostrar badges y acciones, buscamos el run del champion en la lista combinada.
  const champRun = useMemo(() => {
    if (!champion) return undefined;
    const baseRuns = remoteRuns ?? MOCK_RUNS;
    return [...baseRuns, ...extraRuns].find(r => r.run_id === champion.run_id);
  }, [champion?.run_id, remoteRuns, extraRuns]);

  // All completed runs for potential replacement
  const completedRuns = useMemo(() => {
    const baseRuns = remoteRuns ?? MOCK_RUNS;
    return [...baseRuns, ...extraRuns]
      .filter(r => r.family === family && r.dataset_id === datasetId && r.status === 'completed')
      .sort((a, b) => {
        if (fc.metricMode === 'max') return b.primary_metric_value - a.primary_metric_value;
        return a.primary_metric_value - b.primary_metric_value;
      });
  }, [family, datasetId, extraRuns, fc.metricMode, remoteRuns]);

  const [showReplace, setShowReplace] = useState(false);
  const [replaceRunId, setReplaceRunId] = useState('');

  const handleReplace = async () => {
    if (!replaceRunId) return;

    // Obtener run seleccionado para enviar contexto al backend.
    const selected = completedRuns.find(r => r.run_id === replaceRunId);

    // Mapear dataset UI -> dataset backend (periodo).
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    try {
      if (selected) {
        await modelosApi.promote({
          dataset_id: backendDatasetId,
          run_id: selected.run_id,
          model_name: selected.model_name,
          family: selected.family,
          task_type: selected.task_type,
          input_level: selected.input_level,
          target_col: selected.target_col,
          data_plan: (selected as any).data_plan ?? null,
          data_source: (selected as any).data_source ?? null,
        });

        // Actualizar champion local (paridad visual + UX inmediata).
        setRemoteChampion({
          run_id: selected.run_id,
          model_name: selected.model_name,
          primary_metric_value: selected.primary_metric_value,
          primary_metric: selected.primary_metric,
          metric_mode: selected.metric_mode,
          family: selected.family,
          dataset_id: datasetId,
          promoted_at: new Date().toISOString(),
        });
      }
    } catch {
      // Si el backend no está listo, mantenemos el comportamiento del prototipo.
    }

    alert(`Champion reemplazado por: ${replaceRunId}`);
    setShowReplace(false);
  };

  return (
    <div className="space-y-6">
      {/* Champion Card */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <Card className={`p-6 ${champion ? 'bg-gradient-to-br from-yellow-600/10 to-blue-600/10 border-yellow-600/30' : 'bg-[#1a1f2e] border-gray-800'}`}>
          <div className="flex items-center gap-3 mb-4">
            <Award className={`w-6 h-6 ${champion ? 'text-yellow-400' : 'text-gray-600'}`} />
            <h4 className="text-white">Champion para {fc.label}</h4>
            <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/40 text-xs">
              {datasetId}
            </Badge>
          </div>

          {champion && champRun ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-gray-400">Run ID</p>
                  <div className="flex items-center gap-1">
                    <span className="text-cyan-400 font-mono text-sm">{champion.run_id}</span>
                    <CopyButton text={champion.run_id} />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Modelo</p>
                  <p className="text-white capitalize">{champion.model_name.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">{fc.primaryMetric}</p>
                  <p className="text-white text-2xl">{champion.primary_metric_value.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">{fc.metricMode}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Promovido</p>
                  <p className="text-gray-300 text-sm">{formatDate(champion.promoted_at)}</p>
                </div>
              </div>

              {/* Extra info from run */}
              <div className="flex flex-wrap gap-2">
                <BundleStatusBadge status={champRun.bundle_status} />
                <WarmStartBadge warmed={champRun.warm_started} from={champRun.warm_start_from} result={champRun.warm_start_result} />
                <TextFeaturesBadge count={champRun.n_feat_text} />
              </div>

              {/* Actions */}
              <div className="flex flex-wrap gap-2 pt-2">
                <Button
                  size="sm"
                  className="bg-cyan-600 hover:bg-cyan-700 gap-1 text-xs"
                  onClick={() => onUsePredictions(champion.run_id)}
                >
                  <ExternalLink className="w-3 h-3" /> Usar Champion en Predictions
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-1 text-xs"
                  onClick={() => onNavigateToRun(champion.run_id)}
                >
                  <Eye className="w-3 h-3" /> Ver Run
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="border-yellow-600 text-yellow-400 hover:bg-yellow-600/20 gap-1 text-xs"
                  onClick={() => setShowReplace(!showReplace)}
                >
                  <RefreshCw className="w-3 h-3" /> Reemplazar Champion
                </Button>
              </div>
            </div>
          ) : (
            <div className="py-8 text-center">
              <AlertTriangle className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-500">No existe champion para este dataset/family.</p>
              <p className="text-xs text-gray-600 mt-1">Entrena un modelo y promuévelo desde la pestaña Entrenamiento o Runs.</p>
            </div>
          )}
        </Card>
      </motion.div>

      {/* Replace Champion */}
      {showReplace && completedRuns.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <h4 className="text-white mb-3">Seleccionar Nuevo Champion</h4>
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <label className="block text-xs text-gray-400 mb-1">Run</label>
                <Select value={replaceRunId} onValueChange={setReplaceRunId}>
                  <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm">
                    <SelectValue placeholder="Seleccionar run..." />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    {completedRuns.map(r => (
                      <SelectItem key={r.run_id} value={r.run_id}>
                        {r.run_id} — {r.model_name.replace(/_/g, ' ')} — {fc.primaryMetric}: {r.primary_metric_value.toFixed(4)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button
                onClick={() => void handleReplace()}
                disabled={!replaceRunId}
                className="bg-yellow-600 hover:bg-yellow-700 gap-1"
                size="sm"
              >
                <Award className="w-3 h-3" /> Confirmar
              </Button>
            </div>
          </Card>
        </motion.div>
      )}

      {/* Runs ranking */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-5">
        <h4 className="text-white mb-3">Ranking de Runs ({fc.primaryMetric}, {fc.metricMode})</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left text-gray-400 text-xs py-2 px-3">#</th>
                <th className="text-left text-gray-400 text-xs py-2 px-3">Run ID</th>
                <th className="text-left text-gray-400 text-xs py-2 px-3">Modelo</th>
                <th className="text-left text-gray-400 text-xs py-2 px-3">{fc.primaryMetric}</th>
                <th className="text-left text-gray-400 text-xs py-2 px-3">Champion</th>
                <th className="text-left text-gray-400 text-xs py-2 px-3">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {completedRuns.map((run, idx) => {
                const isChamp = champion?.run_id === run.run_id;
                return (
                  <tr
                    key={run.run_id}
                    className={`border-b border-gray-800/50 ${isChamp ? 'bg-yellow-500/5' : 'hover:bg-gray-800/30'} transition-colors`}
                  >
                    <td className="py-2 px-3 text-gray-500">{idx + 1}</td>
                    <td className="py-2 px-3">
                      <span className="text-cyan-400 font-mono text-xs">{run.run_id}</span>
                    </td>
                    <td className="py-2 px-3 text-gray-300 text-xs capitalize">{run.model_name.replace(/_/g, ' ')}</td>
                    <td className="py-2 px-3 text-white">{run.primary_metric_value.toFixed(4)}</td>
                    <td className="py-2 px-3">
                      {isChamp && (
                        <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/40 text-xs gap-1">
                          <Award className="w-3 h-3" /> Champion
                        </Badge>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 w-7 p-0 text-gray-400 hover:text-white"
                        onClick={() => onNavigateToRun(run.run_id)}
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
