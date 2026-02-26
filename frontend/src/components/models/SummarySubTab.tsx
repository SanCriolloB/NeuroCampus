// ============================================================
// NeuroCampus — Resumen (Home) Sub-Tab
// ============================================================
import { useEffect, useMemo, useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Award, Eye, ExternalLink, CheckCircle2, XCircle, Flame, FileText } from 'lucide-react';
import { motion } from 'motion/react';
import { modelosApi } from '@/features/modelos/api';
import {
  DATASETS, MOCK_CHAMPIONS, MOCK_RUNS, FAMILY_CONFIGS, formatDate, formatDuration,
  type Family, type ChampionRecord, type RunRecord,
} from './mockData';
import { BundleStatusBadge, WarmStartBadge, TextFeaturesBadge } from './SharedBadges';

interface SummarySubTabProps {
  family: Family;
  datasetId: string;
  onNavigateToRun: (runId: string) => void;
  onUsePredictions: (runId: string) => void;
}

export function SummarySubTab({ family, datasetId, onNavigateToRun, onUsePredictions }: SummarySubTabProps) {
  const fc = FAMILY_CONFIGS[family];
  const champKey = `${family}__${datasetId}`;

  /**
   * Dataset ID usado por la UI (ej. "ds_2025_1") no siempre coincide con el backend
   * (ej. "2025-1"). Para evitar acoplar UI a backend, convertimos usando `DATASETS`.
   */
  const backendDatasetId = useMemo(
    () => DATASETS.find(d => d.id === datasetId)?.period ?? datasetId,
    [datasetId]
  );

  // ---------------------------------------------------------------------------
  // Estado UI (inicialmente mocks para mantener paridad 1:1).
  // ---------------------------------------------------------------------------
  const [champion, setChampion] = useState<ChampionRecord | undefined>(MOCK_CHAMPIONS[champKey]);
  const [champRun, setChampRun] = useState<RunRecord | undefined>(
    champion ? MOCK_RUNS.find(r => r.run_id === champion.run_id) : undefined
  );

  const [familyRuns, setFamilyRuns] = useState<RunRecord[]>(
    MOCK_RUNS
      .filter(r => r.family === family && r.dataset_id === datasetId && r.status === 'completed')
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  );

  const lastRun: RunRecord | undefined = familyRuns[0];

  // Bundle checklist for champion
  const bundleChecklist = champRun?.bundle_checklist;

  // ---------------------------------------------------------------------------
  // Intento de conectar backend: champion + runs. Si falla, se conservan mocks.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const champResp = await modelosApi.getChampionUI({
          datasetId: backendDatasetId,
          family,
        });

        if (!cancelled) {
          // Mantener paridad visual: dataset_id se conserva como el ID UI seleccionado.
          const champRecord: ChampionRecord = { ...champResp.record, dataset_id: datasetId };
          setChampion(champRecord);

          // Traer detalle del run champion para bundle/curvas, etc.
          const champRunDetails = await modelosApi.getRunDetailsUI(champRecord.run_id);
          const runUI: RunRecord = { ...champRunDetails, dataset_id: datasetId };
          setChampRun(runUI);
        }
      } catch {
        // Backend incompleto/offline: conservar mocks.
      }

      try {
        const runs = await modelosApi.listRunsUI({
          datasetId: backendDatasetId,
          family,
        });

        if (!cancelled) {
          // Normalizar datasetId a la convención de UI para filtros locales.
          const runsUI = runs.map(r => ({ ...r, dataset_id: datasetId }));
          const completed = runsUI
            .filter(r => r.family === family && r.dataset_id === datasetId && r.status === 'completed')
            .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

          setFamilyRuns(completed);
        }
      } catch {
        // Backend incompleto/offline: conservar mocks.
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [backendDatasetId, datasetId, family, champKey]);

  return (
    <div className="space-y-6">
      {/* Main cards row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Champion Card */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <Card className="bg-gradient-to-br from-blue-600/20 to-cyan-600/10 border-blue-600/40 p-5 h-full">
            <div className="flex items-center gap-2 mb-3">
              <Award className="w-5 h-5 text-yellow-400" />
              <h4 className="text-white">Champion Actual</h4>
            </div>
            {champion ? (
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-gray-400">Modelo</p>
                  <p className="text-white">{champion.model_name.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Run ID</p>
                  <p className="text-cyan-400 text-sm font-mono">{champion.run_id}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">{fc.primaryMetric}</p>
                  <p className="text-white text-2xl">{champion.primary_metric_value.toFixed(4)}</p>
                </div>
                <div className="flex gap-2 mt-2">
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
                    className="bg-cyan-600 hover:bg-cyan-700 gap-1 text-xs"
                    onClick={() => onUsePredictions(champion.run_id)}
                  >
                    <ExternalLink className="w-3 h-3" /> Predictions
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-gray-500 text-sm py-4">
                No existe champion para este dataset/family.
                <p className="mt-2 text-xs text-gray-600">Entrena un modelo y promuévelo.</p>
              </div>
            )}
          </Card>
        </motion.div>

        {/* Last Run Card */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Card className="bg-[#1a1f2e] border-gray-800 p-5 h-full">
            <h4 className="text-white mb-3">Último Run Entrenado</h4>
            {lastRun ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/40 text-xs capitalize">
                    {lastRun.model_name.replace(/_/g, ' ')}
                  </Badge>
                  <WarmStartBadge warmed={lastRun.warm_started} resolved={lastRun.warm_start_resolved ?? Boolean(lastRun.warm_start_path)} from={lastRun.warm_start_from} result={lastRun.warm_start_result} reason={lastRun.warm_start_reason} />
                </div>
                <div>
                  <p className="text-xs text-gray-400">Run ID</p>
                  <p className="text-cyan-400 text-sm font-mono">{lastRun.run_id}</p>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-xs text-gray-400">{fc.primaryMetric}</p>
                    <p className="text-white">{lastRun.primary_metric_value.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Duración</p>
                    <p className="text-white">{formatDuration(lastRun.duration_seconds)}</p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">{formatDate(lastRun.created_at)}</p>
                <Button
                  size="sm"
                  variant="outline"
                  className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-1 text-xs"
                  onClick={() => onNavigateToRun(lastRun.run_id)}
                >
                  <Eye className="w-3 h-3" /> Ver Detalle
                </Button>
              </div>
            ) : (
              <p className="text-gray-500 text-sm py-4">Sin runs para este dataset/family.</p>
            )}
          </Card>
        </motion.div>

        {/* Bundle Status Card */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <Card className="bg-[#1a1f2e] border-gray-800 p-5 h-full">
            <h4 className="text-white mb-3">Estado del Bundle</h4>
            {champRun && bundleChecklist ? (
              <div className="space-y-2">
                <BundleStatusBadge status={champRun.bundle_status} />
                <div className="mt-3 space-y-1.5">
                  {Object.entries(bundleChecklist).map(([key, ok]) => (
                    <div key={key} className="flex items-center gap-2 text-sm">
                      {ok ? (
                        <CheckCircle2 className="w-4 h-4 text-green-400" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400" />
                      )}
                      <span className={ok ? 'text-gray-300' : 'text-red-400'}>{key}</span>
                    </div>
                  ))}
                </div>
                {champRun.bundle_status === 'incomplete' && (
                  <p className="text-xs text-yellow-400 mt-2">
                    Algunos artefactos faltan. Reentrena o prepara el feature-pack.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-gray-500 text-sm py-4">Sin bundle disponible.</p>
            )}
          </Card>
        </motion.div>
      </div>

      {/* Mini charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {champRun && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card className="bg-[#1a1f2e] border-gray-800 p-5">
              <h4 className="text-white mb-3">
                {fc.primaryMetric} por Época (Champion)
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={champRun.epochs_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151', fontSize: 12 }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Line type="monotone" dataKey="val_metric" stroke="#06B6D4" strokeWidth={2} name={fc.primaryMetric} dot={false} />
                  <Line type="monotone" dataKey="train_metric" stroke="#3B82F6" strokeWidth={1.5} strokeDasharray="4 4" name="Train" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>
        )}

        {/* Quick Insights */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <h4 className="text-white mb-3">Insights Rápidos</h4>
            <div className="space-y-3">
              {/* Warm Start info */}
              <div className="flex items-center gap-3">
                <Flame className="w-4 h-4 text-orange-400" />
                <div>
                  <p className="text-sm text-gray-300">Warm Start (último run)</p>
                  <p className="text-xs text-gray-500">
                    {lastRun
                      ? lastRun.warm_started
                        ? `${lastRun.warm_start_result} — fuente: ${lastRun.warm_start_from} (${lastRun.warm_start_source_run_id})`
                        : 'No utilizado'
                      : 'N/A'}
                  </p>
                </div>
              </div>
              {/* Text Features */}
              <div className="flex items-center gap-3">
                <FileText className="w-4 h-4 text-purple-400" />
                <div>
                  <p className="text-sm text-gray-300">Features de Texto</p>
                  <p className="text-xs text-gray-500">
                    {lastRun ? `${lastRun.n_feat_text} features TF-IDF de ${lastRun.n_feat_total} totales` : 'N/A'}
                  </p>
                </div>
              </div>
              {/* Runs count */}
              <div className="flex items-center gap-3">
                <Award className="w-4 h-4 text-cyan-400" />
                <div>
                  <p className="text-sm text-gray-300">Runs totales ({family})</p>
                  <p className="text-xs text-gray-500">
                    {familyRuns.length} completados para este dataset
                  </p>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
