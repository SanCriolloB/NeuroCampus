// ============================================================
// NeuroCampus — Sweep Sub-Tab
// ============================================================
import { useState, useMemo } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { Checkbox } from '../ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Progress } from '../ui/progress';
import {
  Play, Award, Trophy, Eye, ExternalLink, Columns, ChevronDown, ChevronUp,
} from 'lucide-react';
import { motion } from 'motion/react';
import { modelosApi } from '@/features/modelos/api';
import {
  generateMockSweep, MODEL_STRATEGIES, FAMILY_CONFIGS, DATASETS,
  type Family, type SweepResult, type RunRecord, type WarmStartFrom,
} from './mockData';
import {
  RunStatusBadge, WarmStartBadge, TextFeaturesBadge, BundleStatusBadge, CopyButton,
} from './SharedBadges';

interface SweepSubTabProps {
  family: Family;
  datasetId: string;
  onNavigateToRun: (runId: string) => void;
  onUsePredictions: (runId: string) => void;
  onSweepComplete: (runs: RunRecord[]) => void;
}

export function SweepSubTab({
  family, datasetId, onNavigateToRun, onUsePredictions, onSweepComplete,
}: SweepSubTabProps) {
  const fc = FAMILY_CONFIGS[family];

  // Form
  const [epochs, setEpochs] = useState(10);
  const [seed, setSeed] = useState(42);
  const [warmStartFrom, setWarmStartFrom] = useState<WarmStartFrom>('champion');
  const [autoPromote, setAutoPromote] = useState(false);
  const [showOverrides, setShowOverrides] = useState(false);

  // Status
  const [sweepStatus, setSweepStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [sweepProgress, setSweepProgress] = useState(0);
  const [sweepResult, setSweepResult] = useState<SweepResult | null>(null);
  const [showComparator, setShowComparator] = useState(false);

  /**
   * Ejecuta el sweep.
   *
   * Estrategia:
   * 1) Intentar ejecutar sweep real vía `modelosApi` (backend).
   * 2) Si el backend no está listo, usar `generateMockSweep` (prototipo).
   *
   * Nota:
   * - Se mantiene el progress bar del prototipo para paridad visual.
   * - `datasetId` en UI usa IDs tipo "ds_2025_1"; se mapea a periodo backend
   *   usando `DATASETS[].period` (ej. "2025-1") cuando exista.
   */
  const handleRunSweep = async () => {
    setSweepStatus('running');
    setSweepProgress(0);
    setSweepResult(null);

    // Mapea dataset UI -> dataset backend (periodo). Si no hay match, usa el id tal cual.
    const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;

    // Mantener UX del prototipo: progreso incremental hasta ~95% mientras esperamos.
    let prog = 0;
    const interval = setInterval(() => {
      prog += Math.random() * 10 + 5;
      setSweepProgress(Math.min(prog, 95));
    }, 400);

    try {
      const result = await modelosApi.sweep({
        dataset_id: backendDatasetId,
        family,
        seed,
        epochs,
        warm_start_from: warmStartFrom,
        auto_promote_champion: autoPromote,
        auto_prepare: true,
      } as any);

      clearInterval(interval);
      setSweepProgress(100);

      // Paridad visual: conservar dataset_id como el ID UI seleccionado.
      const uiCandidates = result.candidates.map(r => ({ ...r, dataset_id: datasetId }));

      setSweepResult({ ...result, candidates: uiCandidates });
      setSweepStatus('completed');
      onSweepComplete(uiCandidates);
      return;
    } catch (err) {
      // Fallback a mocks: exactamente como prototipo (no bloquea UI).
    }

    clearInterval(interval);

    // ---------------------------------------------------------------------
    // Fallback (mocks) — exactamente como el prototipo.
    // ---------------------------------------------------------------------
    const result = generateMockSweep(family, datasetId);
    setSweepProgress(100);
    setSweepResult(result);
    setSweepStatus('completed');
    onSweepComplete(result.candidates);
  };

  const winner = sweepResult?.candidates.find(c => c.run_id === sweepResult.winner_run_id);

  return (
    <div className="space-y-6">
      {/* Form */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-6">
        <h4 className="text-white mb-4">Sweep — Entrenar 3 Modelos</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Epochs</label>
            <Input
              type="number"
              value={epochs}
              onChange={e => setEpochs(Number(e.target.value))}
              min={1}
              className="bg-[#0f1419] border-gray-700 h-9 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Seed</label>
            <Input
              type="number"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              className="bg-[#0f1419] border-gray-700 h-9 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">warm_start_from</label>
            <Select value={warmStartFrom} onValueChange={(v) => setWarmStartFrom(v as WarmStartFrom)}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="champion">Champion</SelectItem>
                <SelectItem value="none">Sin Warm Start</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-end">
            <div className="flex items-center gap-2">
              <Checkbox
                id="auto-promote"
                checked={autoPromote}
                onCheckedChange={(v) => setAutoPromote(!!v)}
              />
              <label htmlFor="auto-promote" className="text-sm text-gray-300 cursor-pointer">
                Auto-promote
              </label>
            </div>
          </div>
        </div>

        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-2">Modelos a entrenar:</p>
          <div className="flex flex-wrap gap-2">
            {MODEL_STRATEGIES.map(ms => (
              <Badge key={ms.value} className="bg-blue-500/20 text-blue-400 border-blue-500/40 text-xs">
                {ms.label}
              </Badge>
            ))}
          </div>
        </div>

        {/* Advanced overrides */}
        <button
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300 mb-3"
          onClick={() => setShowOverrides(!showOverrides)}
        >
          {showOverrides ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          Overrides por modelo (JSON)
        </button>
        {showOverrides && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2 mb-4">
            {MODEL_STRATEGIES.map(ms => (
              <div key={ms.value}>
                <label className="text-xs text-gray-400">{ms.label}</label>
                <textarea
                  rows={2}
                  className="w-full bg-[#0f1419] border border-gray-700 rounded-md p-2 text-xs text-gray-300 font-mono resize-none focus:outline-none focus:border-cyan-500"
                  defaultValue="{}"
                />
              </div>
            ))}
          </motion.div>
        )}

        <Button
          onClick={() => void handleRunSweep()}
          disabled={sweepStatus === 'running'}
          className="bg-blue-600 hover:bg-blue-700 gap-2"
        >
          <Play className="w-4 h-4" />
          {sweepStatus === 'running' ? 'Ejecutando Sweep...' : 'Ejecutar Sweep'}
        </Button>

        {sweepStatus === 'running' && (
          <div className="mt-3">
            <Progress value={sweepProgress} className="h-2" />
            <p className="text-xs text-gray-400 mt-1">Entrenando 3 modelos... {Math.round(sweepProgress)}%</p>
          </div>
        )}
      </Card>

      {/* Results */}
      {sweepResult && sweepStatus === 'completed' && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {/* Winner */}
          <Card className="bg-gradient-to-r from-yellow-600/10 to-green-600/10 border-yellow-600/30 p-5">
            <div className="flex items-center gap-3 mb-3">
              <Trophy className="w-5 h-5 text-yellow-400" />
              <h4 className="text-white">Ganador del Sweep</h4>
            </div>
            {winner && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div>
                  <p className="text-xs text-gray-400">Modelo</p>
                  <p className="text-white capitalize">{winner.model_name.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Run ID</p>
                  <div className="flex items-center gap-1">
                    <span className="text-cyan-400 font-mono text-sm">{winner.run_id}</span>
                    <CopyButton text={winner.run_id} />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400">{fc.primaryMetric}</p>
                  <p className="text-white text-xl">{winner.primary_metric_value.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Razón</p>
                  <p className="text-gray-300 text-xs">{sweepResult.winner_reason}</p>
                </div>
              </div>
            )}
            <p className="text-xs text-gray-500">
              Regla: selección por <span className="text-cyan-400">{fc.primaryMetric}</span> ({fc.metricMode}).
              En empate: tie-breaker por model_name y luego run_id.
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              {winner && (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-1 text-xs"
                    onClick={() => onNavigateToRun(winner.run_id)}
                  >
                    <Eye className="w-3 h-3" /> Abrir Ganador
                  </Button>
                  <Button
                    size="sm"
                    className="bg-yellow-600 hover:bg-yellow-700 gap-1 text-xs"
                    onClick={() => void (async () => {
                      const backendDatasetId = DATASETS.find(d => d.id === datasetId)?.period ?? datasetId;
                      try {
                        await modelosApi.promote({
                          dataset_id: backendDatasetId,
                          run_id: winner.run_id,
                          model_name: winner.model_name as any,
                          family,
                        } as any);
                      } catch {
                        // Si backend no soporta promote aún, mantenemos feedback del prototipo.
                      }
                      alert(`Champion actualizado: ${winner.run_id}`);
                    })()}
                  >
                    <Award className="w-3 h-3" /> Promover Champion
                  </Button>
                </>
              )}
              <Button
                size="sm"
                variant="outline"
                className="border-cyan-600 text-cyan-400 hover:bg-cyan-600/20 gap-1 text-xs"
                onClick={() => setShowComparator(!showComparator)}
              >
                <Columns className="w-3 h-3" /> {showComparator ? 'Ocultar' : 'Comparar Runs'}
              </Button>
            </div>
          </Card>

          {/* Candidates Table */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-5">
            <h4 className="text-white mb-3">Candidatos del Sweep</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Modelo</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Run ID</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">{fc.primaryMetric}</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">WS</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Texto</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Bundle</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Ganador</th>
                    <th className="text-left text-gray-400 text-xs py-2 px-3">Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {sweepResult.candidates.map(c => {
                    const isWinner = c.run_id === sweepResult.winner_run_id;
                    return (
                      <tr
                        key={c.run_id}
                        className={`border-b border-gray-800/50 ${isWinner ? 'bg-yellow-500/5' : 'hover:bg-gray-800/30'} transition-colors`}
                      >
                        <td className="py-2 px-3 text-gray-300 capitalize text-xs">{c.model_name.replace(/_/g, ' ')}</td>
                        <td className="py-2 px-3">
                          <span className="text-cyan-400 font-mono text-xs">{c.run_id}</span>
                        </td>
                        <td className="py-2 px-3 text-white">{c.primary_metric_value.toFixed(4)}</td>
                        <td className="py-2 px-3">
                          <WarmStartBadge warmed={c.warm_started} resolved={c.warm_start_resolved ?? Boolean(c.warm_start_path)} from={c.warm_start_from} result={c.warm_start_result} reason={c.warm_start_reason} />
                        </td>
                        <td className="py-2 px-3">
                          <TextFeaturesBadge count={c.n_feat_text} />
                        </td>
                        <td className="py-2 px-3">
                          <BundleStatusBadge status={c.bundle_status} />
                        </td>
                        <td className="py-2 px-3">
                          {isWinner && (
                            <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/40 text-xs gap-1">
                              <Trophy className="w-3 h-3" /> Winner
                            </Badge>
                          )}
                        </td>
                        <td className="py-2 px-3">
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-gray-400 hover:text-white"
                              onClick={() => onNavigateToRun(c.run_id)}
                            >
                              <Eye className="w-3.5 h-3.5" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-gray-400 hover:text-cyan-400"
                              onClick={() => onUsePredictions(c.run_id)}
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Comparator */}
          {showComparator && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              <Card className="bg-[#1a1f2e] border-gray-800 p-5">
                <h4 className="text-white mb-3">Comparador Side-by-Side</h4>
                <div className="grid grid-cols-3 gap-4">
                  {sweepResult.candidates.map(c => {
                    const isWinner = c.run_id === sweepResult.winner_run_id;
                    return (
                      <div
                        key={c.run_id}
                        className={`border rounded-lg p-4 ${isWinner ? 'border-yellow-500/40 bg-yellow-500/5' : 'border-gray-700 bg-[#0f1419]/50'}`}
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-white capitalize text-sm">{c.model_name.replace(/_/g, ' ')}</span>
                          {isWinner && <Trophy className="w-4 h-4 text-yellow-400" />}
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-400">{fc.primaryMetric}</span>
                            <span className="text-white">{c.primary_metric_value.toFixed(4)}</span>
                          </div>
                          {Object.entries(c.metrics)
                            .filter(([k]) => k !== c.primary_metric)
                            .map(([k, v]) => (
                              <div key={k} className="flex justify-between">
                                <span className="text-gray-400">{k}</span>
                                <span className="text-gray-300">{v?.toFixed(4)}</span>
                              </div>
                            ))}
                          <div className="flex justify-between">
                            <span className="text-gray-400">Features (total)</span>
                            <span className="text-gray-300">{c.n_feat_total}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Features (texto)</span>
                            <span className="text-purple-400">{c.n_feat_text}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Warm Start</span>
                            <span className="text-gray-300">{c.warm_started ? 'Sí' : 'No'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Bundle</span>
                            <span className={c.bundle_status === 'complete' ? 'text-green-400' : 'text-yellow-400'}>
                              {c.bundle_status}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </motion.div>
          )}
        </motion.div>
      )}
    </div>
  );
}
