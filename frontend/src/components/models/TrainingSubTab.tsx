// ============================================================
// NeuroCampus — Entrenamiento Sub-Tab
// ============================================================
import { useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Checkbox } from '../ui/checkbox';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import {
  Play, Package, Zap, ChevronDown, ChevronUp,
  CheckCircle2, Award, Eye, ExternalLink, AlertTriangle,
} from 'lucide-react';
import { motion } from 'motion/react';
import {
  MODEL_STRATEGIES, FAMILY_CONFIGS,
  type Family, type ModelStrategy, type RunRecord, type WarmStartFrom,
} from './mockData';
import { RunStatusBadge, WarmStartBadge } from './SharedBadges';

interface TrainingSubTabProps {
  family: Family;
  datasetId: string;
  onTrainingComplete: (run: RunRecord) => void;
  onNavigateToRun: (runId: string) => void;
  onUsePredictions: (runId: string) => void;
}

export function TrainingSubTab({
  family, datasetId, onTrainingComplete, onNavigateToRun, onUsePredictions,
}: TrainingSubTabProps) {
  const fc = FAMILY_CONFIGS[family];

  // Feature-pack state
  const [featurePackStatus, setFeaturePackStatus] = useState<'idle' | 'preparing' | 'ready'>('idle');

  // Training form state
  const [modelo, setModelo] = useState<ModelStrategy>('dbm_manual');
  const [epochs, setEpochs] = useState(10);
  const [seed, setSeed] = useState(42);
  const [autoPrepare, setAutoPrepare] = useState(true);
  const [warmStart, setWarmStart] = useState(false);
  const [warmStartFrom, setWarmStartFrom] = useState<WarmStartFrom>('champion');
  const [warmStartRunId, setWarmStartRunId] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hparamsJson, setHparamsJson] = useState('{}');
  const [hparamsError, setHparamsError] = useState<string | null>(null);

  // Training state
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'queued' | 'running' | 'completed' | 'failed'>('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainedRun, setTrainedRun] = useState<RunRecord | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Validation
  const canSubmit =
    trainingStatus === 'idle' || trainingStatus === 'completed' || trainingStatus === 'failed';
  const warmStartValid = !warmStart || warmStartFrom !== 'run_id' || warmStartRunId.trim().length > 0;

  const handlePrepareFeaturePack = () => {
    setFeaturePackStatus('preparing');
    setTimeout(() => setFeaturePackStatus('ready'), 2000);
  };

  const handleTrain = () => {
    // Validate hparams JSON
    try {
      JSON.parse(hparamsJson);
      setHparamsError(null);
    } catch {
      setHparamsError('JSON inválido en hparams_overrides');
      return;
    }

    if (!warmStartValid) {
      setTrainingError('Warm start por Run ID requiere un run_id válido.');
      return;
    }

    setTrainingError(null);
    setTrainingStatus('queued');
    setTrainingProgress(0);

    setTimeout(() => {
      setTrainingStatus('running');
      let prog = 0;
      const interval = setInterval(() => {
        prog += Math.random() * 15 + 5;
        if (prog >= 100) {
          prog = 100;
          clearInterval(interval);
          // Build mock result
          const isCls = family === 'sentiment_desempeno';
          const pmv = isCls ? 0.84 + Math.random() * 0.05 : 0.15 + Math.random() * 0.05;
          const runId = `run_${Date.now().toString(36)}`;

          const newRun: RunRecord = {
            run_id: runId,
            dataset_id: datasetId,
            family,
            model_name: modelo,
            task_type: fc.taskType,
            input_level: fc.inputLevel,
            data_source: fc.dataSource,
            target_col: isCls ? 'sentiment_label' : 'score_final',
            primary_metric: fc.primaryMetric,
            metric_mode: fc.metricMode,
            primary_metric_value: +pmv.toFixed(4),
            metrics: isCls
              ? { val_f1_macro: +pmv.toFixed(4), val_accuracy: +(pmv + 0.02).toFixed(4) }
              : { val_rmse: +pmv.toFixed(4), val_mae: +(pmv * 0.85).toFixed(4), val_r2: +(0.80 + Math.random() * 0.15).toFixed(4) },
            status: 'completed',
            bundle_version: '2.1.0',
            bundle_status: 'complete',
            bundle_checklist: {
              'predictor.json': true,
              'metrics.json': true,
              'job_meta.json': true,
              'preprocess.json': true,
              'model/': true,
            },
            warm_started: warmStart,
            warm_start_from: warmStart ? warmStartFrom : 'none',
            warm_start_source_run_id: warmStart && warmStartFrom === 'run_id' ? warmStartRunId : null,
            warm_start_path: warmStart ? `artifacts/runs/${warmStartRunId || 'champion'}/model/` : null,
            warm_start_result: warmStart ? 'ok' : null,
            n_feat_total: 52,
            n_feat_text: 7,
            text_feat_cols: ['tfidf_claridad', 'tfidf_metodologia', 'tfidf_evaluacion', 'tfidf_apoyo', 'tfidf_recursos', 'tfidf_dinamico', 'tfidf_innovador'],
            epochs_data: Array.from({ length: epochs }, (_, i) => ({
              epoch: i + 1,
              train_loss: +(0.7 - 0.5 * ((i + 1) / epochs)).toFixed(4),
              val_loss: +(0.75 - 0.45 * ((i + 1) / epochs)).toFixed(4),
              train_metric: +(0.55 + 0.35 * ((i + 1) / epochs)).toFixed(4),
              val_metric: +(0.50 + 0.33 * ((i + 1) / epochs) + Math.random() * 0.02).toFixed(4),
            })),
            created_at: new Date().toISOString(),
            duration_seconds: Math.floor(100 + Math.random() * 200),
            seed,
            epochs,
            confusion_matrix: isCls ? [[148, 19], [15, 138]] : undefined,
          };

          setTrainedRun(newRun);
          setTrainingStatus('completed');
          onTrainingComplete(newRun);
        }
        setTrainingProgress(Math.min(prog, 100));
      }, 300);
    }, 800);
  };

  const handlePromoteChampion = () => {
    // Mock champion promotion
    if (trainedRun) {
      alert(`Champion actualizado: ${trainedRun.run_id}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Action Buttons Row */}
      <div className="flex flex-wrap gap-3">
        <Button
          onClick={handlePrepareFeaturePack}
          disabled={featurePackStatus === 'preparing'}
          variant="outline"
          className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-2"
        >
          <Package className="w-4 h-4" />
          {featurePackStatus === 'preparing' ? 'Preparando...' : featurePackStatus === 'ready' ? '✓ Feature-Pack Listo' : 'Preparar Feature-Pack'}
        </Button>
        {featurePackStatus === 'ready' && (
          <Badge className="bg-green-500/20 text-green-400 border-green-500/40 self-center">
            <CheckCircle2 className="w-3 h-3 mr-1" /> Feature-pack listo
          </Badge>
        )}
      </div>

      {/* Training Form */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-6">
        <h4 className="text-white mb-4">Entrenar Modelo</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Modelo */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Modelo</label>
            <Select value={modelo} onValueChange={(v) => setModelo(v as ModelStrategy)}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                {MODEL_STRATEGIES.map(ms => (
                  <SelectItem key={ms.value} value={ms.value}>{ms.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Epochs */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Epochs</label>
            <Input
              type="number"
              value={epochs}
              onChange={e => setEpochs(Number(e.target.value))}
              min={1}
              max={100}
              className="bg-[#0f1419] border-gray-700 h-9 text-sm"
            />
          </div>

          {/* Seed */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Seed</label>
            <Input
              type="number"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              className="bg-[#0f1419] border-gray-700 h-9 text-sm"
            />
          </div>
        </div>

        {/* Auto prepare checkbox */}
        <div className="flex items-center gap-2 mt-4">
          <Checkbox
            id="auto-prepare"
            checked={autoPrepare}
            onCheckedChange={(v) => setAutoPrepare(!!v)}
          />
          <label htmlFor="auto-prepare" className="text-sm text-gray-300 cursor-pointer">
            auto_prepare (preparar feature-pack automáticamente)
          </label>
        </div>

        {/* Warm Start Section */}
        <div className="mt-4 p-4 border border-gray-700 rounded-lg bg-[#0f1419]/50">
          <div className="flex items-center gap-3">
            <Checkbox
              id="warm-start"
              checked={warmStart}
              onCheckedChange={(v) => setWarmStart(!!v)}
            />
            <label htmlFor="warm-start" className="text-sm text-gray-300 cursor-pointer flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-orange-400" /> Warm Start
            </label>
          </div>
          {warmStart && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-3 space-y-3"
            >
              <div>
                <label className="block text-xs text-gray-400 mb-1">warm_start_from</label>
                <Select value={warmStartFrom} onValueChange={(v) => setWarmStartFrom(v as WarmStartFrom)}>
                  <SelectTrigger className="bg-[#0f1419] border-gray-700 h-9 text-sm w-[200px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="champion">Champion</SelectItem>
                    <SelectItem value="run_id">Run ID específico</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {warmStartFrom === 'run_id' && (
                <div>
                  <label className="block text-xs text-gray-400 mb-1">warm_start_run_id</label>
                  <Input
                    value={warmStartRunId}
                    onChange={e => setWarmStartRunId(e.target.value)}
                    placeholder="run_xxxxxxxx"
                    className="bg-[#0f1419] border-gray-700 h-9 text-sm"
                  />
                  {!warmStartValid && (
                    <p className="text-xs text-red-400 mt-1">Run ID es requerido para warm start por run.</p>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Advanced: hparams overrides */}
        <button
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-gray-300 mt-4 transition-colors"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          Avanzado (hparams_overrides)
        </button>
        {showAdvanced && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-2">
            <textarea
              value={hparamsJson}
              onChange={e => {
                setHparamsJson(e.target.value);
                setHparamsError(null);
              }}
              rows={4}
              className="w-full bg-[#0f1419] border border-gray-700 rounded-md p-3 text-sm text-gray-300 font-mono resize-none focus:outline-none focus:border-cyan-500"
              placeholder='{ "learning_rate": 0.001 }'
            />
            {hparamsError && <p className="text-xs text-red-400 mt-1">{hparamsError}</p>}
          </motion.div>
        )}

        {/* Submit */}
        <div className="mt-5 flex gap-3">
          <Button
            onClick={handleTrain}
            disabled={!canSubmit || !warmStartValid}
            className="bg-blue-600 hover:bg-blue-700 gap-2"
          >
            <Play className="w-4 h-4" />
            Entrenar Modelo
          </Button>
        </div>

        {/* Error */}
        {trainingError && (
          <div className="mt-3 bg-red-500/10 border border-red-500/30 rounded-md px-3 py-2 text-sm text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            {trainingError}
          </div>
        )}
      </Card>

      {/* Training Progress / Result */}
      {trainingStatus !== 'idle' && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-white">Resultado del Entrenamiento</h4>
              <RunStatusBadge status={trainingStatus as any} />
            </div>

            {(trainingStatus === 'queued' || trainingStatus === 'running') && (
              <div className="space-y-2">
                <Progress value={trainingProgress} className="h-2" />
                <p className="text-xs text-gray-400">
                  {trainingStatus === 'queued' ? 'En cola...' : `Entrenando... ${Math.round(trainingProgress)}%`}
                </p>
              </div>
            )}

            {trainingStatus === 'completed' && trainedRun && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-gray-400">Run ID</p>
                    <p className="text-cyan-400 text-sm font-mono">{trainedRun.run_id}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">{fc.primaryMetric}</p>
                    <p className="text-white text-xl">{trainedRun.primary_metric_value.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Modelo</p>
                    <p className="text-white">{trainedRun.model_name.replace(/_/g, ' ')}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Duración</p>
                    <p className="text-white">{Math.floor(trainedRun.duration_seconds / 60)}m {trainedRun.duration_seconds % 60}s</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-700 gap-1 text-xs"
                    onClick={() => onNavigateToRun(trainedRun.run_id)}
                  >
                    <Eye className="w-3 h-3" /> Abrir Run
                  </Button>
                  <Button
                    size="sm"
                    className="bg-yellow-600 hover:bg-yellow-700 gap-1 text-xs"
                    onClick={handlePromoteChampion}
                  >
                    <Award className="w-3 h-3" /> Promover a Champion
                  </Button>
                  <Button
                    size="sm"
                    className="bg-cyan-600 hover:bg-cyan-700 gap-1 text-xs"
                    onClick={() => onUsePredictions(trainedRun.run_id)}
                  >
                    <ExternalLink className="w-3 h-3" /> Usar en Predictions
                  </Button>
                </div>
              </div>
            )}

            {trainingStatus === 'failed' && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-md px-3 py-2 text-sm text-red-400">
                Error durante el entrenamiento. Revise la configuración e intente de nuevo.
              </div>
            )}
          </Card>
        </motion.div>
      )}
    </div>
  );
}
