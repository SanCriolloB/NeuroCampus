import { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Play, Award } from 'lucide-react';
import { listRuns, getChampion, RunSummary, ChampionInfo, entrenar, estado } from '../services/modelos';
import { setAppFilters, useAppFilters } from '../state/appFilters.store';

/**
 * Mapeo UI (maqueta) -> dataset_id real del backend.
 *
 * IMPORTANTE:
 * - La maqueta usa valores "dataset1"/"dataset2" en el Select.
 * - En producción usamos activeDatasetId (ej: "2025-1" / "2024-2").
 * - Este mapping mantiene UI 1:1 sin cambiar labels del Select.
 */
const DATASET_OPTIONS = [
  { key: 'dataset1', datasetId: '2025-1', label: 'Evaluations_2025_1' },
  { key: 'dataset2', datasetId: '2024-2', label: 'Evaluations_2024_2' },
] as const;

type DatasetKey = typeof DATASET_OPTIONS[number]['key'];

/** Formatea porcentaje sin romper si viene undefined/null. */
function fmtPct(x?: number | null) {
  if (x === null || x === undefined) return '—';
  return `${(x * 100).toFixed(1)}%`;
}

/** Formatea número sin romper si viene undefined/null. */
function fmtNum(x?: number | null, suffix = '') {
  if (x === null || x === undefined) return '—';
  return `${x}${suffix}`;
}

/**
 * Convierte un model_name del backend a etiqueta legible,
 * manteniendo estilo del UI sin tocar layout.
 */
function prettyModelName(modelName: string) {
  const raw = String(modelName || '').trim();
  if (!raw) return '—';
  return raw
    .split(/[_\s]+/g)
    .map((t) => {
      const low = t.toLowerCase();
      if (low === 'rbm' || low === 'dbm' || low === 'bm') return low.toUpperCase();
      return low.charAt(0).toUpperCase() + low.slice(1);
    })
    .join(' ');
}

const trainingLoss = [
  { epoch: 1, train: 0.65, validation: 0.68 },
  { epoch: 2, train: 0.52, validation: 0.56 },
  { epoch: 3, train: 0.43, validation: 0.48 },
  { epoch: 4, train: 0.37, validation: 0.42 },
  { epoch: 5, train: 0.32, validation: 0.39 },
  { epoch: 6, train: 0.28, validation: 0.36 },
  { epoch: 7, train: 0.25, validation: 0.34 },
  { epoch: 8, train: 0.23, validation: 0.33 },
];

const trainingAccuracy = [
  { epoch: 1, train: 0.68, validation: 0.65 },
  { epoch: 2, train: 0.74, validation: 0.71 },
  { epoch: 3, train: 0.78, validation: 0.75 },
  { epoch: 4, train: 0.82, validation: 0.79 },
  { epoch: 5, train: 0.85, validation: 0.82 },
  { epoch: 6, train: 0.87, validation: 0.84 },
  { epoch: 7, train: 0.89, validation: 0.86 },
  { epoch: 8, train: 0.90, validation: 0.87 },
];

const epochTime = [
  { epoch: 1, time: 22 },
  { epoch: 2, time: 21 },
  { epoch: 3, time: 21 },
  { epoch: 4, time: 20 },
  { epoch: 5, time: 21 },
  { epoch: 6, time: 20 },
  { epoch: 7, time: 21 },
  { epoch: 8, time: 20 },
];

const confusionMatrix = [
  { actual: 'High', predicted: 'High', value: 145 },
  { actual: 'High', predicted: 'Low', value: 22 },
  { actual: 'Low', predicted: 'High', value: 18 },
  { actual: 'Low', predicted: 'Low', value: 135 },
];

export function ModelsTab() {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState('DBM');
  const [viewMode, setViewMode] = useState<'comparison' | 'details'>('comparison');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  /**
   * Dataset activo proveniente del store global (mismo que usa DataTab).
   * Ej: "2024-2" / "2025-1".
   */
  const activeDatasetId = useAppFilters((s) => s.activeDatasetId);

  /**
   * Traduce el dataset_id real al key del Select de la maqueta (dataset1/dataset2).
   * Esto permite un Select controlado sin cambiar UI.
   */
  const datasetKey: DatasetKey = useMemo(() => {
    const found = DATASET_OPTIONS.find((d) => d.datasetId === activeDatasetId);
    return (found?.key ?? 'dataset1') as DatasetKey;
  }, [activeDatasetId]);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [champion, setChampion] = useState<ChampionInfo | null>(null);

    /**
     * Cambia dataset desde ModelsTab sin romper consistencia:
     * actualiza activeDatasetId y activePeriodo en el store.
     */
    const handleDatasetChange = (key: string) => {
      const opt = DATASET_OPTIONS.find((d) => d.key === key);
      const datasetId = opt?.datasetId ?? '2025-1';
      setAppFilters({ activeDatasetId: datasetId, activePeriodo: datasetId });
    };

    /**
     * Carga runs y champion cada vez que cambia el dataset activo.
     * - Runs: si no hay artifacts, retorna [] (esperado).
     * - Champion: 404 se interpreta como "no hay champion aún".
     */
    useEffect(() => {
      // Normalizamos null -> undefined para cumplir tipo (string | undefined)
      const ds = activeDatasetId ?? undefined;

      if (!ds) {
        setRuns([]);
        setChampion(null);
        return;
      }

      let cancelled = false;

      async function load() {
        try {
          const data = await listRuns({ dataset_id: ds });
          if (!cancelled) setRuns(data || []);
        } catch (e: any) {
          console.error('[ModelsTab] Error cargando runs:', e);
          if (!cancelled) setRuns([]);
        }

        try {
          const champ = await getChampion({ dataset_id: ds });
          if (!cancelled) setChampion(champ);
        } catch (e: any) {
          const status = e?.response?.status;
          if (status === 404) {
            if (!cancelled) setChampion(null);
          } else {
            console.error('[ModelsTab] Error cargando champion:', e);
            if (!cancelled) setChampion(null);
          }
        }
      }

      load();
      return () => {
        cancelled = true;
      };
    }, [activeDatasetId]);

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  /**
   * Espera un job consultando /modelos/estado/<job_id>.
   * Termina si status = completed/failed/unknown o por timeout defensivo.
   */
  async function waitJob(jobId: string) {
    for (let i = 0; i < 1200; i++) { // ~20 min defensivo
      const st = await estado(jobId);
      if (st.status === 'completed' || st.status === 'failed' || st.status === 'unknown') return st;
      await sleep(1000);
    }
    return await estado(jobId);
  }

  const handleTrainModels = async () => {
    // Normalizamos null -> undefined
    const ds = activeDatasetId ?? undefined;
    if (!ds) return;

    setIsTraining(true);

    try {
      // Entrenamiento secuencial (evita saturación y mantiene trazabilidad simple)
      const models: Array<'rbm_general' | 'rbm_restringida'> = ['rbm_general', 'rbm_restringida'];

      for (const m of models) {
        const resp = await entrenar({
          modelo: m,
          epochs,
          metodologia: 'periodo_actual',
          periodo_actual: ds,
          hparams: {
            lr: learningRate,
            batch_size: batchSize,
          },
        });

        await waitJob(resp.job_id);
      }

      // Refrescar runs/champion al finalizar (sin depender del useEffect)
      const data = await listRuns({ dataset_id: ds });
      setRuns(data || []);

      try {
        const champ = await getChampion({ dataset_id: ds });
        setChampion(champ);
      } catch {
        setChampion(null);
      }
    } catch (e) {
      console.error('[ModelsTab] Error entrenando modelos:', e);
    } finally {
      setIsTraining(false);
    }
  };

  /**
     * Model comparison derivado desde runs reales.
     * Regla: mostramos el último run por model_name.
     */
    const modelComparison = useMemo(() => {
      const ds = activeDatasetId ?? undefined;

      // Defensa: si por alguna razón vinieran runs sin dataset_id, no los filtramos.
      const filtered = ds ? runs.filter((r) => !r.dataset_id || r.dataset_id === ds) : runs;

      // Orden por created_at desc (si existe)
      const sorted = [...filtered].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

      // Último run por modelo
      const byModel = new Map<string, RunSummary>();
      for (const r of sorted) {
        if (!byModel.has(r.model_name)) byModel.set(r.model_name, r);
      }

      return Array.from(byModel.values()).map((r) => {
        const m = r.metrics || {};
        return {
          model: prettyModelName(r.model_name),
          accuracy: m.accuracy,
          f1: (m.f1_macro ?? m.f1),
          precision: m.precision,
          recall: m.recall,
          time: (m.time_sec ?? m.train_time_sec),
        };
      });
    }, [runs, activeDatasetId]);

    /**
     * bestModel:
     * - Si hay champion, lo priorizamos (cuando exista).
     * - Si no hay champion, tomamos el mayor accuracy entre modelComparison.
     */
    const bestModel = useMemo(() => {
      if (champion?.metrics) {
        const m = champion.metrics || {};
        return {
          model: prettyModelName(champion.model_name),
          accuracy: m.accuracy ?? 0,
          f1: (m.f1_macro ?? m.f1 ?? 0),
          precision: m.precision ?? 0,
          recall: m.recall ?? 0,
          time: (m.time_sec ?? m.train_time_sec ?? 0),
        };
      }

      if (!modelComparison.length) {
        return { model: '—', accuracy: 0, f1: 0, precision: 0, recall: 0, time: 0 };
      }

      return modelComparison.reduce((best, current) => {
        const a = current.accuracy ?? -1;
        const b = best.accuracy ?? -1;
        return a > b ? current : best;
      });
    }, [modelComparison, champion]);

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white mb-2">Models</h2>
          <p className="text-gray-400">Training and Comparison</p>
        </div>
        <Button
          onClick={handleTrainModels}
          disabled={isTraining}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <Play className="w-4 h-4 mr-2" />
          {isTraining ? 'Training...' : 'Train All Models'}
        </Button>
      </div>

      {/* Configuration */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-6">
        <h3 className="text-white mb-4">Training Configuration</h3>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Dataset</label>
            <Select value={datasetKey} onValueChange={handleDatasetChange}>
              <SelectTrigger className="bg-[#0f1419] border-gray-700">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="dataset1">Evaluations_2025_1</SelectItem>
                <SelectItem value="dataset2">Evaluations_2024_2</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Epochs</label>
            <Input
              type="number"
              value={epochs}
              min={1}
              step={1}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setEpochs(v);
              }}
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
            <Input
              type="number"
              value={batchSize}
              min={1}
              step={1}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setBatchSize(v);
              }}
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Learning Rate</label>
            <Input
              type="number"
              value={learningRate}
              min={0}
              step={0.001}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setLearningRate(v);
              }}
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
        </div>
      </Card>

      {/* Tabs */}
      <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'comparison' | 'details')}>
        <TabsList className="bg-[#1a1f2e] border border-gray-800">
          <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
          <TabsTrigger value="details">Best Model Details</TabsTrigger>
        </TabsList>

        <TabsContent value="comparison" className="space-y-6 mt-6">
          {/* Comparison Table */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Performance Metrics Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Model</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Accuracy</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">F1 Score</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Precision</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Recall</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {modelComparison.map((model) => {
                    const isBest = model.model === bestModel.model;
                    return (
                      <tr
                        key={model.model}
                        className={`border-b border-gray-800/50 ${
                          isBest ? 'bg-blue-500/10' : 'hover:bg-gray-800/30'
                        }`}
                      >
                        <td className="text-gray-300 py-3 px-4 flex items-center gap-2">
                          {model.model}
                          {isBest && (
                            <Badge className="bg-blue-600 text-white">
                              <Award className="w-3 h-3 mr-1" />
                              Best
                            </Badge>
                          )}
                        </td>
                        <td className="text-gray-300 py-3 px-4">{fmtPct(model.accuracy)}</td>
                        <td className="text-gray-300 py-3 px-4">{fmtPct(model.f1)}</td>
                        <td className="text-gray-300 py-3 px-4">{fmtPct(model.precision)}</td>
                        <td className="text-gray-300 py-3 px-4">{fmtPct(model.recall)}</td>
                        <td className="text-gray-300 py-3 px-4">{fmtNum(model.time, 's')}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Comparison Chart */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Accuracy Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="model" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={[0.7, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: any) => (typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—')}
                />
                <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy" />
                <Bar dataKey="f1" fill="#06B6D4" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="details" className="space-y-6 mt-6">
          {/* Selected Model Info */}
          <Card className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border-blue-600/50 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white mb-2">Selected Model: {bestModel.model}</h3>
                <p className="text-gray-300">
                  Accuracy: {fmtPct(bestModel.accuracy)} •  
                  F1 Score: {fmtPct(bestModel.f1)} • 
                  Training Time: {fmtNum(bestModel.time, 's')}
                </p>
              </div>
              <Award className="w-12 h-12 text-blue-400" />
            </div>
          </Card>

          {/* Training Curves */}
          <div className="grid grid-cols-2 gap-6">
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingLoss}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train" stroke="#3B82F6" strokeWidth={2} name="Training" />
                  <Line type="monotone" dataKey="validation" stroke="#F59E0B" strokeWidth={2} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Training Accuracy</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingAccuracy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[0.6, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train" stroke="#10B981" strokeWidth={2} name="Training" />
                  <Line type="monotone" dataKey="validation" stroke="#06B6D4" strokeWidth={2} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Time per Epoch and Confusion Matrix */}
          <div className="grid grid-cols-2 gap-6">
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Time per Epoch</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={epochTime}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Bar dataKey="time" fill="#8B5CF6" name="Time (seconds)" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Confusion Matrix</h3>
              <div className="grid grid-cols-2 gap-4 mt-8">
                <div className="bg-green-500/20 border-2 border-green-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">True Positive</p>
                  <p className="text-white text-3xl mt-2">145</p>
                </div>
                <div className="bg-red-500/20 border-2 border-red-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">False Positive</p>
                  <p className="text-white text-3xl mt-2">18</p>
                </div>
                <div className="bg-red-500/20 border-2 border-red-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">False Negative</p>
                  <p className="text-white text-3xl mt-2">22</p>
                </div>
                <div className="bg-green-500/20 border-2 border-green-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">True Negative</p>
                  <p className="text-white text-3xl mt-2">135</p>
                </div>
              </div>
              <div className="mt-4 text-sm text-gray-400 space-y-1">
                <p>Precision: {((145 / (145 + 18)) * 100).toFixed(1)}%</p>
                <p>Recall: {((145 / (145 + 22)) * 100).toFixed(1)}%</p>
                <p>Specificity: {((135 / (135 + 18)) * 100).toFixed(1)}%</p>
              </div>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
