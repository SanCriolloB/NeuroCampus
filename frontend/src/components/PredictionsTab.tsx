import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  listDatasets,
  listTeachers,
  listMaterias,
  predictIndividual,
  runBatch,
  getBatchJob,
  getOutputsPreview,
  getArtifactDownloadUrl,
  type DatasetInfo,
  type TeacherInfo,
  type MateriaInfo,
  type IndividualPredictionResponse,
  type BatchJobStatus,
} from '../services/predicciones';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { TrendingUp, AlertTriangle, CheckCircle, Search, Filter } from 'lucide-react';
import { motion } from 'motion/react';

// Teacher and subject data - simulating database

export function PredictionsTab() {
  // --- Modo ---
  const [predictionMode, setPredictionMode] = useState<'individual' | 'batch'>('individual');

  // --- Estado de datos remotos ---
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [teachers, setTeachers] = useState<TeacherInfo[]>([]);
  const [materias, setMaterias] = useState<MateriaInfo[]>([]);

  // --- Selección individual ---
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedTeacher, setSelectedTeacher] = useState<string>('');
  const [selectedMateria, setSelectedMateria] = useState<string>('');
  const [teacherSearch, setTeacherSearch] = useState('');
  const [subjectSearch, setSubjectSearch] = useState('');

  // --- Resultado individual ---
  const [predResult, setPredResult] = useState<IndividualPredictionResponse | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [isLoadingPred, setIsLoadingPred] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);

  // --- Batch ---
  const [showBatchResults, setShowBatchResults] = useState(false);
  const [batchStatus, setBatchStatus] = useState<BatchJobStatus | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchRows, setBatchRows] = useState<any[]>([]);
  const [predictionsUri, setPredictionsUri] = useState<string | null>(null);
  const [riskFilter, setRiskFilter] = useState('all');
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);  const gaugePercent = predResult ? Math.round((predResult.score_total_pred / 50) * 100) : 0;
  const confidenceDisplay = predResult ? `${Math.round(predResult.confidence * 100)}%` : '—';
  const radarData = predResult?.radar ?? [];
  const comparisonData = predResult?.comparison ?? [];
  const historicalData = predResult?.timeline ?? [];
  const predRisk = predResult?.risk ?? 'low';
  const batchTotal = batchStatus?.n_pairs ?? batchRows.length;
  const batchLow = batchRows.filter((r) => r.risk === 'low').length;
  const batchMedium = batchRows.filter((r) => r.risk === 'medium').length;
  const batchHigh = batchRows.filter((r) => r.risk === 'high').length;

  // Cargar datasets al montar
  useEffect(() => {
    listDatasets()
      .then((ds) => {
        setDatasets(ds);
        if (ds.length > 0) setSelectedDataset(ds[ds.length - 1].dataset_id);
      })
      .catch(() => setDatasets([]));
  }, []);

  // Cargar teachers y materias al cambiar dataset
  useEffect(() => {
    if (!selectedDataset) return;
    listTeachers(selectedDataset).then(setTeachers).catch(() => setTeachers([]));
    listMaterias(selectedDataset).then(setMaterias).catch(() => setMaterias([]));
    setSelectedTeacher('');
    setSelectedMateria('');
    setPredResult(null);
    setShowResults(false);
  }, [selectedDataset]);

  // Limpiar polling al desmontar
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);


  // Filter teachers by search
  const filteredTeachers = useMemo(() => {
    if (!teacherSearch) return teachers;
    const search = teacherSearch.toLowerCase();
    return teachers.filter((t) => t.teacher_key.toLowerCase().includes(search));
  }, [teacherSearch, teachers]);

  // Filter subjects by search and selected teacher
  const filteredSubjects = useMemo(() => {
    if (!subjectSearch) return materias;
    const search = subjectSearch.toLowerCase();
    return materias.filter((m) => m.materia_key.toLowerCase().includes(search));
  }, [subjectSearch, materias]);


  const filteredBatchResults = useMemo(() => {
    if (riskFilter === 'all') return batchRows;
    return batchRows.filter((r) => r.risk === riskFilter);
  }, [batchRows, riskFilter]);

  const handleGeneratePrediction = useCallback(async () => {
    if (!selectedDataset || !selectedTeacher || !selectedMateria) return;
    setIsLoadingPred(true);
    setPredError(null);

    try {
      const result = await predictIndividual({
        dataset_id: selectedDataset,
        teacher_key: selectedTeacher,
        materia_key: selectedMateria,
      });
      setPredResult(result);
      setShowResults(true);
      setTimeout(() => {
        document.getElementById('prediction-results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (e: any) {
      const detail = e?.response?.data?.detail ?? e?.message ?? 'Error al generar predicción';
      setPredError(typeof detail === 'string' ? detail : JSON.stringify(detail));
    } finally {
      setIsLoadingPred(false);
    }
  }, [selectedDataset, selectedTeacher, selectedMateria]);


  const handleGenerateBatch = useCallback(async () => {
    if (!selectedDataset) return;

    setBatchError(null);
    setShowBatchResults(false);
    setBatchRows([]);
    setPredictionsUri(null);

    try {
      const job = await runBatch(selectedDataset);
      setBatchStatus(job);

      // Polling cada 2s
      pollingRef.current = setInterval(async () => {
        try {
          const status = await getBatchJob(job.job_id);
          setBatchStatus(status);

          if (status.status === 'completed') {
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;

            if (status.predictions_uri) {
              setPredictionsUri(status.predictions_uri);

              const preview = await getOutputsPreview({
                predictions_uri: status.predictions_uri,
                limit: 500,
                offset: 0,
              });

              setBatchRows(preview.rows ?? []);
            }

            setShowBatchResults(true);
          } else if (status.status === 'failed') {
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;
            setBatchError(status.error ?? 'Error en el procesamiento del lote.');
          }
        } catch {
          // error transitorio: mantener polling
        }
      }, 2000);
    } catch (e: any) {
      const detail = e?.response?.data?.detail ?? e?.message ?? 'Error al lanzar el batch';
      setBatchError(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
  }, [selectedDataset]);


  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-400 bg-green-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/20';
      case 'high': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'low': return CheckCircle;
      case 'medium': return AlertTriangle;
      case 'high': return AlertTriangle;
      default: return CheckCircle;
    }
  };

  // Get heatmap data for batch results
  const heatmapData = useMemo(() => {
    const matrix: Record<string, Record<string, number>> = {};
    batchRows.forEach((r) => {
      const t = String(r.teacher_key ?? '');
      const s = String(r.materia_key ?? '');
      const percent = r.score_total_pred != null ? Math.round((Number(r.score_total_pred) / 50) * 100) : null;

      if (!matrix[t]) matrix[t] = {};
      if (percent != null) matrix[t][s] = percent;
    });
    return matrix;
  }, [batchRows]);

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h2 className="text-white mb-2">Predicciones</h2>
        <p className="text-gray-400">Sistema de Predicción de Rendimiento Docente</p>
      </motion.div>

      {/* Tabs */}
      <Tabs value={predictionMode} onValueChange={(v) => setPredictionMode(v as 'individual' | 'batch')}>
        <TabsList className="bg-[#1a1f2e] border border-gray-800">
          <TabsTrigger value="individual">Predicción Individual</TabsTrigger>
          <TabsTrigger value="batch">Predicción por Lote</TabsTrigger>
        </TabsList>

        {/* 3.1 Individual Prediction */}
        <TabsContent value="individual" className="mt-6">
          <div className="grid grid-cols-3 gap-6">
            {/* Left Column - Selection Form */}
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                <h3 className="text-white mb-4">Seleccionar Docente y Asignatura</h3>
                <div className="space-y-4">
                  {/* Dataset */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Dataset</label>
                    <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                      <SelectTrigger className="bg-[#0f1419] border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1f2e] border-gray-700">
                        {datasets.map((ds) => (
                          <SelectItem key={ds.dataset_id} value={ds.dataset_id}>
                            {ds.dataset_id} — {ds.n_pairs} pares {ds.has_champion ? '✓' : '⚠'}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-500 mt-1">
                      Seleccione un dataset previamente cargado en la pestaña Datos
                    </p>
                  </div>

                  {/* Teacher Selection with Search */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Docente</label>
                    <div className="relative mb-2">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                      <Input
                        value={teacherSearch}
                        onChange={(e) => setTeacherSearch(e.target.value)}
                        placeholder="Buscar por nombre o código..."
                        className="bg-[#0f1419] border-gray-700 pl-10"
                      />
                    </div>
                    <Select value={selectedTeacher} onValueChange={setSelectedTeacher}>
                      <SelectTrigger className="bg-[#0f1419] border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1f2e] border-gray-700 max-h-[300px]">
                        {filteredTeachers.map((t) => (
                          <SelectItem key={t.teacher_key} value={t.teacher_key}>
                            {t.teacher_key}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Subject Selection with Search */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Asignatura</label>
                    <div className="relative mb-2">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                      <Input
                        value={subjectSearch}
                        onChange={(e) => setSubjectSearch(e.target.value)}
                        placeholder="Buscar por nombre o código..."
                        className="bg-[#0f1419] border-gray-700 pl-10"
                      />
                    </div>
                    <Select value={selectedMateria} onValueChange={setSelectedMateria}>
                      <SelectTrigger className="bg-[#0f1419] border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1f2e] border-gray-700 max-h-[300px]">
                        {filteredSubjects.map((m) => (
                          <SelectItem key={m.materia_key} value={m.materia_key}>
                            {m.materia_key}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </Card>

              <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                <h3 className="text-white mb-4">Información Seleccionada</h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-gray-400">Dataset</p>
                    <p className="text-white">{selectedDataset || '—'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Docente</p>
                    <p className="text-white">{selectedTeacher || '—'}</p>
                    <p className="text-xs text-gray-500">
                      Encuestas:{' '}
                      {teachers.find((t) => t.teacher_key === selectedTeacher)?.n_encuestas ?? '—'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Asignatura</p>
                    <p className="text-white">{selectedMateria || '—'}</p>
                    <p className="text-xs text-gray-500">
                      Encuestas:{' '}
                      {materias.find((m) => m.materia_key === selectedMateria)?.n_encuestas ?? '—'}
                    </p>
                  </div>
                  <div className="pt-2 border-t border-gray-700">
                    <p className="text-sm text-gray-400">Score histórico (par)</p>
                    <p className="text-cyan-400 text-2xl">
                      {showResults && predResult ? predResult.historical.mean_score : '—'}
                    </p>
                  </div>
                </div>
              </Card>

              <Button
                onClick={handleGeneratePrediction}
                disabled={isLoadingPred || !selectedTeacher || !selectedMateria}
                className="w-full bg-blue-600 hover:bg-blue-700"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                {isLoadingPred ? 'Generando...' : 'Generar Predicción'}
              </Button>
            </motion.div>

            {/* Right Column - Results */}
            {predError && (
              <div className="col-span-3 p-4 rounded border border-red-500 bg-red-500/10 text-red-400">
                {predError}
              </div>
            )}

            <div className="col-span-2 space-y-6" id="prediction-results">
              {showResults ? (
                <>
                  {/* Prediction Result */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Card className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border-blue-600/50 p-6">
                      <h3 className="text-white mb-4">Resultado de Predicción</h3>
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <p className="text-gray-300 mb-2">Probabilidad de Alto Rendimiento</p>
                          <div className="flex items-end gap-2">
                            <span className="text-5xl text-white">{gaugePercent}%</span>
                            <span className="text-gray-400 mb-2">Confianza: {confidenceDisplay}</span>
                          </div>
                        </div>
                        <div className="flex items-center justify-center">
                          <Badge className={`${getRiskColor(predRisk)} px-6 py-3 text-lg`}>
                            {predRisk === 'low' ? 'BAJO' : predRisk === 'medium' ? 'MODERADO' : 'ALTO'} RIESGO
                          </Badge>
                        </div>
                      </div>
                      
                      {/* Gauge/Progress Bar */}
                      <div className="mt-6">
                        <div className="h-8 bg-gray-800 rounded-full overflow-hidden relative">
                          <motion.div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-400"
                            initial={{ width: 0 }}
                            animate={{ width: `${gaugePercent}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                          />
                          <div
                            className="absolute top-0 h-full w-1 bg-white"
                            style={{ left: `${gaugePercent}%` }}
                          />
                        </div>
                        <div className="flex justify-between mt-2 text-sm text-gray-400">
                          <span>0%</span>
                          <span>50%</span>
                          <span>100%</span>
                        </div>
                      </div>

                      <p className="text-gray-300 mt-4 text-center">
                        {predRisk === 'low' 
                          ? 'Excelente rendimiento esperado. Continuar con estrategias actuales.'
                          : predRisk === 'medium'
                          ? 'Riesgo moderado. Considerar estrategias de apoyo adicionales.'
                          : 'Alto riesgo de bajo rendimiento. Se recomienda intervención inmediata.'}
                      </p>
                    </Card>
                  </motion.div>

                  {/* Radar Chart */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                  >
                    <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                      <h3 className="text-white mb-4">Perfil de Indicadores (Radar)</h3>
                      <ResponsiveContainer width="100%" height={400}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="#374151" />
                          <PolarAngleAxis dataKey="indicator" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                          <PolarRadiusAxis angle={90} domain={[0, 5]} stroke="#9CA3AF" />
                          <Radar name="Promedio Actual" dataKey="actual" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
                          <Radar name="Predicción" dataKey="prediccion" stroke="#10B981" fill="#10B981" fillOpacity={0.4} />
                          <Legend />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </Card>
                  </motion.div>

                  {/* Bar Comparison */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                      <h3 className="text-white mb-4">Análisis Comparativo por Dimensión</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={comparisonData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="dimension" stroke="#9CA3AF" />
                          <YAxis domain={[0, 5]} stroke="#9CA3AF" />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                            labelStyle={{ color: '#fff' }}
                          />
                          <Legend />
                          <Bar dataKey="docente" fill="#3B82F6" name="Docente Seleccionado" />
                          <Bar dataKey="cohorte" fill="#6B7280" name="Promedio Cohorte" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>
                  </motion.div>

                  {/* Temporal Projection */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                  >
                    <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                      <h3 className="text-white mb-4">Proyección Temporal</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={historicalData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="semester" stroke="#9CA3AF" />
                          <YAxis domain={[0, 50]} stroke="#9CA3AF" />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                            labelStyle={{ color: '#fff' }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="real" stroke="#10B981" strokeWidth={2} name="Rendimiento Real" />
                          <Line
                            type="monotone"
                            dataKey="predicted"
                            stroke="#3B82F6"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            name="Predicción"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  </motion.div>
                </>
              ) : (
                <Card className="bg-[#1a1f2e] border-gray-800 p-12">
                  <div className="text-center text-gray-500">
                    <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Seleccione un docente y asignatura, luego haga clic en "Generar Predicción"</p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* 3.2 Batch Prediction */}
        <TabsContent value="batch" className="mt-6 space-y-6">
          {/* Dataset and Model Selection */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Seleccionar Dataset</h3>
            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-2">
                <label className="block text-sm text-gray-400 mb-2">Dataset</label>
                <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    {datasets.map((ds) => (
                    <SelectItem key={ds.dataset_id} value={ds.dataset_id}>
                      {ds.dataset_id} — {ds.n_pairs} pares {ds.has_champion ? '✓' : '⚠'}
                    </SelectItem>
                  ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-1">
                  Seleccione un dataset previamente cargado en la pestaña Datos
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Champion automático</p>
                <p className="text-white text-sm">{batchStatus?.champion_run_id ?? '—'}</p>
              </div>
            </div>
            <Button 
              onClick={handleGenerateBatch}
              className="w-full bg-blue-600 hover:bg-blue-700 mt-4"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Generar Predicciones del Lote
            </Button>
          </Card>

          {batchStatus && batchStatus.status === 'running' && (
            <div className="w-full mt-4">
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Procesando lote…</span>
                <span>{Math.round((batchStatus.progress ?? 0) * 100)}%</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${Math.round((batchStatus.progress ?? 0) * 100)}%` }}
                />
              </div>
            </div>
          )}
          {batchError && (
            <div className="p-4 rounded border border-red-500 bg-red-500/10 text-red-400">
              {batchError}
            </div>
          )}

          {showBatchResults && (
            <>
              {/* Batch Results Summary */}
              <motion.div
                className="grid grid-cols-4 gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">Registros Procesados</p>
                  <p className="text-white text-3xl">{batchTotal}</p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">Alto Rendimiento</p>
                  <p className="text-green-400 text-3xl">
                    {batchLow}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(batchTotal ? ((batchLow / batchTotal) * 100).toFixed(0) : '0')}%
                  </p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">Riesgo Medio</p>
                  <p className="text-yellow-400 text-3xl">
                    {batchMedium}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(batchTotal ? ((batchMedium / batchTotal) * 100).toFixed(0) : '0')}%
                  </p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">En Riesgo</p>
                  <p className="text-red-400 text-3xl">
                    {batchHigh}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(batchTotal ? ((batchHigh / batchTotal) * 100).toFixed(0) : '0')}%
                  </p>
                </Card>
              </motion.div>

              {/* Distribution Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <h3 className="text-white mb-4">Distribución de Riesgo por Materia</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={materias.slice(0, 8).map((m) => {
                        const subjectResults = batchRows.filter((r) => r.materia_key === m.materia_key);
                        return {
                          subject: m.materia_key,
                          bajo: subjectResults.filter((r) => r.risk === 'low').length,
                          medio: subjectResults.filter((r) => r.risk === 'medium').length,
                          alto: subjectResults.filter((r) => r.risk === 'high').length,
                        };
                      })}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="subject" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Legend />
                      <Bar dataKey="bajo" stackId="a" fill="#10B981" name="Bajo Riesgo" />
                      <Bar dataKey="medio" stackId="a" fill="#F59E0B" name="Medio Riesgo" />
                      <Bar dataKey="alto" stackId="a" fill="#EF4444" name="Alto Riesgo" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </motion.div>

              {/* Heatmap */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <h3 className="text-white mb-4">Mapa de Calor: Score (% del máximo)</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-800">
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Docente</th>
                          {materias.slice(0, 4).map((s) => (
                            <th key={s.materia_key} className="text-center text-gray-400 text-sm py-3 px-2">{s.materia_key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(heatmapData).map(([teacher, subjects]: [string, any]) => (
                          <tr key={teacher} className="border-b border-gray-800/50">
                            <td className="text-gray-300 text-sm py-3 px-4">{teacher.split(' ').slice(0, 2).join(' ')}</td>
                            {materias.slice(0, 4).map((s) => {
                              const prob = subjects[s.materia_key];
                              const bgColor = prob != null ?
                                prob > 80 ? 'bg-green-500/80' :
                                prob > 70 ? 'bg-green-500/50' :
                                prob > 60 ? 'bg-yellow-500/50' :
                                'bg-red-500/50' : 'bg-gray-700';
                              return (
                                <td key={s.materia_key} className="text-center py-3 px-2">
                                  <div className={`${bgColor} rounded px-2 py-1 text-white text-sm`}>
                                    {prob != null ? `${prob}%` : '-'}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </motion.div>

              {/* Results Table */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-white">Tabla de Predicciones</h3>
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge 
                        className={`cursor-pointer ${riskFilter === 'all' ? 'bg-blue-600' : 'bg-gray-700'}`}
                        onClick={() => setRiskFilter('all')}
                      >
                        Todos
                      </Badge>
                      <Badge 
                        className={`cursor-pointer ${riskFilter === 'low' ? 'bg-green-600' : 'bg-gray-700'}`}
                        onClick={() => setRiskFilter('low')}
                      >
                        Bajo
                      </Badge>
                      <Badge 
                        className={`cursor-pointer ${riskFilter === 'medium' ? 'bg-yellow-600' : 'bg-gray-700'}`}
                        onClick={() => setRiskFilter('medium')}
                      >
                        Medio
                      </Badge>
                      <Badge 
                        className={`cursor-pointer ${riskFilter === 'high' ? 'bg-red-600' : 'bg-gray-700'}`}
                        onClick={() => setRiskFilter('high')}
                      >
                        Alto
                      </Badge>
                      {predictionsUri && (
                        <a href={getArtifactDownloadUrl(predictionsUri)} download>
                          <Button className="bg-blue-600 hover:bg-blue-700">Descargar Predicciones</Button>
                        </a>
                      )}
                    </div>
                  </div>
                  <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                    <table className="w-full">
                      <thead className="sticky top-0 bg-[#1a1f2e]">
                        <tr className="border-b border-gray-800">
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Docente</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Materia</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Score (0–50)</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Confianza</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Nivel de Riesgo</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredBatchResults.map((result, index) => {
                          const RiskIcon = getRiskIcon(result.risk);
                          return (
                            <tr key={index} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                              <td className="text-gray-300 text-sm py-3 px-4">{result.teacher_key ?? '—'}</td>
                              <td className="text-gray-300 text-sm py-3 px-4">{result.materia_key ?? '—'}</td>
                              <td className="text-gray-300 text-sm py-3 px-4">
                                {result.score_total_pred != null ? Number(result.score_total_pred).toFixed(2) : '—'}
                              </td>
                              <td className="text-gray-300 text-sm py-3 px-4">
                                {result.confidence != null ? `${Math.round(Number(result.confidence) * 100)}%` : '—'}
                              </td>
                              <td className="text-gray-300 text-sm py-3 px-4">
                                <Badge className={getRiskColor(result.risk)}>
                                  <RiskIcon className="w-3 h-3 mr-1" />
                                  {result.risk === 'low' ? 'Bajo' : result.risk === 'medium' ? 'Medio' : 'Alto'}
                                </Badge>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </motion.div>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
