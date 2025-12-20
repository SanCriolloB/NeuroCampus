import { useState, useMemo } from 'react';
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
const teachersDatabase = [
  { id: 'garcia', name: 'Dr. María García', code: 'T001', baseScore: 92, department: 'Matemáticas' },
  { id: 'martinez', name: 'Prof. Juan Martínez', code: 'T002', baseScore: 88, department: 'Física' },
  { id: 'lopez', name: 'Dr. Ana López', code: 'T003', baseScore: 85, department: 'Ingeniería' },
  { id: 'rodriguez', name: 'Prof. Carlos Rodríguez', code: 'T004', baseScore: 82, department: 'Química' },
  { id: 'fernandez', name: 'Dr. Laura Fernández', code: 'T005', baseScore: 78, department: 'Matemáticas' },
  { id: 'santos', name: 'Dra. Patricia Santos', code: 'T006', baseScore: 90, department: 'Física' },
  { id: 'torres', name: 'Prof. Miguel Torres', code: 'T007', baseScore: 84, department: 'Química' },
  { id: 'ramirez', name: 'Dr. Roberto Ramírez', code: 'T008', baseScore: 87, department: 'Ingeniería' },
];

const subjectsDatabase = [
  { id: 'calculus', name: 'Cálculo I', code: 'MAT101', basePerformance: 80, department: 'Matemáticas' },
  { id: 'physics', name: 'Física II', code: 'FIS201', basePerformance: 75, department: 'Física' },
  { id: 'programming', name: 'Programación Avanzada', code: 'ING301', basePerformance: 85, department: 'Ingeniería' },
  { id: 'organic_chem', name: 'Química Orgánica', code: 'QUI201', basePerformance: 72, department: 'Química' },
  { id: 'inorganic_chem', name: 'Química Inorgánica', code: 'QUI101', basePerformance: 74, department: 'Química' },
  { id: 'mathematics', name: 'Matemáticas Discretas', code: 'MAT201', basePerformance: 78, department: 'Matemáticas' },
  { id: 'statistics', name: 'Estadística Aplicada', code: 'MAT301', basePerformance: 82, department: 'Matemáticas' },
];

const periodsData = ['2023-1', '2023-2', '2024-1', '2024-2', '2025-1'];

// Datasets available (from Data tab)
const datasetsAvailable = [
  { id: 'ds1', name: 'Evaluaciones 2024-2', period: '2024-2', rows: 1250, date: '2024-11-15' },
  { id: 'ds2', name: 'Evaluaciones 2024-1', period: '2024-1', rows: 1180, date: '2024-06-20' },
  { id: 'ds3', name: 'Evaluaciones 2023-2', period: '2023-2', rows: 1100, date: '2023-12-10' },
  { id: 'ds4', name: 'Dataset Completo 2023-2024', period: 'Multi', rows: 4500, date: '2024-12-01' },
];

// Mock batch results with heatmap data
const generateBatchResults = () => {
  const results: any[] = [];
  teachersDatabase.slice(0, 5).forEach(teacher => {
    subjectsDatabase.slice(0, 4).forEach(subject => {
      const probHigh = 0.6 + Math.random() * 0.35;
      const probLow = 1 - probHigh;
      let risk = 'low';
      if (probHigh < 0.7) risk = 'high';
      else if (probHigh < 0.8) risk = 'medium';
      
      results.push({
        teacher: teacher.name,
        subject: subject.name,
        probHigh,
        probLow,
        risk,
      });
    });
  });
  return results;
};

export function PredictionsTab() {
  const [predictionMode, setPredictionMode] = useState<'individual' | 'batch'>('individual');
  
  // Individual prediction state
  const [selectedTeacher, setSelectedTeacher] = useState('garcia');
  const [selectedSubject, setSelectedSubject] = useState('calculus');
  const [teacherSearch, setTeacherSearch] = useState('');
  const [subjectSearch, setSubjectSearch] = useState('');
  const [showResults, setShowResults] = useState(false);
  
  // Batch prediction state
  const [selectedDataset, setSelectedDataset] = useState('ds1');
  const [selectedModel, setSelectedModel] = useState('dbm');
  const [showBatchResults, setShowBatchResults] = useState(false);
  const [riskFilter, setRiskFilter] = useState('all');

  // Filter teachers by search
  const filteredTeachers = useMemo(() => {
    if (!teacherSearch) return teachersDatabase;
    const search = teacherSearch.toLowerCase();
    return teachersDatabase.filter(t => 
      t.name.toLowerCase().includes(search) || 
      t.code.toLowerCase().includes(search)
    );
  }, [teacherSearch]);

  // Filter subjects by search and selected teacher
  const filteredSubjects = useMemo(() => {
    let subjects = subjectsDatabase;
    
    // Filter by teacher's department
    if (selectedTeacher) {
      const teacher = teachersDatabase.find(t => t.id === selectedTeacher);
      if (teacher) {
        // Show subjects from same department or related ones
        subjects = subjects.filter(s => 
          s.department === teacher.department || 
          teacher.department === 'Matemáticas' // Math teachers can teach many subjects
        );
      }
    }
    
    if (!subjectSearch) return subjects;
    const search = subjectSearch.toLowerCase();
    return subjects.filter(s => 
      s.name.toLowerCase().includes(search) || 
      s.code.toLowerCase().includes(search)
    );
  }, [subjectSearch, selectedTeacher]);

  // Generate prediction data based on selected teacher and subject
  const predictionData = useMemo(() => {
    const teacher = teachersDatabase.find(t => t.id === selectedTeacher)!;
    const subject = subjectsDatabase.find(s => s.id === selectedSubject)!;
    
    const baseProb = Math.round((teacher.baseScore + subject.basePerformance) / 2);
    const variation = Math.random() * 10 - 5;
    const highProb = Math.max(55, Math.min(95, Math.round(baseProb + variation)));
    const lowProb = 100 - highProb;
    
    let risk = 'low';
    if (highProb < 70) risk = 'high';
    else if (highProb < 80) risk = 'medium';
    
    // Radar data - 10 indicators: Promedio Actual del Docente vs Predicción
    const radarData = [
      { indicator: 'Planificación', actual: 3.5 + (teacher.baseScore / 100) * 1.5, prediccion: 3.7 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Metodología', actual: 3.8 + (teacher.baseScore / 100) * 1.5, prediccion: 4.0 + (teacher.baseScore / 100) * 1.3 },
      { indicator: 'Claridad', actual: 3.2 + (teacher.baseScore / 100) * 1.5, prediccion: 3.4 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Evaluación', actual: 3.4 + (teacher.baseScore / 100) * 1.5, prediccion: 3.6 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Materiales', actual: 3.6 + (teacher.baseScore / 100) * 1.5, prediccion: 3.8 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Interacción', actual: 4.0 + (teacher.baseScore / 100) * 1.5, prediccion: 4.2 + (teacher.baseScore / 100) * 1.3 },
      { indicator: 'Retroalimentación', actual: 3.5 + (teacher.baseScore / 100) * 1.5, prediccion: 3.7 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Innovación', actual: 3.3 + (teacher.baseScore / 100) * 1.5, prediccion: 3.5 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Puntualidad', actual: 3.7 + (teacher.baseScore / 100) * 1.5, prediccion: 3.8 + (teacher.baseScore / 100) * 1.4 },
      { indicator: 'Disponibilidad', actual: 3.9 + (teacher.baseScore / 100) * 1.5, prediccion: 4.0 + (teacher.baseScore / 100) * 1.3 },
    ];
    
    // Comparison data - by dimension
    const comparisonData = [
      { dimension: 'Planificación', docente: 4.2, cohorte: 3.8 },
      { dimension: 'Metodología', docente: 4.5, cohorte: 4.0 },
      { dimension: 'Evaluación', docente: 4.0, cohorte: 3.7 },
      { dimension: 'Interacción', docente: 4.7, cohorte: 4.2 },
      { dimension: 'Materiales', docente: 4.3, cohorte: 4.1 },
    ].map(d => ({
      ...d,
      docente: Math.min(5, d.docente * (teacher.baseScore / 90)),
      cohorte: d.cohorte,
    }));
    
    // Historical data - using last 5 periods
    const historicalData = periodsData.map((per, idx) => ({
      semester: per,
      real: idx < periodsData.length - 1 ? Math.round(baseProb - (periodsData.length - idx - 1) * 2 + Math.random() * 3) : undefined,
      predicted: Math.round(baseProb - (periodsData.length - idx - 1) * 2 + Math.random() * 3),
    }));
    
    return {
      highProb,
      lowProb,
      confidenceInterval: [Math.max(0, highProb - 8), Math.min(100, highProb + 7)],
      risk,
      radarData,
      comparisonData,
      historicalData,
    };
  }, [selectedTeacher, selectedSubject]);

  const batchResults = useMemo(() => generateBatchResults(), []);

  const filteredBatchResults = useMemo(() => {
    if (riskFilter === 'all') return batchResults;
    return batchResults.filter(r => r.risk === riskFilter);
  }, [batchResults, riskFilter]);

  const handleGeneratePrediction = () => {
    setShowResults(true);
    setTimeout(() => {
      document.getElementById('prediction-results')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleGenerateBatch = () => {
    setShowBatchResults(true);
  };

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
    const matrix: any = {};
    batchResults.forEach(r => {
      if (!matrix[r.teacher]) matrix[r.teacher] = {};
      matrix[r.teacher][r.subject] = r.probHigh;
    });
    return matrix;
  }, [batchResults]);

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
                        {filteredTeachers.map((teacher) => (
                          <SelectItem key={teacher.id} value={teacher.id}>
                            {teacher.name} ({teacher.code})
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
                    <Select value={selectedSubject} onValueChange={setSelectedSubject}>
                      <SelectTrigger className="bg-[#0f1419] border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1f2e] border-gray-700 max-h-[300px]">
                        {filteredSubjects.map((subject) => (
                          <SelectItem key={subject.id} value={subject.id}>
                            {subject.name} ({subject.code})
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
                    <p className="text-sm text-gray-400">Docente</p>
                    <p className="text-white">{teachersDatabase.find(t => t.id === selectedTeacher)?.name}</p>
                    <p className="text-xs text-gray-500">{teachersDatabase.find(t => t.id === selectedTeacher)?.code}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Asignatura</p>
                    <p className="text-white">{subjectsDatabase.find(s => s.id === selectedSubject)?.name}</p>
                    <p className="text-xs text-gray-500">{subjectsDatabase.find(s => s.id === selectedSubject)?.code}</p>
                  </div>
                  <div className="pt-2 border-t border-gray-700">
                    <p className="text-sm text-gray-400">Puntuación Histórica</p>
                    <p className="text-cyan-400 text-2xl">{teachersDatabase.find(t => t.id === selectedTeacher)?.baseScore}</p>
                  </div>
                </div>
              </Card>

              <Button
                onClick={handleGeneratePrediction}
                className="w-full bg-blue-600 hover:bg-blue-700"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Generar Predicción
              </Button>
            </motion.div>

            {/* Right Column - Results */}
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
                            <span className="text-5xl text-white">{predictionData.highProb}%</span>
                            <span className="text-gray-400 mb-2">
                              IC 95%: [{predictionData.confidenceInterval[0]}-{predictionData.confidenceInterval[1]}%]
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center justify-center">
                          <Badge className={`${getRiskColor(predictionData.risk)} px-6 py-3 text-lg`}>
                            {predictionData.risk === 'low' ? 'BAJO' : predictionData.risk === 'medium' ? 'MODERADO' : 'ALTO'} RIESGO
                          </Badge>
                        </div>
                      </div>
                      
                      {/* Gauge/Progress Bar */}
                      <div className="mt-6">
                        <div className="h-8 bg-gray-800 rounded-full overflow-hidden relative">
                          <motion.div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-400"
                            initial={{ width: 0 }}
                            animate={{ width: `${predictionData.highProb}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                          />
                          <div
                            className="absolute top-0 h-full w-1 bg-white"
                            style={{ left: `${predictionData.highProb}%` }}
                          />
                        </div>
                        <div className="flex justify-between mt-2 text-sm text-gray-400">
                          <span>0%</span>
                          <span>50%</span>
                          <span>100%</span>
                        </div>
                      </div>

                      <p className="text-gray-300 mt-4 text-center">
                        {predictionData.risk === 'low' 
                          ? 'Excelente rendimiento esperado. Continuar con estrategias actuales.'
                          : predictionData.risk === 'medium'
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
                        <RadarChart data={predictionData.radarData}>
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
                        <BarChart data={predictionData.comparisonData}>
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
                        <LineChart data={predictionData.historicalData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="semester" stroke="#9CA3AF" />
                          <YAxis domain={[60, 100]} stroke="#9CA3AF" />
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
            <h3 className="text-white mb-4">Seleccionar Dataset y Modelo</h3>
            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-2">
                <label className="block text-sm text-gray-400 mb-2">Dataset</label>
                <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    {datasetsAvailable.map(ds => (
                      <SelectItem key={ds.id} value={ds.id}>
                        {ds.name} - {ds.rows} registros ({ds.date})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-1">
                  Seleccione un dataset previamente cargado en la pestaña Datos
                </p>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Modelo</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="dbm">DBM (Champion)</SelectItem>
                    <SelectItem value="rbm">RBM</SelectItem>
                    <SelectItem value="bm">BM Clásica</SelectItem>
                  </SelectContent>
                </Select>
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
                  <p className="text-white text-3xl">{batchResults.length}</p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">Alto Rendimiento</p>
                  <p className="text-green-400 text-3xl">
                    {batchResults.filter(r => r.risk === 'low').length}
                  </p>
                  <p className="text-xs text-gray-500">
                    {((batchResults.filter(r => r.risk === 'low').length / batchResults.length) * 100).toFixed(0)}%
                  </p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">Riesgo Medio</p>
                  <p className="text-yellow-400 text-3xl">
                    {batchResults.filter(r => r.risk === 'medium').length}
                  </p>
                  <p className="text-xs text-gray-500">
                    {((batchResults.filter(r => r.risk === 'medium').length / batchResults.length) * 100).toFixed(0)}%
                  </p>
                </Card>
                <Card className="bg-[#1a1f2e] border-gray-800 p-6">
                  <p className="text-gray-400 text-sm mb-2">En Riesgo</p>
                  <p className="text-red-400 text-3xl">
                    {batchResults.filter(r => r.risk === 'high').length}
                  </p>
                  <p className="text-xs text-gray-500">
                    {((batchResults.filter(r => r.risk === 'high').length / batchResults.length) * 100).toFixed(0)}%
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
                  <h3 className="text-white mb-4">Distribución de Riesgo por Asignatura</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={subjectsDatabase.slice(0, 4).map(s => {
                      const subjectResults = batchResults.filter(r => r.subject === s.name);
                      return {
                        subject: s.name,
                        bajo: subjectResults.filter(r => r.risk === 'low').length,
                        medio: subjectResults.filter(r => r.risk === 'medium').length,
                        alto: subjectResults.filter(r => r.risk === 'high').length,
                      };
                    })}>
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
                  <h3 className="text-white mb-4">Mapa de Calor: Probabilidad de Alto Rendimiento</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-800">
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Docente</th>
                          {subjectsDatabase.slice(0, 4).map(s => (
                            <th key={s.id} className="text-center text-gray-400 text-sm py-3 px-2">{s.code}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(heatmapData).map(([teacher, subjects]: [string, any]) => (
                          <tr key={teacher} className="border-b border-gray-800/50">
                            <td className="text-gray-300 text-sm py-3 px-4">{teacher.split(' ').slice(0, 2).join(' ')}</td>
                            {subjectsDatabase.slice(0, 4).map(s => {
                              const prob = subjects[s.name];
                              const bgColor = prob ? 
                                prob > 0.8 ? 'bg-green-500/80' :
                                prob > 0.7 ? 'bg-green-500/50' :
                                prob > 0.6 ? 'bg-yellow-500/50' :
                                'bg-red-500/50' : 'bg-gray-700';
                              return (
                                <td key={s.id} className="text-center py-3 px-2">
                                  <div className={`${bgColor} rounded px-2 py-1 text-white text-sm`}>
                                    {prob ? `${(prob * 100).toFixed(0)}%` : '-'}
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
                    <div className="flex gap-2">
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
                    </div>
                  </div>
                  <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                    <table className="w-full">
                      <thead className="sticky top-0 bg-[#1a1f2e]">
                        <tr className="border-b border-gray-800">
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Docente</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Asignatura</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Prob. Alto</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Prob. Bajo</th>
                          <th className="text-left text-gray-400 text-sm py-3 px-4">Nivel de Riesgo</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredBatchResults.map((result, index) => {
                          const RiskIcon = getRiskIcon(result.risk);
                          return (
                            <tr key={index} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                              <td className="text-gray-300 text-sm py-3 px-4">{result.teacher}</td>
                              <td className="text-gray-300 text-sm py-3 px-4">{result.subject}</td>
                              <td className="text-gray-300 text-sm py-3 px-4">{(result.probHigh * 100).toFixed(0)}%</td>
                              <td className="text-gray-300 text-sm py-3 px-4">{(result.probLow * 100).toFixed(0)}%</td>
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
