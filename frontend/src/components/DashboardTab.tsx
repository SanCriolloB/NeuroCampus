import { useState, useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Card } from './ui/card';
import { TrendingUp, TrendingDown, Target, Database, Users, Award } from 'lucide-react';
import { motion } from 'motion/react';
import { Badge } from './ui/badge';

// Base data structure
const teachersData = {
  'garcia': { name: 'Dr. María García', baseScore: 92 },
  'martinez': { name: 'Prof. Juan Martínez', baseScore: 88 },
  'lopez': { name: 'Dr. Ana López', baseScore: 85 },
  'rodriguez': { name: 'Prof. Carlos Rodríguez', baseScore: 82 },
  'fernandez': { name: 'Dr. Laura Fernández', baseScore: 78 },
  'santos': { name: 'Dra. Patricia Santos', baseScore: 90 },
  'torres': { name: 'Prof. Miguel Torres', baseScore: 84 },
  'ramirez': { name: 'Dr. Roberto Ramírez', baseScore: 87 },
};

const subjectsData = {
  'calculus': { name: 'Cálculo I', lowRisk: 45, mediumRisk: 30, highRisk: 15 },
  'physics': { name: 'Física II', lowRisk: 40, mediumRisk: 35, highRisk: 20 },
  'programming': { name: 'Programación Avanzada', lowRisk: 55, mediumRisk: 25, highRisk: 10 },
  'organic_chem': { name: 'Química Orgánica', lowRisk: 38, mediumRisk: 32, highRisk: 25 },
  'inorganic_chem': { name: 'Química Inorgánica', lowRisk: 42, mediumRisk: 30, highRisk: 22 },
  'mathematics': { name: 'Matemáticas Discretas', lowRisk: 48, mediumRisk: 28, highRisk: 18 },
  'statistics': { name: 'Estadística Aplicada', lowRisk: 50, mediumRisk: 30, highRisk: 15 },
};

const semestersData = ['2023-1', '2023-2', '2024-1', '2024-2', '2025-1'];
const programsData = ['all', 'engineering', 'sciences', 'mathematics'];

// Word cloud data
const wordCloudData = [
  { word: 'excelente', count: 145, sentiment: 'positive' },
  { word: 'claridad', count: 132, sentiment: 'positive' },
  { word: 'metodología', count: 98, sentiment: 'neutral' },
  { word: 'puntual', count: 87, sentiment: 'positive' },
  { word: 'difícil', count: 76, sentiment: 'negative' },
  { word: 'apoyo', count: 112, sentiment: 'positive' },
  { word: 'evaluación', count: 91, sentiment: 'neutral' },
  { word: 'recursos', count: 68, sentiment: 'neutral' },
  { word: 'disponible', count: 103, sentiment: 'positive' },
  { word: 'confuso', count: 45, sentiment: 'negative' },
  { word: 'innovador', count: 72, sentiment: 'positive' },
  { word: 'dinámico', count: 64, sentiment: 'positive' },
  { word: 'carga', count: 53, sentiment: 'negative' },
  { word: 'retroalimentación', count: 89, sentiment: 'positive' },
  { word: 'interactivo', count: 77, sentiment: 'positive' },
];

export function DashboardTab() {
  // Global filters
  const [semester, setSemester] = useState('2025-1');
  const [subject, setSubject] = useState('all');
  const [teacher, setTeacher] = useState('all');
  const [rankingMode, setRankingMode] = useState<'best' | 'risk'>('best');

  // Generate dynamic data based on filters
  const dashboardData = useMemo(() => {
    // KPI Data
    const totalPredictions = 1250;
    const modelAccuracy = 0.89;
    const totalEvaluations = 5000;
    const highPerformancePercent = 68;

    const kpiData = [
      { 
        title: 'Predicciones Totales', 
        value: totalPredictions.toString(), 
        change: 12, 
        isPositive: true,
        icon: Target,
      },
      { 
        title: 'Exactitud del Modelo', 
        value: `${(modelAccuracy * 100).toFixed(0)}%`, 
        change: 3, 
        isPositive: true,
        icon: Award,
        subtitle: 'F1-Score Champion',
      },
      { 
        title: 'Evaluaciones Registradas', 
        value: totalEvaluations.toString(), 
        change: 15, 
        isPositive: true,
        icon: Users,
      },
      { 
        title: '% Alto Rendimiento', 
        value: `${highPerformancePercent}%`, 
        change: 5, 
        isPositive: true,
        icon: TrendingUp,
      },
    ];

    // Historical Performance by semester
    const semesterIndex = semestersData.indexOf(semester);
    const historicalTrend = semestersData.slice(0, semesterIndex + 1).map((sem, idx) => ({
      semester: sem,
      promedio: 75 + idx * 2 + Math.random() * 3,
      actual: idx === semesterIndex ? 82 : undefined,
    }));

    // Risk by subject
    const riskBySubject = Object.values(subjectsData).map(s => {
      const multiplier = subject === 'all' || subjectsData[subject as keyof typeof subjectsData]?.name === s.name ? 1 : 0.7;
      return {
        subject: s.name,
        bajo: Math.round(s.lowRisk * multiplier),
        medio: Math.round(s.mediumRisk * multiplier),
        alto: Math.round(s.highRisk * multiplier),
      };
    });

    // Teacher rankings
    const teacherRankings = Object.entries(teachersData)
      .map(([id, data]) => ({
        id,
        name: data.name,
        score: subject === 'all' ? data.baseScore : Math.round(data.baseScore * (0.95 + Math.random() * 0.1)),
      }))
      .sort((a, b) => rankingMode === 'best' ? b.score - a.score : a.score - b.score)
      .slice(0, 8);

    // Historical by teacher/subject (for line chart)
    const selectedTeacherData = teacher !== 'all' ? teachersData[teacher as keyof typeof teachersData] : null;
    const historicalByEntity = semestersData.map((sem, idx) => {
      const base = selectedTeacherData ? selectedTeacherData.baseScore : 80;
      return {
        semester: sem,
        performance: Math.round(base - (semestersData.length - idx - 1) * 2 + Math.random() * 3),
      };
    });

    // Scatter data - Real vs Predicted
    const scatterData = Array.from({ length: 30 }, (_, i) => {
      const real = 60 + Math.random() * 35;
      const predicted = real + (Math.random() * 10 - 5);
      return { real: Math.round(real), predicted: Math.round(predicted) };
    });

    // Radar data - 10 indicators comparing historical average vs current semester
    const selectedTeacherScore = selectedTeacherData ? selectedTeacherData.baseScore : 85;
    const radarData = [
      { 
        indicator: 'Planificación', 
        historico: 3.5 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.6 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Metodología', 
        historico: 3.8 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.9 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Claridad', 
        historico: 3.2 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.3 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Evaluación', 
        historico: 3.4 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.5 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Materiales', 
        historico: 3.6 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.7 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Interacción', 
        historico: 4.0 + (selectedTeacherScore / 100) * 1.3, 
        actual: 4.1 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Retroalimentación', 
        historico: 3.5 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.6 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Innovación', 
        historico: 3.3 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.4 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Puntualidad', 
        historico: 3.7 + (selectedTeacherScore / 100) * 1.3, 
        actual: 3.8 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
      { 
        indicator: 'Disponibilidad', 
        historico: 3.9 + (selectedTeacherScore / 100) * 1.3, 
        actual: 4.0 + (selectedTeacherScore / 100) * 1.4 + (semesterIndex * 0.05)
      },
    ];

    return {
      kpiData,
      historicalTrend,
      riskBySubject,
      teacherRankings,
      historicalByEntity,
      scatterData,
      radarData,
    };
  }, [semester, subject, teacher, rankingMode]);

  // Calculate word sizes for word cloud
  const maxCount = Math.max(...wordCloudData.map(w => w.count));
  const getWordSize = (count: number) => 12 + (count / maxCount) * 24;
  const getWordColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return '#10B981';
      case 'negative': return '#EF4444';
      default: return '#9CA3AF';
    }
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h2 className="text-white mb-2">Dashboard</h2>
        <p className="text-gray-400">Diagnóstico General de la Institución</p>
      </motion.div>

      {/* Global Filters Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <Card className="bg-[#1a1f2e] border-gray-800 p-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Semestre / Periodo</label>
              <Select value={semester} onValueChange={setSemester}>
                <SelectTrigger className="bg-[#0f1419] border-gray-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1f2e] border-gray-700">
                  {semestersData.map(sem => (
                    <SelectItem key={sem} value={sem}>{sem}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Asignatura</label>
              <Select value={subject} onValueChange={setSubject}>
                <SelectTrigger className="bg-[#0f1419] border-gray-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1f2e] border-gray-700">
                  <SelectItem value="all">Todas las Asignaturas</SelectItem>
                  {Object.entries(subjectsData).map(([id, data]) => (
                    <SelectItem key={id} value={id}>{data.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Docente</label>
              <Select value={teacher} onValueChange={setTeacher}>
                <SelectTrigger className="bg-[#0f1419] border-gray-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1f2e] border-gray-700">
                  <SelectItem value="all">Todos los Docentes</SelectItem>
                  {Object.entries(teachersData).map(([id, data]) => (
                    <SelectItem key={id} value={id}>{data.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* KPIs - 4 cards */}
      <div className="grid grid-cols-4 gap-4">
        {dashboardData.kpiData.map((kpi, index) => {
          const Icon = kpi.icon;
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <Card className="bg-[#1a1f2e] border-gray-800 p-6 hover:bg-[#1f2937] transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <p className="text-gray-400 text-sm">{kpi.title}</p>
                  <Icon className="w-5 h-5 text-cyan-400" />
                </div>
                <div className="flex items-end justify-between">
                  <div>
                    <span className="text-white text-2xl block">{kpi.value}</span>
                    {kpi.subtitle && (
                      <span className="text-xs text-gray-500">{kpi.subtitle}</span>
                    )}
                  </div>
                  {kpi.change !== 0 && (
                    <div className={`flex items-center gap-1 text-sm ${kpi.isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {kpi.isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>{Math.abs(kpi.change)}%</span>
                    </div>
                  )}
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Section: ¿Cómo estamos ahora? */}
      <div>
        <h3 className="text-white mb-4 text-lg">¿Cómo estamos ahora? - Vista Transversal de Riesgo</h3>
        <div className="grid grid-cols-2 gap-6">
          {/* Risk by Subject */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Distribución de Riesgo por Asignatura</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={dashboardData.riskBySubject}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="subject" stroke="#9CA3AF" angle={-15} textAnchor="end" height={80} />
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

          {/* Teacher Rankings */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white">Ranking de Docentes</h3>
                <div className="flex gap-2">
                  <Badge 
                    className={`cursor-pointer ${rankingMode === 'best' ? 'bg-blue-600' : 'bg-gray-700'}`}
                    onClick={() => setRankingMode('best')}
                  >
                    Top Mejores
                  </Badge>
                  <Badge 
                    className={`cursor-pointer ${rankingMode === 'risk' ? 'bg-orange-600' : 'bg-gray-700'}`}
                    onClick={() => setRankingMode('risk')}
                  >
                    A Intervenir
                  </Badge>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={dashboardData.teacherRankings} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9CA3AF" domain={[0, 100]} />
                  <YAxis type="category" dataKey="name" stroke="#9CA3AF" width={150} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                    {dashboardData.teacherRankings.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`}
                        fill={rankingMode === 'best' 
                          ? entry.score > 85 ? '#3B82F6' : '#6B7280'
                          : entry.score < 80 ? '#EF4444' : '#F59E0B'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>
        </div>
      </div>

      {/* Section: Análisis de Indicadores */}
      <div>
        <h3 className="text-white mb-4 text-lg">Análisis de Indicadores - Comparación Histórica vs Semestre Actual</h3>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">
              {teacher !== 'all' 
                ? `Perfil de ${teachersData[teacher as keyof typeof teachersData]?.name}`
                : 'Perfil Global de Indicadores'}
            </h3>
            <ResponsiveContainer width="100%" height={450}>
              <RadarChart data={dashboardData.radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="indicator" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis angle={90} domain={[0, 5]} stroke="#9CA3AF" />
                <Radar 
                  name="Promedio Histórico (Todos los Semestres)" 
                  dataKey="historico" 
                  stroke="#6B7280" 
                  fill="#6B7280" 
                  fillOpacity={0.3} 
                />
                <Radar 
                  name={`Semestre ${semester}`} 
                  dataKey="actual" 
                  stroke="#3B82F6" 
                  fill="#3B82F6" 
                  fillOpacity={0.6} 
                />
                <Legend />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                />
              </RadarChart>
            </ResponsiveContainer>
            <p className="text-gray-400 text-sm mt-2 text-center">
              Comparación del desempeño promedio histórico vs el semestre seleccionado
            </p>
          </Card>
        </motion.div>
      </div>

      {/* Section: ¿Cómo hemos cambiado? */}
      <div>
        <h3 className="text-white mb-4 text-lg">¿Cómo hemos cambiado? - Vista Temporal</h3>
        <div className="grid grid-cols-2 gap-6">
          {/* Historical by Entity */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">
                {teacher !== 'all' 
                  ? `Histórico - ${teachersData[teacher as keyof typeof teachersData]?.name}`
                  : 'Histórico por Entidad Seleccionada'}
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData.historicalByEntity}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="semester" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[70, 95]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="performance" 
                    stroke="#06B6D4" 
                    strokeWidth={3} 
                    name="Desempeño"
                    dot={{ fill: '#06B6D4', r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>

          {/* Historical Average vs Current */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Promedio Histórico vs Semestre Actual</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData.historicalTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="semester" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[70, 90]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="promedio" 
                    stroke="#6B7280" 
                    strokeWidth={2} 
                    name="Promedio Histórico"
                    strokeDasharray="5 5"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#10B981" 
                    strokeWidth={3} 
                    name="Semestre Actual"
                    dot={{ fill: '#10B981', r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>
        </div>
      </div>

      {/* Section: ¿Qué tan bien predice el modelo? */}
      <div>
        <h3 className="text-white mb-4 text-lg">¿Qué tan bien predice el modelo? - Calidad del Modelo</h3>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Real vs Predicho - Scatter Plot</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  type="number" 
                  dataKey="real" 
                  name="Real" 
                  stroke="#9CA3AF"
                  label={{ value: 'Desempeño Real', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                />
                <YAxis 
                  type="number" 
                  dataKey="predicted" 
                  name="Predicho" 
                  stroke="#9CA3AF"
                  label={{ value: 'Predicción', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                  labelStyle={{ color: '#fff' }}
                  cursor={{ strokeDasharray: '3 3' }}
                />
                <Legend />
                <Scatter 
                  name="Predicciones" 
                  data={dashboardData.scatterData} 
                  fill="#06B6D4"
                  opacity={0.7}
                />
                {/* Reference line y=x */}
                <Line 
                  type="linear" 
                  dataKey="real" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  strokeDasharray="5 5" 
                  dot={false}
                  name="Línea Perfecta (y=x)"
                />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-gray-400 text-sm mt-2 text-center">
              Cuanto más cerca de la línea naranja, mejor es la predicción del modelo
            </p>
          </Card>
        </motion.div>
      </div>

      {/* Section: Contexto Cualitativo - Word Cloud */}
      <div>
        <h3 className="text-white mb-4 text-lg">Contexto Cualitativo - Tendencias en Comentarios</h3>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Nube de Palabras - Análisis de Sentimientos</h3>
            <div className="flex flex-wrap gap-3 justify-center items-center min-h-[300px] p-8">
              {wordCloudData.map((item, index) => (
                <motion.span
                  key={index}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.05 }}
                  style={{
                    fontSize: `${getWordSize(item.count)}px`,
                    color: getWordColor(item.sentiment),
                    fontWeight: item.count > 100 ? 'bold' : 'normal',
                  }}
                  className="cursor-pointer hover:opacity-70 transition-opacity"
                  title={`${item.word}: ${item.count} menciones (${item.sentiment})`}
                >
                  {item.word}
                </motion.span>
              ))}
            </div>
            <div className="flex justify-center gap-6 mt-4 pt-4 border-t border-gray-700">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm text-gray-400">Positivo</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-gray-400 rounded"></div>
                <span className="text-sm text-gray-400">Neutral</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm text-gray-400">Negativo</span>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
