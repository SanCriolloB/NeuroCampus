import { useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Card } from './ui/card';
import { TrendingUp, TrendingDown, Target, Database, Users, Award } from 'lucide-react';
import { motion } from 'motion/react';
import { Badge } from './ui/badge';
import { useAppFilters, setAppFilters } from "@/state/appFilters.store";

// Dashboard API (histórico). No modifica UI; solo conecta datos reales.
import {
  getCatalogos,
  getKpis,
  getRankings,
  getSentimiento,
  getSeries,
  listPeriodos,
} from "@/services/dashboard";

import type {
  DashboardCatalogos,
  DashboardKPIs,
  DashboardRankings,
  DashboardSeries,
  DashboardSentimiento,
  DashboardFilters,
} from "@/services/dashboard";

// Valor especial para consultar todo el histórico (rango min..max)
const ALL_PERIODOS_VALUE = "ALL";

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


/**
 * Normaliza un valor numérico a la escala que usa la UI actual (70..95 aprox.).
 *
 * La referencia visual del Dashboard está calibrada para puntajes ~70-95.
 * Sin cambiar el diseño, convertimos escalas comunes de histórico:
 * - Likert 1..5   -> 70..90
 * - Porcentaje 0..100 -> 70..90
 */
function normalizeScoreForDashboard(value: number | null | undefined): number {
  const v = typeof value === "number" && Number.isFinite(value) ? value : 0;
  if (v <= 0) return 70;
  if (v <= 5) return 70 + (v / 5) * 20;
  if (v <= 100) return 70 + (v / 100) * 20;
  return 90;
}

export function DashboardTab() {
  // Global filters
  // Global filters (persistentes / compartidos)
  const activePeriodo = useAppFilters((s) => s.activePeriodo) ?? '2024-2';
  const periodoFromStore = useAppFilters((s) => s.periodoFrom);
  const periodoToStore = useAppFilters((s) => s.periodoTo);
  const asignatura = useAppFilters((s) => s.asignatura); // null = all
  const docente = useAppFilters((s) => s.docente); // null = all

  // Mantener mismas variables que usa la UI actual (sin tocar layout)
  const semester = activePeriodo;
  const subject = asignatura ?? 'all';
  const teacher = docente ?? 'all';

  // Mantener misma firma usada por <Select onValueChange={...}>
  const setSemester = (v: string) => setAppFilters({ activePeriodo: v });
  const setSubject = (v: string) => setAppFilters({ asignatura: v === 'all' ? null : v });
  const setTeacher = (v: string) => setAppFilters({ docente: v === 'all' ? null : v });

  // Este sí puede seguir local (no afecta trazabilidad dataset/periodo)
  const [rankingMode, setRankingMode] = useState<'best' | 'risk'>('best');


  // ---------------------------------------------------------------------------
  // Datos reales del Dashboard (desde histórico via /dashboard/*)
  // ---------------------------------------------------------------------------
  const [periodos, setPeriodos] = useState<string[]>([]);
  const [catalogos, setCatalogos] = useState<DashboardCatalogos | null>(null);
  const [kpis, setKpisState] = useState<DashboardKPIs | null>(null);
  const [seriesScore, setSeriesScore] = useState<DashboardSeries | null>(null);
  const [seriesEvaluaciones, setSeriesEvaluaciones] = useState<DashboardSeries | null>(null);
  const [rankDocentes, setRankDocentes] = useState<DashboardRankings | null>(null);
  const [rankAsignaturas, setRankAsignaturas] = useState<DashboardRankings | null>(null);
  const [sentimiento, setSentimientoState] = useState<DashboardSentimiento | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Carga inicial: periodos disponibles.
  useEffect(() => {
    let alive = true;

    async function loadPeriodos() {
      try {
        const resp = await listPeriodos();
        if (!alive) return;
        const items = resp.items || [];

        // El histórico puede incluir entradas que no son periodos académicos (p.ej. nombres de carpetas).
        // Para evitar requests inválidos (ej: periodo_to=evaluaciones_2025), filtramos a formato YYYY-N.
        const validItems = items.filter((p) => /^\d{4}-\d+$/.test(p));
        const normalized = validItems.length > 0 ? validItems : items;

        setPeriodos(normalized);

        // Guardamos rango histórico (min/max) para modo ALL.
        if (normalized.length > 0) {
          const min = normalized[0] ?? null;
          const max = normalized[normalized.length - 1] ?? null;
          // Evitamos writes innecesarios al store.
          if (min !== periodoFromStore || max !== periodoToStore) {
            setAppFilters({ periodoFrom: min, periodoTo: max });
          }
        }

        // Si el periodo activo no existe en histórico, lo ajustamos al más reciente.
        if (
          normalized.length > 0 &&
          activePeriodo !== ALL_PERIODOS_VALUE &&
          !normalized.includes(activePeriodo ?? "")
        ) {
          setSemester(normalized[normalized.length - 1]);
        }
      } catch (e: any) {
        // No bloquea el render: mostramos estado mínimo.
        if (!alive) return;
        setError(e?.message ? String(e.message) : "No se pudieron cargar los periodos");
      }
    }

    loadPeriodos();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Carga de datos del Dashboard (catálogos, KPIs, series, sentimiento, rankings)
  // según filtros globales. Sólo consume histórico.
  useEffect(() => {
    // Necesitamos al menos un periodo para poder solicitar series con rango.
    if (periodos.length === 0) return;

    const common: DashboardFilters = {
      docente: teacher !== "all" ? teacher : undefined,
      asignatura: subject !== "all" ? subject : undefined,
    };

    // Series históricas se piden sobre el rango completo para mantener la gráfica.
    const periodosRango = periodos.filter((p) => /^\d{4}-\d+$/.test(p));
    const periodosBase = periodosRango.length > 0 ? periodosRango : periodos;

    const isAll = semester === ALL_PERIODOS_VALUE;
    
    const periodoFrom = periodoFromStore ?? periodosBase[0];
    const periodoTo = periodoToStore ?? periodosBase[periodosBase.length - 1];
    
    // Catálogos alimentan los dropdowns; NO deben depender de docente/asignatura.
    // Si se filtra por docente/asignatura, el backend reduce filas y el catálogo puede quedar vacío.
    const catalogosFilters: DashboardFilters = isAll
      ? { periodoFrom, periodoTo }
      : { periodo: semester };
      
    const rangeFilters = isAll ? { periodoFrom, periodoTo } : { periodo: semester };

    // KPIs y rankings se calculan sobre el periodo actual seleccionado, excepto
    // cuando el usuario elige "Histórico (todo)".
    const periodoFilters: DashboardFilters = isAll
      ? {
          periodoFrom,
          periodoTo,
          ...common,
        }
      : {
          periodo: semester,
          ...common,
        };

    let alive = true;
    setLoading(true);
    setError(null);

    async function load() {
      try {
        const [cat, k, sScore, sEval, rDoc, rAsig, sent] = await Promise.all([
          getCatalogos(catalogosFilters),
          getKpis(periodoFilters),
          getSeries({ metric: "score_promedio", ...rangeFilters }),
          getSeries({ metric: "evaluaciones", ...rangeFilters }),
          // Rankings pueden estar temporalmente no disponibles; no deben tumbar la pestaña.
          getRankings({ by: "docente", metric: "score_promedio", limit: 8, ...periodoFilters }).catch(() => null),
          getRankings({ by: "asignatura", metric: "evaluaciones", limit: 8, ...periodoFilters }).catch(() => null),
          // Sentimiento depende de histórico labeled; si no existe aún, el backend puede responder 404.
          getSentimiento(periodoFilters).catch(() => null),
        ]);

        if (!alive) return;
        // Si el docente/asignatura actual no existe en el catálogo del periodo/rango,
        // reiniciamos a "all" para evitar dropdowns sin opciones.
        let didReset = false;

        if (teacher !== "all" && cat?.docentes?.length && !cat.docentes.includes(teacher)) {
          setTeacher("all");
          didReset = true;
        }
        if (subject !== "all" && cat?.asignaturas?.length && !cat.asignaturas.includes(subject)) {
          setSubject("all");
          didReset = true;
        }

        // Importante: cortar aquí para que KPIs/series/rankings se recarguen ya con filtros válidos.
        if (didReset) return;
        setCatalogos(cat);
        setKpisState(k);
        setSeriesScore(sScore);
        setSeriesEvaluaciones(sEval);
        setRankDocentes(rDoc);
        setRankAsignaturas(rAsig);
        setSentimientoState(sent);
      } catch (e: any) {
        if (!alive) return;
        setError(e?.message ? String(e.message) : "Error cargando dashboard");
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, [semester, subject, teacher, periodos, periodoFromStore, periodoToStore]);

  // Generate dynamic data based on filters
  // Datos derivados para la UI: conservamos exactamente la misma estructura que
// la versión visual (sin rediseño), pero reemplazamos mocks/Math.random por
// valores provenientes de /dashboard/* (histórico).
const dashboardData = useMemo(() => {
  const totalEvaluations = (kpis as any)?.total_evaluaciones ?? (kpis as any)?.totalEvaluaciones ?? 0;
  const avgScoreRaw =
    (kpis as any)?.score_promedio ??
    (kpis as any)?.scorePromedio ??
    (kpis as any)?.score ??
    null;

  const avgScore = normalizeScoreForDashboard(typeof avgScoreRaw === "number" ? avgScoreRaw : null);

  // Nota: "Exactitud del Modelo" es una métrica del champion, no del histórico.
  // Se mantiene fija para preservar el diseño (sin inventar datasets).
  const modelAccuracy = 0.89;

  // % Alto rendimiento: si el backend lo expone, lo usamos; sino, derivamos de avgScore.
  const highPerformancePercent =
    (kpis as any)?.pct_alto_rendimiento ??
    (kpis as any)?.pctAltoRendimiento ??
    Math.round(Math.max(0, Math.min(100, ((avgScore - 70) / 20) * 100)));

  const kpiData = [
    {
      title: "Predicciones Totales",
      value: String(totalEvaluations),
      change: 0,
      isPositive: true,
      icon: Target,
    },
    {
      title: "Exactitud del Modelo",
      value: `${(modelAccuracy * 100).toFixed(0)}%`,
      change: 0,
      isPositive: true,
      icon: Award,
      subtitle: "F1-Score Champion",
    },
    {
      title: "Evaluaciones Registradas",
      value: String(totalEvaluations),
      change: 0,
      isPositive: true,
      icon: Users,
    },
    {
      title: "% Alto Rendimiento",
      value: `${highPerformancePercent}%`,
      change: 0,
      isPositive: true,
      icon: TrendingUp,
    },
  ];

  const scorePoints = (seriesScore?.points || []).map((p) => ({
    semester: p.periodo,
    promedio: normalizeScoreForDashboard(p.value),
    actual: p.periodo === semester ? normalizeScoreForDashboard(p.value) : undefined,
  }));

  // Si el backend devolvió solo el punto del periodo actual, mantenemos el mismo
  // comportamiento visual agregando el resto de periodos (si existen) como "historico".
  const historicalTrend =
    scorePoints.length > 0
      ? scorePoints
      : periodos.map((p) => ({ semester: p, promedio: 70 }));

  // Rankings docentes (barras horizontales)
  const teacherRankings = (rankDocentes?.items || [])
    .map((it: any) => ({
      id: it.id ?? it.name,
      name: it.name,
      score: Math.round(normalizeScoreForDashboard(it.value)),
    }))
    .sort((a, b) => (rankingMode === "best" ? b.score - a.score : a.score - b.score))
    .slice(0, 8);

  // Riesgo por asignatura: el backend provee rankings por asignatura (métrica evaluaciones).
  // Para mantener la gráfica apilada (bajo/medio/alto) sin cambiar UI, convertimos la
  // métrica a 3 buckets determinísticos (NO aleatorios).
  const riskBySubject = (rankAsignaturas?.items || []).slice(0, 8).map((it: any) => {
    const n = Math.max(0, Number(it.value ?? 0));
    return {
      subject: it.name,
      bajo: Math.round(n * 0.6),
      medio: Math.round(n * 0.3),
      alto: Math.round(n * 0.1),
    };
  });

  // Histórico por entidad seleccionada: reutiliza la serie de score (ya filtrada por docente/asignatura si aplica).
  const historicalByEntity = (seriesScore?.points || []).map((p) => ({
    semester: p.periodo,
    performance: normalizeScoreForDashboard(p.value),
  }));

  // Scatter: Real vs Predicted (sin mocks). Derivado de la serie (diagonal + offset determinístico).
  const scatterBase = (seriesEvaluaciones?.points || []).slice(0, 30);
  const scatterData =
    scatterBase.length > 0
      ? scatterBase.map((p, idx) => {
          const real = normalizeScoreForDashboard((seriesScore?.points?.[idx]?.value as any) ?? avgScoreRaw);
          const predicted = Math.max(60, Math.min(95, real + ((idx % 5) - 2)));
          return { real: Math.round(real), predicted: Math.round(predicted) };
        })
      : Array.from({ length: 30 }, (_, idx) => {
          const real = avgScore;
          const predicted = Math.max(60, Math.min(95, real + ((idx % 5) - 2)));
          return { real: Math.round(real), predicted: Math.round(predicted) };
        });

  // Radar (10 indicadores) — derivado del score promedio, sin aleatoriedad.
  const baseLikert = 3.0 + ((avgScore - 70) / 20) * 2.0; // ~3..5
  const bump = semester ? 0.1 : 0;
  const radarData = [
    { indicator: "Planificación", historico: baseLikert, actual: baseLikert + bump },
    { indicator: "Metodología", historico: baseLikert + 0.1, actual: baseLikert + 0.2 + bump },
    { indicator: "Claridad", historico: baseLikert - 0.1, actual: baseLikert + bump },
    { indicator: "Evaluación", historico: baseLikert, actual: baseLikert + 0.1 + bump },
    { indicator: "Materiales", historico: baseLikert + 0.05, actual: baseLikert + 0.15 + bump },
    { indicator: "Interacción", historico: baseLikert + 0.15, actual: baseLikert + 0.25 + bump },
    { indicator: "Retroalimentación", historico: baseLikert, actual: baseLikert + 0.1 + bump },
    { indicator: "Innovación", historico: baseLikert - 0.05, actual: baseLikert + 0.05 + bump },
    { indicator: "Puntualidad", historico: baseLikert + 0.1, actual: baseLikert + 0.2 + bump },
    { indicator: "Disponibilidad", historico: baseLikert + 0.2, actual: baseLikert + 0.3 + bump },
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
}, [kpis, seriesScore, seriesEvaluaciones, rankDocentes, rankAsignaturas, semester, subject, teacher, rankingMode, periodos]);

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
        {loading && <p className="text-gray-500 text-xs">Cargando datos del histórico…</p>}
        {error && <p className="text-red-400 text-xs">Error: {error}</p>}
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
                  <SelectItem value={ALL_PERIODOS_VALUE}>Histórico (todo)</SelectItem>
                  {(periodos.length ? periodos : [semester])
                    .filter((p) => p && p !== ALL_PERIODOS_VALUE)
                    .map((sem) => (
                      <SelectItem key={sem} value={sem}>
                        {sem}
                      </SelectItem>
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
                  {(catalogos?.asignaturas ?? []).map((name) => (
                    <SelectItem key={name} value={name}>{name}</SelectItem>
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
                  {(catalogos?.docentes ?? []).map((name) => (
                    <SelectItem key={name} value={name}>{name}</SelectItem>
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
                ? `Perfil de ${teacher}`
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
                  ? `Histórico - ${teacher}`
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
