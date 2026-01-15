import { useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Search, X } from 'lucide-react';

// Colors matching existing DataTab
const COLORS = {
  positive: '#10B981',
  neutral: '#6B7280',
  negative: '#EF4444',
};


type TeacherSentiment = {
  teacher: string;
  pos: number;
  neu: number;
  neg: number;
  total: number;
};

type SortKey = 'total' | 'negPercentage' | 'posPercentage' | 'netScore';
type DisplayMode = 'percentage' | 'count';

interface TeacherSentimentChartProps {
  /** Data REAL (mapeada desde /datos/sentimientos). Nunca uses mock en producción */
  data: TeacherSentiment[];

  /** Título del card (para calcar la maqueta o reutilizarlo) */
  title?: string;

  /** Estado de carga (por ejemplo mientras corre BETO o se hace refetch) */
  isLoading?: boolean;

  /** Mensaje de error (si la petición falla) */
  error?: string | null;

  /**
   * Para resetear estados internos cuando cambias dataset (ej: "2024-2").
   * Pásale datasetForQueries o activeDatasetId.
   */
  resetKey?: string;

  initialVisibleCount?: number;
  loadMoreStep?: number;
}

export default function TeacherSentimentChart({
  data,
  title = 'Sentiment Distribution by Teacher',
  isLoading = false,
  error = null,
  resetKey,
  initialVisibleCount = 10,
  loadMoreStep = 10,
}: TeacherSentimentChartProps) {
  
  const [visibleCount, setVisibleCount] = useState(initialVisibleCount);
  const [query, setQuery] = useState('');
  const [selectedTeachers, setSelectedTeachers] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>('total');
  const [displayMode, setDisplayMode] = useState<DisplayMode>('percentage');
  const [showSuggestions, setShowSuggestions] = useState(false);

  useEffect(() => {
    if (!resetKey) return;

    // Reset de UI para evitar “se queda pegado” al cambiar dataset
    setVisibleCount(initialVisibleCount);
    setQuery('');
    setSelectedTeachers([]);
    setSortKey('total');
    setDisplayMode('percentage');
    setShowSuggestions(false);
  }, [resetKey, initialVisibleCount]);

  // Normalize and sort data
  const sortedData = useMemo(() => {
    const normalized = data.map(t => ({
      ...t,
      total: t.total || (t.pos + t.neu + t.neg),
    }));

    return [...normalized].sort((a, b) => {
      switch (sortKey) {
        case 'total':
          return b.total - a.total;
        case 'negPercentage': {
          const bn = b.total ? (b.neg / b.total) : 0;
          const an = a.total ? (a.neg / a.total) : 0;
          return bn - an;
        }
        case 'posPercentage': {
          const bp = b.total ? (b.pos / b.total) : 0;
          const ap = a.total ? (a.pos / a.total) : 0;
          return bp - ap;
        }
        case 'netScore': {
          const bs = b.total ? ((b.pos - b.neg) / b.total) : 0;
          const as = a.total ? ((a.pos - a.neg) / a.total) : 0;
          return bs - as;
        }
        default:
          return 0;
      }
    });
  }, [data, sortKey]);

  // Search suggestions
  
  const suggestions = useMemo(() => {
    if (!query.trim()) return [];
    const lowerQuery = query.toLowerCase().trim();
    const norm = (s: string) =>
    s.toLowerCase().normalize('NFD').replace(/\p{Diacritic}/gu, '');
    return sortedData
      .filter(t => norm(t.teacher).includes(lowerQuery))
      .slice(0, 10);
  }, [query, sortedData]);

  // Chart data (either selected teachers or top N)
  const chartData = useMemo(() => {
    const isCompareMode = selectedTeachers.length > 0;
    
    if (isCompareMode) {
      return sortedData.filter(t => selectedTeachers.includes(t.teacher));
    }
    
    return sortedData.slice(0, visibleCount);
  }, [sortedData, selectedTeachers, visibleCount]);

  // Transformed data for recharts based on display mode
  const transformedChartData = useMemo(() => {
    return chartData.map(t => {
      if (displayMode === 'percentage') {
        return {
          teacher: t.teacher.length > 15 ? t.teacher.substring(0, 15) + '...' : t.teacher,
          fullName: t.teacher,
          Positive: t.total ? (t.pos / t.total) * 100 : 0,
          Neutral: t.total ? (t.neu / t.total) * 100 : 0,
          Negative: t.total ? (t.neg / t.total) * 100 : 0,
          posCount: t.pos,
          neuCount: t.neu,
          negCount: t.neg,
          total: t.total,
        };
      } else {
        return {
          teacher: t.teacher.length > 15 ? t.teacher.substring(0, 15) + '...' : t.teacher,
          fullName: t.teacher,
          Positive: t.pos,
          Neutral: t.neu,
          Negative: t.neg,
          total: t.total,
        };
      }
    });
  }, [chartData, displayMode]);

  const handleSelectTeacher = (teacher: string) => {
    if (selectedTeachers.includes(teacher)) return;
    
    if (selectedTeachers.length >= 5) {
      // Could show a toast here
      return;
    }
    
    setSelectedTeachers([...selectedTeachers, teacher]);
    setQuery('');
    setShowSuggestions(false);
  };

  const handleRemoveTeacher = (teacher: string) => {
    setSelectedTeachers(selectedTeachers.filter(t => t !== teacher));
  };

  const handleClearAll = () => {
    setSelectedTeachers([]);
    setVisibleCount(initialVisibleCount);
  };

  const handleLoadMore = () => {
    setVisibleCount(prev => Math.min(prev + loadMoreStep, sortedData.length));
  };

  const isCompareMode = selectedTeachers.length > 0;
  const canLoadMore = !isCompareMode && visibleCount < sortedData.length;

  // Subtitle text
  const subtitleText = isCompareMode 
    ? `Comparando ${selectedTeachers.length} profesor${selectedTeachers.length > 1 ? 'es' : ''}`
    : `Top ${visibleCount} por comentarios`;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-[#1a1f2e] border border-gray-700 p-3 rounded-lg shadow-lg">
        <p className="text-white font-medium mb-2">{data.fullName}</p>
        <div className="space-y-1 text-sm">
          {displayMode === 'percentage' ? (
            <>
              <p className="text-green-400">Positivo: {data.posCount} ({data.Positive.toFixed(1)}%)</p>
              <p className="text-gray-400">Neutral: {data.neuCount} ({data.Neutral.toFixed(1)}%)</p>
              <p className="text-red-400">Negativo: {data.negCount} ({data.Negative.toFixed(1)}%)</p>
            </>
          ) : (
            <>
              <p className="text-green-400">Positivo: {data.Positive}</p>
              <p className="text-gray-400">Neutral: {data.Neutral}</p>
              <p className="text-red-400">Negativo: {data.Negative}</p>
            </>
          )}
          <p className="text-gray-300 pt-1 border-t border-gray-700">Total: {data.total}</p>
        </div>
      </div>
    );
  };

  return (
    <Card className="bg-[#1a1f2e] border-gray-800 p-6 col-span-2">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="text-white mb-1">{title}</h4>
          <p className="text-sm text-gray-400">{subtitleText}</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Buscar profesor..."
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setShowSuggestions(true);
                }}
                onFocus={() => setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                className="bg-[#0f1419] border-gray-700 pl-9 w-48 h-9 text-sm"
              />
            </div>
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute top-full mt-1 w-full bg-[#1a1f2e] border border-gray-700 rounded-lg shadow-lg z-10 max-h-60 overflow-y-auto">
                {suggestions.map((teacher) => (
                  <button
                    key={teacher.teacher}
                    onClick={() => handleSelectTeacher(teacher.teacher)}
                    className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 transition-colors"
                    disabled={selectedTeachers.includes(teacher.teacher)}
                  >
                    <div className="flex items-center justify-between">
                      <span className={selectedTeachers.includes(teacher.teacher) ? 'text-gray-500' : ''}>
                        {teacher.teacher}
                      </span>
                      <span className="text-xs text-gray-500">{teacher.total} comentarios</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
          
          {/* Sort */}
          <Select value={sortKey} onValueChange={(value) => setSortKey(value as SortKey)}>
            <SelectTrigger className="bg-[#0f1419] border-gray-700 w-40 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1f2e] border-gray-700">
              <SelectItem value="total">Comentarios (total)</SelectItem>
              <SelectItem value="negPercentage">% Negativo</SelectItem>
              <SelectItem value="posPercentage">% Positivo</SelectItem>
              <SelectItem value="netScore">Score neto</SelectItem>
            </SelectContent>
          </Select>

          {/* Display Mode Toggle */}
          <div className="flex bg-[#0f1419] border border-gray-700 rounded-lg overflow-hidden h-9">
            <button
              onClick={() => setDisplayMode('percentage')}
              className={`px-3 text-sm transition-colors ${
                displayMode === 'percentage' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              %
            </button>
            <button
              onClick={() => setDisplayMode('count')}
              className={`px-3 text-sm transition-colors ${
                displayMode === 'count' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              #
            </button>
          </div>
        </div>
      </div>

      {/* Comparison chips */}
      {isCompareMode && (
        <div className="flex flex-wrap items-center gap-2 mb-4 pb-4 border-b border-gray-800">
          {selectedTeachers.map((teacher) => (
            <Badge
              key={teacher}
              variant="secondary"
              className="bg-gray-800 text-gray-300 pl-3 pr-2 py-1 gap-1"
            >
              <span className="max-w-32 truncate">{teacher}</span>
              <button
                onClick={() => handleRemoveTeacher(teacher)}
                className="hover:bg-gray-700 rounded p-0.5 transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </Badge>
          ))}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearAll}
            className="h-7 text-xs text-gray-400 hover:text-gray-300"
          >
            Limpiar
          </Button>
          {selectedTeachers.length >= 5 && (
            <span className="text-xs text-amber-400">Máximo 5 para mantener legibilidad</span>
          )}
        </div>
      )}

      {/* Chart */}
      {/* Estados (loading / error / empty) */}
      {isLoading ? (
        <div className="h-[350px] flex items-center justify-center text-sm text-gray-400">
          Procesando sentimientos…
        </div>
      ) : error ? (
        <div className="h-[350px] flex items-center justify-center text-sm text-red-300">
          No se pudo cargar sentimientos: {error}
        </div>
      ) : transformedChartData.length === 0 ? (
        <div className="h-[350px] flex items-center justify-center text-sm text-gray-400">
          No hay datos por profesor para este periodo.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={transformedChartData}>
            ...
          </BarChart>
        </ResponsiveContainer>
      )}
      {/* Footer */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-800">
        <p className="text-sm text-gray-400">
          Mostrando {chartData.length} de {sortedData.length} profesores
        </p>
        {canLoadMore && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleLoadMore}
            className="bg-transparent border-gray-700 text-gray-300 hover:bg-gray-800"
          >
            Cargar más
          </Button>
        )}
      </div>
    </Card>
  );
}
