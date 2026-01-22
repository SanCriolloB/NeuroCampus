import { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge } from "./ui/badge";
import { Search, X } from "lucide-react";

// Mantén relación con colores existentes
const COLORS = {
  positive: "#10B981",
  neutral: "#6B7280",
  negative: "#EF4444",
};

export type TeacherSentiment = {
  teacher: string;
  pos: number;
  neu: number;
  neg: number;
  total: number;
};

type SortKey = "total" | "negPercentage" | "posPercentage" | "netScore";
type DisplayMode = "percentage" | "count";

export interface TeacherSentimentChartProps {
  /** Data real (mapeada desde /datos/sentimientos) */
  data: TeacherSentiment[];

  title?: string;

  /** loading del hook (beto/refetch) */
  isLoading?: boolean;

  /** error del hook */
  error?: string | null;

  /**
   * Para resetear UI cuando cambia dataset (ej: "2024-2" -> "2025-1")
   * Pasa: datasetForQueries o activeDatasetId.
   */
  resetKey?: string;

  initialVisibleCount?: number;
  loadMoreStep?: number;

  /** máximo en comparación */
  maxCompare?: number;
}

type Row = TeacherSentiment & {
  posPct: number;
  neuPct: number;
  negPct: number;
};

function normalizeForSearch(s: string) {
  return (s ?? "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

export default function TeacherSentimentChart({
  data,
  title = "Sentiment Distribution by Teacher",
  isLoading = false,
  error = null,
  resetKey,
  initialVisibleCount = 10,
  loadMoreStep = 10,
  maxCompare = 5,
}: TeacherSentimentChartProps) {
  const [visibleCount, setVisibleCount] = useState(initialVisibleCount);
  const [query, setQuery] = useState("");
  const [selectedTeachers, setSelectedTeachers] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("total");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("percentage");
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Reset UI cuando cambias dataset
  useEffect(() => {
    if (!resetKey) return;
    setVisibleCount(initialVisibleCount);
    setQuery("");
    setSelectedTeachers([]);
    setSortKey("total");
    setDisplayMode("percentage");
    setShowSuggestions(false);
  }, [resetKey, initialVisibleCount]);

  // Normaliza + ordena
  const sortedData = useMemo<TeacherSentiment[]>(() => {
    const normalized: TeacherSentiment[] = (data ?? [])
      .map((t) => {
        const pos = Number((t as any).pos ?? (t as any).positive ?? 0);
        const neu = Number((t as any).neu ?? (t as any).neutral ?? 0);
        const neg = Number((t as any).neg ?? (t as any).negative ?? 0);
        const total = Number((t as any).total ?? pos + neu + neg) || 0;

        return {
          teacher: String((t as any).teacher ?? ""),
          pos,
          neu,
          neg,
          total,
        };
      })
      .filter((t) => t.teacher.trim().length > 0);

    const safeDiv = (a: number, b: number) => (b > 0 ? a / b : 0);

    return [...normalized].sort((a, b) => {
      switch (sortKey) {
        case "total":
          return b.total - a.total;
        case "negPercentage":
          return safeDiv(b.neg, b.total) - safeDiv(a.neg, a.total);
        case "posPercentage":
          return safeDiv(b.pos, b.total) - safeDiv(a.pos, a.total);
        case "netScore":
          return safeDiv(b.pos - b.neg, b.total) - safeDiv(a.pos - a.neg, a.total);
        default:
          return 0;
      }
    });
  }, [data, sortKey]);

  // Suggestions (top 10)
  const suggestions = useMemo(() => {
    const q = normalizeForSearch(query);
    if (!q) return [];
    return sortedData
      .filter((t) => normalizeForSearch(t.teacher).includes(q))
      .slice(0, 10);
  }, [query, sortedData]);

  const isCompareMode = selectedTeachers.length > 0;

  // Datos a graficar: o comparados (en orden de selección) o top N
  const chartData = useMemo<TeacherSentiment[]>(() => {
    if (isCompareMode) {
      const map = new Map(sortedData.map((t) => [t.teacher, t]));
      return selectedTeachers.map((name) => map.get(name)).filter(Boolean) as TeacherSentiment[];
    }
    return sortedData.slice(0, visibleCount);
  }, [sortedData, selectedTeachers, visibleCount, isCompareMode]);

  // % para modo porcentaje
  const transformedChartData = useMemo<Row[]>(() => {
    return chartData.map((t) => {
      const total = t.total || 0;
      const posPct = total ? (t.pos / total) * 100 : 0;
      const neuPct = total ? (t.neu / total) * 100 : 0;
      const negPct = total ? (t.neg / total) * 100 : 0;
      return { ...t, posPct, neuPct, negPct };
    });
  }, [chartData]);

  const teacherShort = (name: string) => (name.length > 28 ? `${name.slice(0, 28)}…` : name);

  const handleSelectTeacher = (teacher: string) => {
    setSelectedTeachers((prev) => {
      if (prev.includes(teacher)) return prev;
      if (prev.length >= maxCompare) return prev;
      return [...prev, teacher];
    });
    setQuery("");
    setShowSuggestions(false);
  };

  const handleRemoveTeacher = (teacher: string) => {
    setSelectedTeachers((prev) => prev.filter((t) => t !== teacher));
  };

  const handleClearAll = () => {
    setSelectedTeachers([]);
    setVisibleCount(initialVisibleCount);
  };

  const handleLoadMore = () => {
    setVisibleCount((prev) => Math.min(prev + loadMoreStep, sortedData.length));
  };

  const canLoadMore = !isCompareMode && visibleCount < sortedData.length;

  const subtitleText = isCompareMode
    ? `Comparando ${selectedTeachers.length} profesor${selectedTeachers.length > 1 ? "es" : ""}`
    : `Mostrando top ${Math.min(visibleCount, sortedData.length)} por comentarios`;

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const row: Row = payload[0]?.payload;
    if (!row) return null;

    const showPct = displayMode === "percentage";

    return (
      <div className="bg-[#1a1f2e] border border-gray-700 p-3 rounded-lg shadow-lg">
        <p className="text-white font-medium mb-2">{row.teacher}</p>
        <div className="space-y-1 text-sm">
          <p className="text-green-400">
            Positivo: {row.pos}
            {showPct ? ` (${row.posPct.toFixed(1)}%)` : ""}
          </p>
          <p className="text-gray-400">
            Neutral: {row.neu}
            {showPct ? ` (${row.neuPct.toFixed(1)}%)` : ""}
          </p>
          <p className="text-red-400">
            Negativo: {row.neg}
            {showPct ? ` (${row.negPct.toFixed(1)}%)` : ""}
          </p>
          <p className="text-gray-300 pt-1 border-t border-gray-700">Total: {row.total}</p>
        </div>
      </div>
    );
  };

  const posKey = displayMode === "percentage" ? "posPct" : "pos";
  const neuKey = displayMode === "percentage" ? "neuPct" : "neu";
  const negKey = displayMode === "percentage" ? "negPct" : "neg";

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
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            <Input
              placeholder="Buscar profesor..."
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setShowSuggestions(true);
              }}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setShowSuggestions(false)}
              className="bg-[#0f1419] border-gray-700 pl-9 w-56 h-9 text-sm"
            />

            {/* IMPORTANTE: z-50 + onMouseDown para que NO se bloquee la selección */}
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute top-full mt-1 w-full bg-[#1a1f2e] border border-gray-700 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                {suggestions.map((t) => {
                  const disabled =
                    selectedTeachers.includes(t.teacher) || selectedTeachers.length >= maxCompare;

                  return (
                    <button
                      key={t.teacher}
                      type="button"
                      onMouseDown={(e) => {
                        e.preventDefault(); // evita blur antes del click
                        if (!disabled) handleSelectTeacher(t.teacher);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                        disabled ? "text-gray-500" : "text-gray-300 hover:bg-gray-800"
                      }`}
                      disabled={disabled}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="truncate">{t.teacher}</span>
                        <span className="text-xs text-gray-500 shrink-0">{t.total} comentarios</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Sort */}
          <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
            <SelectTrigger className="bg-[#0f1419] border-gray-700 w-44 h-9 text-sm">
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
              type="button"
              onClick={() => setDisplayMode("percentage")}
              className={`px-3 text-sm transition-colors ${
                displayMode === "percentage"
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:text-gray-300"
              }`}
            >
              %
            </button>
            <button
              type="button"
              onClick={() => setDisplayMode("count")}
              className={`px-3 text-sm transition-colors ${
                displayMode === "count" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-gray-300"
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
              <span className="max-w-52 truncate">{teacher}</span>
              <button
                type="button"
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

          {selectedTeachers.length >= maxCompare && (
            <span className="text-xs text-amber-400">Máximo {maxCompare} para mantener legibilidad</span>
          )}
        </div>
      )}

      {/* Chart */}
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
          <BarChart
            data={transformedChartData.map((r) => ({
              ...r,
              teacherShort: teacherShort(r.teacher),
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="teacherShort"
              stroke="#9CA3AF"
              interval={0}
              angle={-12}
              textAnchor="end"
              height={65}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              stroke="#9CA3AF"
              domain={displayMode === "percentage" ? [0, 100] : [0, "dataMax"]}
              ticks={displayMode === "percentage" ? [0, 30, 60, 100] : undefined}
              allowDecimals={false}
              tickFormatter={(v) => {
                const n = Number(v);
                if (displayMode === "percentage") return `${Math.round(n)}%`;
                return Number.isFinite(n) ? n.toLocaleString() : String(v);
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar dataKey={posKey} stackId="a" fill={COLORS.positive} name="Positive" />
            <Bar dataKey={neuKey} stackId="a" fill={COLORS.neutral} name="Neutral" />
            <Bar dataKey={negKey} stackId="a" fill={COLORS.negative} name="Negative" />
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
