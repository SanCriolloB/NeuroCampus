// frontend/src/services/dashboard.ts
// Cliente de API para la pestaña «Dashboard».
//
// Reglas de negocio (fuente de verdad)
// -----------------------------------
// - El Dashboard SOLO consume histórico (no consulta datasets individuales):
//   - historico/unificado.parquet (processed)
//   - historico/unificado_labeled.parquet (labeled)
// - Los filtros siempre se aplican sobre histórico:
//   - periodo (exacto) o rango (periodo_from..periodo_to)
//   - docente / asignatura / programa (opcionales)
//
// Este fichero NO cambia el layout del Dashboard; solo encapsula llamadas HTTP y
// tipa respuestas para facilitar wiring en componentes.

import api, { qs } from "./apiClient";

// ---------------------------------------------------------------------------
// Tipos (alineados con backend/src/neurocampus/app/schemas/dashboard.py)
// ---------------------------------------------------------------------------

/** Estado de archivo del histórico (existencia + mtime ISO UTC). */
export type DashboardFileStatus = {
  path: string;
  exists: boolean;
  mtime: string | null;
};

/** Respuesta de GET /dashboard/status. */
export type DashboardStatus = {
  manifest_exists: boolean;
  manifest_updated_at: string | null;
  manifest_corrupt: boolean;
  periodos_disponibles: string[];
  processed: DashboardFileStatus;
  labeled: DashboardFileStatus;
  ready_processed: boolean;
  ready_labeled: boolean;
};

/** Respuesta de GET /dashboard/periodos. */
export type DashboardPeriodos = {
  items: string[];
};

/** Respuesta de GET /dashboard/catalogos. */
export type DashboardCatalogos = {
  docentes: string[];
  asignaturas: string[];
  programas: string[];
};

/** Respuesta de GET /dashboard/kpis. */
export type DashboardKPIs = {
  /** Predicciones persistidas (conteo) para el periodo/rango y filtros aplicados. */
  predicciones: number;
  /** Evaluaciones registradas (conteo) para el periodo/rango y filtros aplicados. */
  evaluaciones: number;
  /** Docentes únicos (conteo) para el periodo/rango y filtros aplicados. */
  docentes: number;
  /** Asignaturas únicas (conteo) para el periodo/rango y filtros aplicados. */
  asignaturas: number;
  /** Score promedio (si aplica) en el periodo/rango solicitado. */
  score_promedio: number | null;
};

/** Punto de una serie por periodo (GET /dashboard/series). */
export type DashboardSeriesPoint = {
  periodo: string;
  value: number;
};

/** Respuesta de GET /dashboard/series. */
export type DashboardSeries = {
  metric: string;
  points: DashboardSeriesPoint[];
};

/** Bucket de sentimiento (neg/neu/pos). */
export type DashboardSentimientoBucket = {
  label: string;
  value: number;
};

/** Respuesta de GET /dashboard/sentimiento. */
export type DashboardSentimiento = {
  buckets: DashboardSentimientoBucket[];
};

/** Ítem de ranking (docente/asignatura). */
export type DashboardRankingItem = {
  name: string;
  value: number;
};

/** Respuesta de GET /dashboard/rankings. */
export type DashboardRankings = {
  by: string;
  metric: string;
  order: string;
  items: DashboardRankingItem[];
};

/** Punto del radar (promedio por pregunta_1..10). */
export type DashboardRadarItem = {
  key: string;
  value: number | null;
};

/** Respuesta de GET /dashboard/radar. */
export type DashboardRadar = {
  items: DashboardRadarItem[];
};

/** Término del wordcloud. */
export type DashboardWordcloudItem = {
  text: string;
  value: number;
  sentiment?: "positive" | "neutral" | "negative";
};

/** Respuesta de GET /dashboard/wordcloud. */
export type DashboardWordcloud = {
  items: DashboardWordcloudItem[];
};

// ---------------------------------------------------------------------------
// Tipos de filtros (query params)
// ---------------------------------------------------------------------------

/**
 * Filtros estándar del Dashboard.
 *
 * Nota: El backend espera `periodo_from` / `periodo_to` (snake_case).
 * En frontend mantenemos camelCase para uso natural en TS.
 */
export type DashboardFilters = {
  /** Periodo exacto (tiene prioridad sobre rango si se envía). */
  periodo?: string | null;
  /** Inicio de rango inclusivo. */
  periodoFrom?: string | null;
  /** Fin de rango inclusivo. */
  periodoTo?: string | null;
  /** Filtro por docente (opcional). */
  docente?: string | null;
  /** Filtro por asignatura (opcional). */
  asignatura?: string | null;
  /** Filtro por programa (opcional). */
  programa?: string | null;
};

/** Mapea filtros TS -> query params backend (snake_case). */
function filtersToQuery(filters?: DashboardFilters) {
  const f = filters || {};
  return {
    periodo: f.periodo ?? undefined,
    periodo_from: f.periodoFrom ?? undefined,
    periodo_to: f.periodoTo ?? undefined,
    docente: f.docente ?? undefined,
    asignatura: f.asignatura ?? undefined,
    programa: f.programa ?? undefined,
  };
}

// ---------------------------------------------------------------------------
// Funciones del cliente
// ---------------------------------------------------------------------------

/** GET /dashboard/status — liviano: usa manifest + mtimes (no carga parquet). */
export async function getDashboardStatus() {
  const { data } = await api.get<DashboardStatus>("/dashboard/status");
  return data;
}

/** GET /dashboard/periodos — lista ordenada (desde manifest). */
export async function listPeriodos() {
  const { data } = await api.get<DashboardPeriodos>("/dashboard/periodos");
  return data;
}

/** GET /dashboard/catalogos — listas válidas para dropdowns (docente/asignatura/programa). */
export async function getCatalogos(filters?: DashboardFilters) {
  const q = qs(filtersToQuery(filters));
  const { data } = await api.get<DashboardCatalogos>(`/dashboard/catalogos${q}`);
  return data;
}

/** GET /dashboard/kpis — KPIs agregados desde histórico processed. */
export async function getKpis(filters?: DashboardFilters) {
  const q = qs(filtersToQuery(filters));
  const { data } = await api.get<DashboardKPIs>(`/dashboard/kpis${q}`);
  return data;
}

/**
 * GET /dashboard/series — serie por periodo para una métrica.
 *
 * `metric` es requerido y debe coincidir con las métricas soportadas por backend
 * (p.ej. evaluaciones, score_promedio, docentes, asignaturas).
 */
export async function getSeries(args: { metric: string } & DashboardFilters) {
  const { metric, ...filters } = args;
  const q = qs({ metric, ...filtersToQuery(filters) });
  const { data } = await api.get<DashboardSeries>(`/dashboard/series${q}`);
  return data;
}

/**
 * GET /dashboard/sentimiento — distribución neg/neu/pos desde histórico labeled.
 *
 * Importante: si el histórico labeled aún no existe, el backend responde 404.
 * El Dashboard debe manejarlo como estado "no disponible" sin romper el layout.
 */
export async function getSentimiento(filters?: DashboardFilters) {
  const q = qs(filtersToQuery(filters));
  const { data } = await api.get<DashboardSentimiento>(`/dashboard/sentimiento${q}`);
  return data;
}

/**
 * GET /dashboard/rankings — top/bottom por docente o asignatura.
 *
 * `by` y `metric` son requeridos; `order` y `limit` tienen defaults en backend.
 */
export async function getRankings(args: { by: "docente" | "asignatura"; metric: string; order?: "asc" | "desc"; limit?: number } & DashboardFilters) {
  const { by, metric, order, limit, ...filters } = args;
  const q = qs({ by, metric, order, limit, ...filtersToQuery(filters) });
  const { data } = await api.get<DashboardRankings>(`/dashboard/rankings${q}`);
  return data;
}


/** GET /dashboard/radar — promedios por pregunta_1..10 (histórico processed). */
export async function getRadar(filters?: DashboardFilters) {
  const q = qs(filtersToQuery(filters));
  const { data } = await api.get<DashboardRadar>(`/dashboard/radar${q}`);
  return data;
}

/**
 * GET /dashboard/wordcloud — top términos desde histórico labeled.
 *
 * `limit` controla la cantidad máxima de tokens retornados.
 */
export async function getWordcloud(args?: ({ limit?: number } & DashboardFilters)) {
  const a = args || {};
  const limit = a.limit ?? undefined;
  const { limit: _omit, ...filters } = a as any;
  const q = qs({ limit, ...filtersToQuery(filters) });
  const { data } = await api.get<DashboardWordcloud>(`/dashboard/wordcloud${q}`);
  return data;
}
