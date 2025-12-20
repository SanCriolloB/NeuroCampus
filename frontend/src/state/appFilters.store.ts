// frontend/src/state/appFilters.store.ts
import { useSyncExternalStore } from "react";

export type AppFiltersState = {
  activeDatasetId: string | null;
  activePeriodo: string | null;

  // Filtros opcionales (se activan cuando dashboard lo requiera)
  programa: string | null;
  asignatura: string | null;
  docente: string | null;
};

const STORAGE_KEY = "NC_APP_FILTERS_V1";

const defaultState: AppFiltersState = {
  activeDatasetId: null,
  activePeriodo: null,
  programa: null,
  asignatura: null,
  docente: null,
};

function safeParse(json: string | null): AppFiltersState | null {
  if (!json) return null;
  try {
    const obj = JSON.parse(json);
    if (!obj || typeof obj !== "object") return null;
    return {
      ...defaultState,
      ...obj,
    } as AppFiltersState;
  } catch {
    return null;
  }
}

function loadInitial(): AppFiltersState {
  try {
    const fromStorage = safeParse(localStorage.getItem(STORAGE_KEY));
    return fromStorage ?? defaultState;
  } catch {
    return defaultState;
  }
}

let state: AppFiltersState = loadInitial();
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore
  }
}

export function getAppFilters(): AppFiltersState {
  return state;
}

export function setAppFilters(patch: Partial<AppFiltersState>) {
  state = { ...state, ...patch };
  emit();
}

export function resetAppFilters() {
  state = { ...defaultState };
  emit();
}

export function subscribeAppFilters(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Hook de lectura (con selector opcional).
 * Ej:
 *  const datasetId = useAppFilters(s => s.activeDatasetId)
 */
export function useAppFilters<T = AppFiltersState>(
  selector?: (s: AppFiltersState) => T,
): T {
  const getSnapshot = () => (selector ? selector(state) : (state as unknown as T));
  const getServerSnapshot = () =>
    selector ? selector(defaultState) : (defaultState as unknown as T);

  return useSyncExternalStore(subscribeAppFilters, getSnapshot, getServerSnapshot);
}
