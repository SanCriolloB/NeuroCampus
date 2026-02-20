// frontend/src/features/modelos/hooks/useModelosContext.ts
// =============================================================================
// NeuroCampus — Feature Modelos: Hook de contexto (dataset/family/model)
// =============================================================================
//
// Este hook centraliza el estado de selección que la pestaña Modelos necesita
// (dataset_id, family, model_name), y lo sincroniza con el store global
// (si existe) para mantener coherencia entre pestañas.
//
// Principios:
// - No alterar la UI del prototipo.
// - Evitar acoplar componentes UI a detalles del store; este hook funciona como
//   "puente".
//
// Notas:
// - Si el store global no está disponible o cambia de nombre, este hook mantiene
//   un estado local y expone setters.
// - Cuando el backend esté completo, este hook será el lugar para añadir
//   persistencia (query params, localStorage, etc.) sin tocar la UI.

import { useCallback, useMemo, useState } from "react";

import { type Family, type ModeloName } from "../types";

/**
 * Contrato mínimo de un store global de filtros.
 *
 * El repositorio puede tener stores con nombres diferentes; por eso evitamos
 * importar directamente. La sincronización real se hace de forma opcional en
 * `bindGlobalFilters`.
 */
interface GlobalFiltersLike {
  activeDatasetId?: string;
  setActiveDatasetId?: (id: string) => void;

  activeFamily?: Family;
  setActiveFamily?: (family: Family) => void;

  activeModelName?: ModeloName;
  setActiveModelName?: (name: ModeloName) => void;
}

/**
 * Intenta enlazar con un store global si existe.
 *
 * Implementación:
 * - En este paso NO asumimos un path exacto.
 * - La integración concreta se hará cuando confirmemos dónde vive el store.
 *
 * @returns `null` cuando no hay store detectable.
 */
function bindGlobalFilters(): GlobalFiltersLike | null {
  // TODO(P16): conectar con store real del proyecto.
  // Ejemplo esperado en NeuroCampus:
  //   import { useAppFiltersStore } from "@/stores/appFilters.store";
  //   return useAppFiltersStore.getState();
  return null;
}

/**
 * Estado expuesto por el hook de contexto de Modelos.
 */
export interface ModelosContextState {
  datasetId: string;
  family: Family;
  modelName: ModeloName;

  setDatasetId: (id: string) => void;
  setFamily: (family: Family) => void;
  setModelName: (name: ModeloName) => void;
}

/**
 * Hook para manejar dataset/family/model seleccionados en la pestaña Modelos.
 *
 * @param initialDatasetId dataset inicial (fallback).
 * @param initialFamily family inicial (fallback).
 * @param initialModelName modelo inicial (fallback).
 */
export function useModelosContext(params?: {
  initialDatasetId?: string;
  initialFamily?: Family;
  initialModelName?: ModeloName;
}): ModelosContextState {
  const global = useMemo(() => bindGlobalFilters(), []);

  // Defaults conservadores (no rompen UI)
  const defaultDatasetId = params?.initialDatasetId ?? "2024-2";
  const defaultFamily: Family = params?.initialFamily ?? "sentiment_desempeno";
  const defaultModel: ModeloName = params?.initialModelName ?? "rbm_general";

  // Si hay store global, usarlo como fuente inicial.
  const [datasetId, setDatasetIdLocal] = useState<string>(global?.activeDatasetId ?? defaultDatasetId);
  const [family, setFamilyLocal] = useState<Family>(global?.activeFamily ?? defaultFamily);
  const [modelName, setModelNameLocal] = useState<ModeloName>(global?.activeModelName ?? defaultModel);

  const setDatasetId = useCallback(
    (id: string) => {
      setDatasetIdLocal(id);
      global?.setActiveDatasetId?.(id);
    },
    [global]
  );

  const setFamily = useCallback(
    (f: Family) => {
      setFamilyLocal(f);
      global?.setActiveFamily?.(f);
    },
    [global]
  );

  const setModelName = useCallback(
    (name: ModeloName) => {
      setModelNameLocal(name);
      global?.setActiveModelName?.(name);
    },
    [global]
  );

  return {
    datasetId,
    family,
    modelName,
    setDatasetId,
    setFamily,
    setModelName,
  };
}
