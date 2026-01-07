// frontend/src/features/datos/api.ts
// Adaptadores de acceso para la Feature "Datos".
// No contiene UI. Sólo orquesta llamadas a services existentes.

import * as datosSvc from "@/services/datos";
import * as jobsSvc from "@/services/jobs";

import type {
  EsquemaResp,
  ValidarResp,
  UploadResp,
  DatasetResumen,
  DatasetSentimientos,
} from "@/types/neurocampus";

import type { BetoPreprocJob } from "@/services/jobs";

export const datosApi = {
  esquema: async (): Promise<EsquemaResp> => datosSvc.esquema(),
  validar: async (
    file: File,
    datasetId: string,
    opts?: { fmt?: "csv" | "xlsx" | "parquet" },
  ): Promise<ValidarResp> => datosSvc.validar(file, datasetId, opts),

  // Nota: en el backend actual, `periodo` se envía en el campo "periodo" y
  // `dataset_id` se manda por compatibilidad. El cliente `uploadWithProgress`
  // ya lo hace internamente.
  uploadWithProgress: async (
    file: File,
    periodo: string,
    overwrite: boolean,
    onProgress?: (pct: number) => void,
  ): Promise<UploadResp> => datosSvc.uploadWithProgress(file, periodo, overwrite, onProgress),

  resumen: async (dataset: string): Promise<DatasetResumen> =>
    datosSvc.resumen({ dataset }),

  sentimientos: async (dataset: string): Promise<DatasetSentimientos> =>
    datosSvc.sentimientos({ dataset }),
};

export const jobsApi = {
  launchBetoPreproc: async (dataset: string): Promise<BetoPreprocJob> =>
    jobsSvc.launchBetoPreproc({ dataset }),

  getBetoJob: async (jobId: string): Promise<BetoPreprocJob> =>
    jobsSvc.getBetoJob(jobId),
};
