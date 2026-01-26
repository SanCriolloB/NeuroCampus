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

import type { BetoPreprocJob, DataUnifyJob, FeaturesPrepareJob } from "@/services/jobs";

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
  /**
   * Lanza preprocesamiento BETO desde Datos.
   * - `opts` es opcional para mantener compatibilidad (si no se envía, backend usa defaults).
   */
  launchBetoPreproc: async (
    dataset: string,
    opts?: Omit<Parameters<typeof jobsSvc.launchBetoPreproc>[0], "dataset">,
  ): Promise<BetoPreprocJob> =>
    jobsSvc.launchBetoPreproc({ dataset, ...(opts ?? {}) }),

  /** Estado del job BETO. */
  getBetoJob: async (jobId: string): Promise<BetoPreprocJob> =>
    jobsSvc.getBetoJob(jobId),

  // -----------------------------------------------------------------------
  // Jobs del dominio Datos: Unificación histórica
  // -----------------------------------------------------------------------

  /** Lanza job de unificación (historico/*). */
  launchDataUnify: async (
    opts?: Parameters<typeof jobsSvc.launchDataUnify>[0],
  ): Promise<DataUnifyJob> =>
    jobsSvc.launchDataUnify(opts ?? {}),

  /** Estado del job de unificación. */
  getDataUnifyJob: async (jobId: string): Promise<DataUnifyJob> =>
    jobsSvc.getDataUnifyJob(jobId),

  /** Lista jobs de unificación recientes. */
  listDataUnifyJobs: async (limit = 20): Promise<DataUnifyJob[]> =>
    jobsSvc.listDataUnifyJobs(limit),

  // -----------------------------------------------------------------------
  // Jobs del dominio Datos: Feature-pack (artifacts/features/<dataset_id>/)
  // -----------------------------------------------------------------------

  /** Lanza job de feature-pack. */
  launchFeaturesPrepare: async (
    params: Parameters<typeof jobsSvc.launchFeaturesPrepare>[0],
  ): Promise<FeaturesPrepareJob> =>
    jobsSvc.launchFeaturesPrepare(params),

  /** Estado del job de feature-pack. */
  getFeaturesPrepareJob: async (jobId: string): Promise<FeaturesPrepareJob> =>
    jobsSvc.getFeaturesPrepareJob(jobId),

  /** Lista jobs de feature-pack recientes. */
  listFeaturesPrepareJobs: async (limit = 20): Promise<FeaturesPrepareJob[]> =>
    jobsSvc.listFeaturesPrepareJobs(limit),
};

