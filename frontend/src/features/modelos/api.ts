// frontend/src/features/modelos/api.ts
// =============================================================================
// NeuroCampus — Feature Modelos: API wrapper estable para la UI
// =============================================================================
//
// Este wrapper expone funciones de alto nivel para la pestaña Modelos,
// encapsulando:
// - llamadas HTTP (`src/services/modelos.ts`)
// - compatibilidad legacy (endpoints/schemas antiguos)
// - mapeo DTO -> UI (mappers)
//
// Principio clave:
// - La UI del prototipo NO debe conocer la API real ni sus detalles.
// - La UI solo consume datos en la forma del prototipo (RunRecord, etc).
//
// Nota sobre documentación:
// - Comentarios JSDoc + secciones claras, pensando en documentación futura.
// - Los TODO indican contratos sugeridos para backend futuro (bundle/artifacts).

import * as modelosService from "@/services/modelos";
import {
  type ChampionInfoDto,
  type EntrenarRequestDto,
  type EstadoResponseDto,
  type ModelSweepRequestDto,
  type ReadinessResponseDto,
  type RunDetailsDto,
  type RunSummaryDto,
  type PromoteChampionRequestDto,
  type Family,
  type ModeloName,
} from "./types";

import {
  mapChampionToChampionRecord,
  mapChampionToResolvedModel,
  mapMetricsToRunMetrics,
  mapRunSummaryToRunRecord,
  mergeRunDetails,
  normalizeFamily,
} from "./mappers";

import {
  FAMILY_CONFIGS,
  type ChampionRecord,
  type ResolvedModel,
  type RunRecord,
  type SweepResult,
} from "@/components/models/mockData";

/**
 * Filtra y normaliza `hparams` hacia el contrato actual del servicio (`number | null`).
 *
 * Nota:
 * - La UI puede producir strings/bools en el futuro; hoy el backend soporta numéricos.
 * - Los valores no numéricos se omiten (conservador).
 */
function toNumericHparams(
  hparams: Record<string, number | string | boolean | null> | undefined
): Record<string, number | null> | undefined {
  if (!hparams) return undefined;

  const out: Record<string, number | null> = {};
  for (const [k, v] of Object.entries(hparams)) {
    if (v === null) {
      out[k] = null;
      continue;
    }
    if (typeof v === "number" && Number.isFinite(v)) {
      out[k] = v;
      continue;
    }
    // Ignorar strings/bools para no romper contrato del backend.
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * Normaliza un valor que en DTO puede ser `null` para ajustarse a un
 * contrato de servicio que espera `undefined` cuando no aplica.
 */
function nullToUndef<T>(value: T | null | undefined): T | undefined {
  return value === null ? undefined : value;
}


/**
 * Wrapper estable para consumirse desde hooks/UI.
 */
export const modelosApi = {
  /** Readiness del dataset (si artifacts existen). */
  async readiness(datasetId: string): Promise<ReadinessResponseDto> {
    return (await modelosService.getReadiness({ dataset_id: datasetId })) as unknown as ReadinessResponseDto;
  },

  /**
   * Lista runs como `RunRecord` (forma UI prototipo).
   *
   * @param datasetId dataset/periodo.
   * @param family family (Ruta 2).
   * @param modelName filtra por modelo si se desea.
   */
  async listRunsUI(params: {
    datasetId: string;
    family?: Family;
    modelName?: ModeloName;
  }): Promise<RunRecord[]> {
    const runs: RunSummaryDto[] = await modelosService.listRuns({
      dataset_id: params.datasetId,
      family: params.family,
      model_name: params.modelName,
    });

    return runs.map(mapRunSummaryToRunRecord);
  },

  /**
   * Obtiene detalle del run y devuelve `RunRecord` enriquecido.
   */
  async getRunDetailsUI(runId: string, base?: RunRecord): Promise<RunRecord> {
    const details: RunDetailsDto = await modelosService.getRunDetails(runId);

    const baseRecord = base ?? mapRunSummaryToRunRecord({
      run_id: details.run_id,
      model_name: (details as any).model_name ?? "rbm_general",
      dataset_id: details.dataset_id ?? "unknown",
      family: details.family ?? "sentiment_desempeno",
      task_type: details.task_type ?? null,
      input_level: details.input_level ?? null,
      target_col: details.target_col ?? null,
      data_plan: details.data_plan ?? null,
      data_source: details.data_source ?? null,
      created_at: new Date().toISOString(),
      metrics: details.metrics ?? {},
    });

    return mergeRunDetails(baseRecord, details);
  },

  /**
   * Obtiene champion actual y lo devuelve en dos formas:
   * - `resolved` (para header)
   * - `record` (para UI champion tab)
   */
  async getChampionUI(params: {
    datasetId: string;
    family?: Family;
    modelName?: ModeloName;
  }): Promise<{ resolved: ResolvedModel; record: ChampionRecord; raw: ChampionInfoDto }> {
    const raw = (await modelosService.getChampion({
      dataset_id: params.datasetId,
      family: params.family,
      model_name: params.modelName,
    })) as unknown as ChampionInfoDto;

    return {
      raw,
      resolved: mapChampionToResolvedModel(raw),
      record: mapChampionToChampionRecord(raw),
    };
  },

  /**
   * Entrena un modelo y retorna el `job_id` para polling.
   *
   * Nota:
   * - La UI del prototipo simula; aquí solo enviamos request.
   */
  async train(request: EntrenarRequestDto): Promise<{ jobId: string }> {
    const resp = await modelosService.entrenar({
      ...request,
      hparams: toNumericHparams(request.hparams),
    } as any);
    return { jobId: resp.job_id };
  },

  /**
   * Polling del estado del job (train o sweep legacy).
   */
  async getJobStatus(jobId: string): Promise<EstadoResponseDto> {
    return (await modelosService.estado(jobId)) as unknown as EstadoResponseDto;
  },

  /**
   * Promueve un run a champion.
   */
  async promote(request: PromoteChampionRequestDto): Promise<{ ok: boolean }> {
    await modelosService.promoteChampion(request);
    return { ok: true };
  },

  /**
   * Ejecuta sweep:
   * - preferido: `POST /modelos/sweep` (sync o semi-sync)
   * - fallback: `/modelos/entrenar/sweep` (legacy async), si existe.
   *
   * Retorna `SweepResult` con candidates listos para UI.
   */
  async sweep(params: ModelSweepRequestDto): Promise<SweepResult> {
    // Primero intenta endpoint moderno.
    const resp = (await modelosService.sweep({
      ...params,
      // El servicio no espera `null`; se usa `undefined` cuando no aplica.
      warm_start_from: nullToUndef(params.warm_start_from),
      warm_start_run_id: nullToUndef(params.warm_start_run_id),
    } as any)) as any;

    const family = normalizeFamily(resp.family);
    const candidates = (resp.candidates ?? []).map((c: any) => {
      // Construimos un RunRecord mínimo a partir de cada candidato.
      const record = mapRunSummaryToRunRecord({
        run_id: c.run_id ?? `sweep_${String(c.model_name)}_unknown`,
        model_name: String(c.model_name),
        dataset_id: resp.dataset_id,
        family,
        task_type: null,
        input_level: null,
        target_col: null,
        data_plan: null,
        data_source: null,
        created_at: new Date().toISOString(),
        metrics: (c.metrics ?? {}) as any,
      });

      // Enriquecemos primary metric value
      // FAMILY_CONFIGS define la métrica primaria y el modo por family (prototipo UI).
      const fc = FAMILY_CONFIGS[family];
      const pmv = (c.primary_metric_value ?? 0) as number;
      return {
        ...record,
        primary_metric: fc.primaryMetric,
        metric_mode: fc.metricMode,
        primary_metric_value: pmv,
        metrics: mapMetricsToRunMetrics(c.metrics ?? {}),
        status: c.status === "failed" ? "failed" : "completed",
      } as RunRecord;
    });

    const bestId = resp.best?.run_id ?? candidates[0]?.run_id ?? "unknown";

    return {
      candidates,
      winner_run_id: bestId,
      winner_reason: `Best candidate selected by ${resp.primary_metric} (${resp.primary_metric_mode})`,
      auto_promoted: Boolean(resp.champion_promoted),
    };
  },

  /**
   * Recupera summary del sweep (si el backend lo expone).
   */
  async getSweepSummary(sweepId: string): Promise<modelosService.SweepSummaryResp> {
    return modelosService.getSweepSummary(sweepId);
  },
};
