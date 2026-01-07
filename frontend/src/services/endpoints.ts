// frontend/src/services/endpoints.ts
// Centraliza rutas del backend para evitar strings hardcodeados en Features.

export const endpoints = {
  datos: {
    esquema: "/datos/esquema",
    validar: "/datos/validar",
    upload: "/datos/upload",
    resumen: (dataset: string) => `/datos/resumen?dataset=${encodeURIComponent(dataset)}`,
    sentimientos: (dataset: string) => `/datos/sentimientos?dataset=${encodeURIComponent(dataset)}`,
    // recomendado para “selector de datasets” (si existe en backend)
    list: "/datos/list",
  },
  jobs: {
    betoRun: "/jobs/preproc/beto/run",
    betoJob: (jobId: string) => `/jobs/preproc/beto/${encodeURIComponent(jobId)}`,
    betoList: (limit = 20) => `/jobs/preproc/beto?limit=${encodeURIComponent(String(limit))}`,
    rbmSearchRun: "/jobs/training/rbm-search",
    rbmSearchJob: (jobId: string) => `/jobs/training/rbm-search/${encodeURIComponent(jobId)}`,
    rbmSearchList: (limit = 20) => `/jobs/training/rbm-search?limit=${encodeURIComponent(String(limit))}`,
  },
  modelos: {
    entrenar: "/modelos/entrenar",
    estado: (jobId: string) => `/modelos/estado/${encodeURIComponent(jobId)}`,
    runs: "/modelos/runs",
    runDetails: (runId: string) => `/modelos/runs/${encodeURIComponent(runId)}`,
    champion: "/modelos/champion",
  },
  prediccion: {
    online: "/prediccion/online",
    batch: "/prediccion/batch",
  },
  admin: {
    cleanupInventory: "/admin/cleanup/inventory",
    cleanup: "/admin/cleanup",
    cleanupLogs: "/admin/cleanup/logs",
  },
} as const;
