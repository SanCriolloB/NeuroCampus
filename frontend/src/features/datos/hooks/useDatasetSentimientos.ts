// frontend/src/features/datos/hooks/useDatasetSentimientos.ts
import { useEffect, useRef, useState } from "react";
import { datosApi } from "@/features/datos/api";
import type { DatasetSentimientos } from "@/types/neurocampus";
import { errMsg } from "./_utils";

type RefetchOpts = {
  retryOn404?: boolean;
  maxAttempts?: number; // total de intentos (incluye el primero)
  delayMs?: number;
};

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export function useDatasetSentimientos(dataset: string | null) {
  const [data, setData] = useState<DatasetSentimientos | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // para evitar que respuestas “viejas” pisen el estado
  const reqIdRef = useRef(0);

  async function refetch(opts?: RefetchOpts) {
    if (!dataset) return;

    const retryOn404 = opts?.retryOn404 ?? false;
    const maxAttempts = opts?.maxAttempts ?? 15; // ~30s si delay=2000
    const delayMs = opts?.delayMs ?? 2000;

    const reqId = ++reqIdRef.current;

    setLoading(true);
    setError(null);

    try {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          const res = await datosApi.sentimientos(dataset);
          if (reqId !== reqIdRef.current) return; // cancelado/obsoleto
          setData(res);
          setError(null);
          return;
        } catch (e: any) {
          const status = e?.response?.status;
          const msg = errMsg(e);

          // Si es 404 y está habilitado retry, esperamos y reintentamos
          if (retryOn404 && status === 404 && attempt < maxAttempts) {
            await sleep(delayMs);
            continue;
          }

          if (reqId !== reqIdRef.current) return;
          setError(msg);
          return;
        }
      }
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }

  // comportamiento actual: al cambiar dataset, intenta traer sentimientos 1 vez (sin retry)
  useEffect(() => {
    if (!dataset) {
      setData(null);
      setError(null);
      return;
    }
    void refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  return { data, loading, error, refetch };
}
