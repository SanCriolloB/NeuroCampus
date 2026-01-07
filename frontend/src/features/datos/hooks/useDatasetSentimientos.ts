// frontend/src/features/datos/hooks/useDatasetSentimientos.ts
import { useEffect, useState } from "react";
import type { DatasetSentimientos } from "@/types/neurocampus";
import { datosApi } from "@/features/datos/api";
import { errMsg } from "./_utils";

export function useDatasetSentimientos(dataset: string | null) {
  const [data, setData] = useState<DatasetSentimientos | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refetch() {
    if (!dataset) return;
    setLoading(true);
    setError(null);
    try {
      const res = await datosApi.sentimientos(dataset);
      setData(res);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  return { data, loading, error, refetch };
}
