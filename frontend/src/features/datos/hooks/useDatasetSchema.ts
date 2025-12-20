// frontend/src/features/datos/hooks/useDatasetSchema.ts
import { useEffect, useState } from "react";
import type { EsquemaResp } from "@/types/neurocampus";
import { datosApi } from "@/features/datos/api";
import { errMsg } from "./_utils";

export function useDatasetSchema() {
  const [data, setData] = useState<EsquemaResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refetch() {
    setLoading(true);
    setError(null);
    try {
      const res = await datosApi.esquema();
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
  }, []);

  return { data, loading, error, refetch };
}
