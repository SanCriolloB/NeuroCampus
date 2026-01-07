// frontend/src/features/datos/hooks/useValidateDataset.ts
import { useState } from "react";
import type { ValidarResp } from "@/types/neurocampus";
import { datosApi } from "@/features/datos/api";
import { errMsg } from "./_utils";

export function useValidateDataset() {
  const [data, setData] = useState<ValidarResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(file: File, datasetId: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await datosApi.validar(file, datasetId);
      setData(res);
      return res;
    } catch (e) {
      const msg = errMsg(e);
      setError(msg);
      throw new Error(msg);
    } finally {
      setLoading(false);
    }
  }

  return { data, loading, error, run };
}
