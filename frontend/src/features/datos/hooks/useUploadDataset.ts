// frontend/src/features/datos/hooks/useUploadDataset.ts
import { useState } from "react";
import type { UploadResp } from "@/types/neurocampus";
import { datosApi } from "@/features/datos/api";
import { errMsg } from "./_utils";

export function useUploadDataset() {
  const [data, setData] = useState<UploadResp | null>(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(file: File, periodo: string, overwrite = false) {
    setUploading(true);
    setError(null);
    setProgress(0);

    try {
      const res = await datosApi.uploadWithProgress(file, periodo, overwrite, (pct) => setProgress(pct));
      setData(res);
      setProgress(100);
      return res;
    } catch (e) {
      const msg = errMsg(e);
      setError(msg);
      throw new Error(msg);
    } finally {
      setUploading(false);
    }
  }

  return { data, progress, uploading, error, run };
}
