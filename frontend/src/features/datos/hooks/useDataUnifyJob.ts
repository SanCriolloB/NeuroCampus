// frontend/src/features/datos/hooks/useDataUnifyJob.ts
import { useEffect, useRef, useState } from "react";
import type { DataUnifyJob } from "@/services/jobs";
import { jobsApi } from "@/features/datos/api";
import { errMsg } from "./_utils";

/**
 * Hook de polling para jobs de unificación histórica (historico/*).
 *
 * Sigue el mismo patrón que `useBetoPreprocJob` para mantener consistencia.
 */
export function useDataUnifyJob(jobId: string | null, opts?: { intervalMs?: number }) {
  const intervalMs = opts?.intervalMs ?? 2000;

  const [job, setJob] = useState<DataUnifyJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const id = jobId; // <- Narrowing explícito para TypeScript

    let alive = true;

    async function tick() {
      setLoading(true);
      setError(null);
      try {
        const res = await jobsApi.getDataUnifyJob(id);
        if (!alive) return;

        setJob(res);

        if (res.status === "done" || res.status === "failed") {
          if (timerRef.current) window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
      } catch (e) {
        if (!alive) return;
        setError(errMsg(e));
      } finally {
        if (alive) setLoading(false);
      }
    }

    void tick();
    timerRef.current = window.setInterval(() => void tick(), intervalMs);

    return () => {
      alive = false;
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [jobId, intervalMs]);


  return { job, loading, error };
}
