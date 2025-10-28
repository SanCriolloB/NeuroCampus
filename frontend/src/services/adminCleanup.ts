// frontend/src/services/adminCleanup.ts

const API_BASE = import.meta.env.VITE_API_BASE as string; // asegura string

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("NC_ADMIN_TOKEN");
  // Devuelve SIEMPRE un objeto; si no hay token, objeto vacío
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export type Candidate = {
  path: string;
  size: number;
  age_days: number;
  reason: string;
};

export type CleanupResponse = {
  summary: {
    total_files: number;
    total_size_bytes: number;
    candidates_count: number;
    candidates_size_bytes: number;
  };
  candidates: Candidate[];
  dry_run: boolean;
  force: boolean;
  moved_bytes: number;
  actions: any[];
  log_file: string;
  trash_dir: string;
};

export async function getInventory(params: {
  retention_days: number;
  keep_last: number;
  exclude_globs?: string;
}) {
  const q = new URLSearchParams({
    retention_days: String(params.retention_days),
    keep_last: String(params.keep_last),
    ...(params.exclude_globs ? { exclude_globs: params.exclude_globs } : {}),
  });

  const res = await fetch(`${API_BASE}/admin/cleanup/inventory?${q.toString()}`, {
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    }, // <- ya es HeadersInit válido
  });
  if (!res.ok) throw new Error(`Inventory failed: ${res.status}`);
  return (await res.json()) as CleanupResponse;
}

export async function postCleanup(body: {
  retention_days: number;
  keep_last: number;
  exclude_globs?: string;
  dry_run: boolean;
  force: boolean;
  trash_dir?: string;
  trash_retention_days?: number;
}) {
  const res = await fetch(`${API_BASE}/admin/cleanup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Cleanup failed: ${res.status}`);
  return (await res.json()) as CleanupResponse;
}

export async function getLogs(limit = 200) {
  const res = await fetch(`${API_BASE}/admin/cleanup/logs?limit=${limit}`, {
    headers: {
      ...authHeaders(),
    },
  });
  if (!res.ok) throw new Error(`Logs failed: ${res.status}`);
  return (await res.json()) as { lines: string[] };
}
