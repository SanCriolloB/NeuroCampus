import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import AdminCleanupPage from "../../pages/AdminCleanup";

beforeEach(() => {
  vi.restoreAllMocks();
});

// Mock de fetch
vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
  ok: true,
  json: async () => ({
    summary: {
      total_files: 10,
      total_size_bytes: 1024,
      candidates_count: 2,
      candidates_size_bytes: 512,
    },
    candidates: [
      { path: "artifacts/modelX/runA/file.bin", size: 256, age_days: 100, reason: "age" },
      { path: "data/tmp/tmp.csv",               size: 256, age_days: 5,   reason: "surplus" },
    ],
    dry_run: true, force: false, moved_bytes: 0, actions: [],
    log_file: "logs/cleanup.log", trash_dir: ".trash"
  }),
}));

describe("AdminCleanupPage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("renderiza y permite invocar inventario", async () => {
    render(<AdminCleanupPage />);
    const btn = await screen.findByText(/Inventario \(dry-run\)/i);
    fireEvent.click(btn);
    // El mock responde OK; esperamos que aparezca el resumen
    const resumen = await screen.findByText(/Total archivos: 10/i);
    expect(resumen).toBeInTheDocument();
  });
});
