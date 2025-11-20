import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import AdminCleanupPage from "../../pages/AdminCleanup";

beforeEach(() => {
  // limpiamos solo storage entre tests de esta suite
  localStorage.clear();
});

describe("AdminCleanupPage", () => {
  it("renderiza y permite invocar inventario", async () => {
    const mockJson = {
      summary: {
        total_files: 10,
        total_size_bytes: 1024,
        candidates_count: 2,
        candidates_size_bytes: 512,
      },
      candidates: [],
      dry_run: true,
      force: false,
      moved_bytes: 0,
      trash_dir: "/tmp/trash",
    };

    // `setupTests` ya deja global.fetch como vi.fn(); aquí definimos la respuesta
    // @ts-ignore
    global.fetch.mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: async () => mockJson,
    });

    render(<AdminCleanupPage />);

    const btn = await screen.findByText(/Inventario \(dry-run\)/i);
    fireEvent.click(btn);

    // El mock responde OK; esperamos que aparezca el resumen
    const resumen = await screen.findByText(/Total archivos: 10/i);
    expect(resumen).toBeInTheDocument();
  });
});
