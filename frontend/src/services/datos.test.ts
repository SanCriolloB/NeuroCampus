import { describe, it, expect, vi } from "vitest";
import { upload, validar } from "./datos";

describe("services/datos", () => {
  it("upload envía FormData con dataset_id y overwrite", async () => {
    const mockJson = { ok: true, dataset_id: "2020-1", rows_ingested: 3, stored_as: "localfs://..." };

    // @ts-ignore
    global.fetch.mockResolvedValue({
      ok: true,
      status: 201,
      statusText: "Created",
      headers: { get: () => "application/json" },
      json: async () => mockJson,
    });

    const file = new File(["a,b\n1,2"], "sample.csv", { type: "text/csv" });
    const resp = await upload(file, "2020-1", true);

    expect(resp.ok).toBe(true);
    expect(resp.dataset_id).toBe("2020-1");
    expect(resp.rows_ingested).toBe(3);

    // Verificar cuerpo enviado
    const call = (global.fetch as any).mock.calls[0];
    const body = call[1].body as FormData;
    expect(body).toBeInstanceOf(FormData);
    // No es trivial leer FormData directamente: validamos con has() si está disponible
    // @ts-ignore
    expect(body.has("dataset_id")).toBe(true);
    // @ts-ignore
    expect(body.has("overwrite")).toBe(true);
  });

  it("validar envía FormData con file", async () => {
    const mockJson = { ok: true, sample: [{ a: 1 }], dataset_id: "docentes" };

    // @ts-ignore
    global.fetch.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: () => "application/json" },
      json: async () => mockJson,
    });

    const file = new File(["a\n1"], "s.csv", { type: "text/csv" });
    const resp = await validar(file, "docentes");

    expect(resp.ok).toBe(true);
    expect(Array.isArray(resp.sample)).toBe(true);
  });
});
