// frontend/src/services/datos.test.ts
//
// Tests de la capa de servicios de Datos:
//  - upload: subida de dataset con FormData.
//  - validar: validación sin guardar.
//  - resumen: resumen estadístico de un dataset.
//  - sentimientos: análisis de sentimientos (BETO).

import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  upload,
  validar,
  resumen,
  sentimientos,
  type UploadResp,
  type ValidarResp,
  type DatasetResumen,
  type DatasetSentimientos,
} from "./datos";

describe("services/datos", () => {
  beforeEach(() => {
    // Sobrescribimos fetch con un mock limpio en cada test
    // @ts-ignore
    global.fetch = vi.fn();
  });

  it("upload envía FormData con periodo, dataset_id y overwrite", async () => {
    const mockJson: UploadResp = {
      ok: true,
      dataset_id: "2020-1",
      stored_as: "localfs://...",
      message: "ingesta-ok",
    };

    // @ts-ignore
    (global.fetch as any).mockResolvedValue({
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

    const call = (global.fetch as any).mock.calls[0];
    const url = call[0] as string;
    const init = call[1] as RequestInit;

    expect(url).toContain("/datos/upload");
    expect(init.body).toBeInstanceOf(FormData);

    const body = init.body as FormData;
    expect(body.get("periodo")).toBe("2020-1");
    expect(body.get("dataset_id")).toBe("2020-1");
    expect(body.get("overwrite")).toBe("true");
    expect(body.get("file")).toBeInstanceOf(File);
  });

  it("validar envía FormData con dataset_id y file", async () => {
    const mockJson: ValidarResp = {
      ok: true,
      dataset_id: "docentes",
      sample: [{ a: 1 }],
    };

    // @ts-ignore
    (global.fetch as any).mockResolvedValue({
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

    const call = (global.fetch as any).mock.calls[0];
    const url = call[0] as string;
    const init = call[1] as RequestInit;

    expect(url).toContain("/datos/validar");
    expect(init.body).toBeInstanceOf(FormData);

    const body = init.body as FormData;
    expect(body.get("dataset_id")).toBe("docentes");
    expect(body.get("file")).toBeInstanceOf(File);
  });

  it("resumen llama a /datos/resumen con dataset y devuelve DatasetResumen", async () => {
    const mockJson: DatasetResumen = {
      dataset_id: "2024-2",
      n_rows: 100,
      n_cols: 10,
      periodos: ["2024-2"],
      fecha_min: "2024-08-01",
      fecha_max: "2024-11-01",
      n_docentes: 5,
      n_asignaturas: 8,
      columns: [
        {
          name: "docente",
          dtype: "string",
          non_nulls: 100,
          sample_values: ["DOC1", "DOC2"],
        },
      ],
    };

    // @ts-ignore
    (global.fetch as any).mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: () => "application/json" },
      json: async () => mockJson,
    });

    const resp = await resumen({ dataset: "2024-2" });

    const call = (global.fetch as any).mock.calls[0];
    const url = call[0] as string;

    expect(url).toContain("/datos/resumen?dataset=2024-2");
    expect(resp).toEqual(mockJson);
  });

  it("sentimientos llama a /datos/sentimientos con dataset y devuelve DatasetSentimientos", async () => {
    const mockJson: DatasetSentimientos = {
      dataset_id: "2024-2",
      total_comentarios: 20,
      global_counts: [
        { label: "neg", count: 5, proportion: 0.25 },
        { label: "neu", count: 5, proportion: 0.25 },
        { label: "pos", count: 10, proportion: 0.5 },
      ],
      por_docente: [
        {
          group: "DOC1",
          counts: [
            { label: "neg", count: 1, proportion: 0.1 },
            { label: "neu", count: 2, proportion: 0.2 },
            { label: "pos", count: 7, proportion: 0.7 },
          ],
        },
      ],
      por_asignatura: [
        {
          group: "MAT1",
          counts: [
            { label: "neg", count: 0, proportion: 0 },
            { label: "neu", count: 1, proportion: 0.2 },
            { label: "pos", count: 4, proportion: 0.8 },
          ],
        },
      ],
    };

    // @ts-ignore
    (global.fetch as any).mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: () => "application/json" },
      json: async () => mockJson,
    });

    const resp = await sentimientos({ dataset: "2024-2" });

    const call = (global.fetch as any).mock.calls[0];
    const url = call[0] as string;

    expect(url).toContain("/datos/sentimientos?dataset=2024-2");
    expect(resp).toEqual(mockJson);
  });
});
