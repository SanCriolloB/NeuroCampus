// frontend/src/services/datos.test.ts
//
// Tests de la capa de servicios de Datos:
//  - upload: subida de dataset con FormData.
//  - validar: validación sin guardar.
//  - resumen: resumen estadístico de un dataset.
//  - sentimientos: análisis de sentimientos (BETO) asociado a un dataset.
//

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

// Mocks compartidos para apiClient (default export y named export `api`)
const postMock = vi.fn();
const getMock = vi.fn();

// Mock del módulo apiClient para interceptar llamadas HTTP que hace datos.ts
vi.mock("./apiClient", () => ({
  __esModule: true,
  default: {
    post: postMock,
    get: getMock,
  },
  api: {
    post: postMock,
    get: getMock,
  },
}));

describe("services/datos", () => {
  beforeEach(() => {
    postMock.mockReset();
    getMock.mockReset();
  });

  it("upload envía FormData con periodo, dataset_id y overwrite", async () => {
    const mockJson: UploadResp = {
      ok: true,
      dataset_id: "2020-1",
      stored_as: "localfs://...",
      message: "ingesta-ok",
    };

    // api.post devolverá { data: mockJson }
    postMock.mockResolvedValue({ data: mockJson });

    const file = new File(["a,b\n1,2"], "sample.csv", { type: "text/csv" });
    const resp = await upload(file, "2020-1", true);

    expect(resp.ok).toBe(true);
    expect(resp.dataset_id).toBe("2020-1");

    // Verificar llamada a apiClient
    expect(postMock).toHaveBeenCalledTimes(1);
    const [path, body] = postMock.mock.calls[0];

    expect(path).toBe("/datos/upload");
    expect(body).toBeInstanceOf(FormData);

    const fd = body as FormData;
    // campos que construye datos.upload()
    expect(fd.get("periodo")).toBe("2020-1");
    expect(fd.get("dataset_id")).toBe("2020-1");
    expect(fd.get("overwrite")).toBe("true");
    expect(fd.get("file")).toBeInstanceOf(File);
  });

  it("validar envía FormData con dataset_id y file", async () => {
    const mockJson: ValidarResp = {
      ok: true,
      dataset_id: "docentes",
      sample: [{ a: 1 }],
    };

    postMock.mockResolvedValue({ data: mockJson });

    const file = new File(["a\n1"], "s.csv", { type: "text/csv" });
    const resp = await validar(file, "docentes");

    expect(resp.ok).toBe(true);
    expect(Array.isArray(resp.sample)).toBe(true);

    expect(postMock).toHaveBeenCalledTimes(1);
    const [path, body] = postMock.mock.calls[0];

    expect(path).toBe("/datos/validar");
    expect(body).toBeInstanceOf(FormData);

    const fd = body as FormData;
    expect(fd.get("dataset_id")).toBe("docentes");
    expect(fd.get("file")).toBeInstanceOf(File);
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

    getMock.mockResolvedValue({ data: mockJson });

    const resp = await resumen({ dataset: "2024-2" });

    expect(getMock).toHaveBeenCalledTimes(1);
    const [path] = getMock.mock.calls[0];

    // datos.ts construye la URL con query param: /datos/resumen?dataset=...
    expect(path).toBe("/datos/resumen?dataset=2024-2");
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

    getMock.mockResolvedValue({ data: mockJson });

    const resp = await sentimientos({ dataset: "2024-2" });

    expect(getMock).toHaveBeenCalledTimes(1);
    const [path] = getMock.mock.calls[0];

    expect(path).toBe("/datos/sentimientos?dataset=2024-2");
    expect(resp).toEqual(mockJson);
  });
});
