// frontend/src/pages/DataUpload.test.tsx
//
// Test básico de la página Datos / Ingesta y análisis.
// Verifica que, al subir un dataset con rows_ingested > 0,
// se muestra el bloque de resultado de carga.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import DataUpload from "./DataUpload";
import {
  esquema,
  upload,
  validar,
  resumen,
  sentimientos,
} from "../services/datos";
import { launchBetoPreproc } from "../services/jobs";

// Mock de los servicios usados por la página
vi.mock("../services/datos", () => ({
  __esModule: true,
  esquema: vi.fn(),
  upload: vi.fn(),
  validar: vi.fn(),
  resumen: vi.fn(),
  sentimientos: vi.fn(),
}));

vi.mock("../services/jobs", () => ({
  __esModule: true,
  launchBetoPreproc: vi.fn(),
}));

const esquemaMock: any = {
  version: "0.3.0",
  fields: [
    { name: "docente", dtype: "string", required: true },
    { name: "asignatura", dtype: "string", required: true },
  ],
};

const uploadRespMock: any = {
  ok: true,
  dataset_id: "2020-1",
  rows_ingested: 3,
  stored_as: "localfs://datasets/2020-1.parquet",
  message: "ok",
};

const resumenMockResp: any = {
  dataset_id: "2020-1",
  n_rows: 100,
  n_cols: 10,
  n_docentes: 5,
  n_asignaturas: 8,
  periodos: ["2020-1"],
  fecha_min: "2024-01-01",
  fecha_max: "2024-04-01",
  columns: [
    {
      name: "docente",
      dtype: "string",
      non_nulls: 100,
      sample_values: ["Docente 1", "Docente 2"],
    },
  ],
};

const sentimientosMockResp: any = {
  dataset_id: "2020-1",
  total_comentarios: 10,
  global_counts: [
    { label: "pos", count: 6, proportion: 0.6 },
    { label: "neu", count: 2, proportion: 0.2 },
    { label: "neg", count: 2, proportion: 0.2 },
  ],
  por_docente: [],
  por_asignatura: [],
};

const betoJobMock: any = {
  id: "beto-123",
  dataset: "2020-1",
  src: "data/processed/2020-1.parquet",
  dst: "data/labeled/2020-1_beto.parquet",
  status: "created",
  created_at: "2025-01-01T00:00:00Z",
};

describe("DataUpload page", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // @ts-ignore
    (esquema as vi.Mock).mockResolvedValue(esquemaMock);
    // @ts-ignore
    (upload as vi.Mock).mockResolvedValue(uploadRespMock);
    // @ts-ignore
    (validar as vi.Mock).mockResolvedValue({ ok: true, sample: [] });
    // @ts-ignore
    (resumen as vi.Mock).mockResolvedValue(resumenMockResp);
    // @ts-ignore
    (sentimientos as vi.Mock).mockResolvedValue(sentimientosMockResp);
    // @ts-ignore
    (launchBetoPreproc as vi.Mock).mockResolvedValue(betoJobMock);
  });

  it("muestra éxito cuando rows_ingested > 0", async () => {
    render(<DataUpload />);

    // Esperar a que se monte correctamente la página (efecto de esquema)
    await screen.findByText(/ingreso de dataset/i);

    // Seleccionar archivo mediante el input oculto del dropzone
    const fileInput = (await screen.findByLabelText(/archivo/i)) as HTMLInputElement;
    const file = new File(["a,b\n1,2"], "sample.csv", { type: "text/csv" });

    fireEvent.change(fileInput, {
      target: { files: [file] },
    });

    // Click en el nuevo botón principal "Cargar y procesar"
    const btn = screen.getByRole("button", { name: /cargar y procesar/i });
    fireEvent.click(btn);

    // Esperar a que se haya llamado al servicio de upload
    await waitFor(() => {
      expect(upload).toHaveBeenCalledTimes(1);
    });

    // Verificar que se muestra el bloque de resultado con dataset_id y rows_ingested
    expect(await screen.findByText(/dataset_id:/i)).toBeInTheDocument();
    expect(await screen.findByText(/rows_ingested:/i)).toBeInTheDocument();
  });
});
