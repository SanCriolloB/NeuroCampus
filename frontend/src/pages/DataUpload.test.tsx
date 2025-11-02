import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import DataUpload from "./DataUpload";

describe("DataUpload page", () => {
  it("muestra éxito cuando rows_ingested > 0", async () => {
    // Mock de fetch para /datos/upload
    // @ts-ignore
    global.fetch.mockResolvedValue({
      ok: true,
      status: 201,
      statusText: "Created",
      headers: { get: () => "application/json" },
      json: async () => ({ ok: true, dataset_id: "2020-1", rows_ingested: 3, stored_as: "localfs://..." }),
    });

    render(<DataUpload />);

    // dataset_id/periodo input
    const dsInput = screen.getByLabelText(/dataset id/i);
    fireEvent.change(dsInput, { target: { value: "2020-1" } });

    // archivo
    const fileInput = screen.getByLabelText(/archivo/i);
    const file = new File(["a,b\n1,2"], "sample.csv", { type: "text/csv" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    // botón subir
    const btn = screen.getByRole("button", { name: /subir dataset/i });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByText(/Resultado de carga/i)).toBeInTheDocument();
      expect(screen.getByText(/Ingesta realizada correctamente/i)).toBeInTheDocument();
    });
  });
});
