// frontend/src/pages/DatosDiagnostico.tsx
import React, { useState, ChangeEvent, FormEvent } from "react";
import { validar, ValidarResp } from "@/services/datos";

type Fmt = "" | "csv" | "xlsx" | "parquet";

export default function DatosDiagnostico() {
  const [file, setFile] = useState<File | null>(null);
  const [datasetId, setDatasetId] = useState<string>("docentes");
  const [fmt, setFmt] = useState<Fmt>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [resp, setResp] = useState<ValidarResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
  };

  const onDatasetChange = (e: ChangeEvent<HTMLInputElement>) => {
    setDatasetId(e.target.value);
  };

  const onFmtChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setFmt(e.target.value as Fmt);
  };

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setResp(null);
    if (!file) {
      setError("Seleccione un archivo CSV/XLSX/Parquet.");
      return;
    }
    try {
      setLoading(true);
      const r = await validar(file, datasetId, { fmt: fmt || undefined });
      setResp(r);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg || "Error al validar");
    } finally {
      setLoading(false);
    }
  }

  const onClear = () => {
    setFile(null);
    setResp(null);
    setError(null);
  };

  return (
    <div className="p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Diagnóstico de Validación</h1>
        <p className="text-sm text-gray-500">
          Sube un archivo y valida contra el backend en <code>/datos/validar</code>.
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="flex items-center gap-3">
          <label className="w-32 text-sm" htmlFor="datasetId">Dataset ID</label>
          <input
            id="datasetId"
            className="border rounded px-3 py-2 w-64"
            value={datasetId}
            onChange={onDatasetChange}
            placeholder="docentes"
          />
        </div>

        <div className="flex items-center gap-3">
          <label className="w-32 text-sm" htmlFor="file">Archivo</label>
          <input
            id="file"
            type="file"
            accept=".csv,.xlsx,.parquet"
            onChange={onFileChange}
          />
        </div>

        <div className="flex items-center gap-3">
          <label className="w-32 text-sm" htmlFor="fmt">Forzar formato</label>
          <select
            id="fmt"
            className="border rounded px-3 py-2 w-64"
            value={fmt}
            onChange={onFmtChange}
          >
            <option value="">(auto por extensión)</option>
            <option value="csv">csv</option>
            <option value="xlsx">xlsx</option>
            <option value="parquet">parquet</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={loading}
            className="border rounded px-4 py-2 disabled:opacity-50"
          >
            {loading ? "Validando..." : "Validar"}
          </button>
          <button
            type="button"
            onClick={onClear}
            className="border rounded px-4 py-2"
          >
            Limpiar
          </button>
        </div>
      </form>

      {error && (
        <div className="text-red-600 text-sm">
          <b>Error:</b> {error}
        </div>
      )}

      {resp && (
        <section className="space-y-2">
          <div>
            <b>OK:</b> {String(resp.ok)} {resp.dataset_id ? `— ${resp.dataset_id}` : ""}
          </div>
          {resp.message && (
            <div>
              <b>Mensaje:</b> {resp.message}
            </div>
          )}
          {resp.missing?.length ? (
            <div>
              <b>Faltantes:</b> {resp.missing.join(", ")}
            </div>
          ) : null}
          {resp.extra?.length ? (
            <div>
              <b>Extras:</b> {resp.extra.join(", ")}
            </div>
          ) : null}
          <div className="mt-2">
            <b>Sample (primeras filas):</b>
            <pre className="bg-gray-50 border rounded p-3 text-xs overflow-auto">
{JSON.stringify(resp.sample ?? [], null, 2)}
            </pre>
          </div>
        </section>
      )}
    </div>
  );
}
