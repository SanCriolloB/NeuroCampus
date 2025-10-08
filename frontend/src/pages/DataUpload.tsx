// src/pages/DataUpload.tsx
import React, { useEffect, useState } from "react";
import { getEsquema, uploadDatos, EsquemaCol, UploadResponse } from "../services/datos";

function UploadDrop({ onPick }: { onPick: (f: File) => void }) {
  return (
    <div className="border-2 border-dashed rounded-xl p-6 text-center">
      <p className="mb-3">Arrastra tu archivo CSV/XLSX aquí o selecciónalo:</p>
      <input
        type="file"
        accept=".csv,.xlsx"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />
    </div>
  );
}

export default function DataUpload() {
  const [columns, setColumns] = useState<EsquemaCol[]>([]);
  const [version, setVersion] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState("2024-2");
  const [overwrite, setOverwrite] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const schema = await getEsquema();
        setColumns(schema.columns);
        setVersion(schema.version);
      } catch {
        setError("No se pudo obtener el esquema.");
      }
    })();
  }, []);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!file) return setError("Selecciona un archivo.");
    if (!periodo) return setError("Ingresa un periodo (ej. 2024-2).");

    setSubmitting(true);
    try {
      const r = await uploadDatos({ file, periodo, overwrite });
      setResult(r);
    } catch {
      setError("Error al subir el archivo (revisa backend).");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Datos — Carga de Evaluaciones</h1>
        <p className="text-sm opacity-80">Esquema versión <strong>{version || "…"}</strong> (GET /datos/esquema)</p>
      </header>

      <form onSubmit={onSubmit} className="space-y-4">
        <UploadDrop onPick={setFile} />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <label className="block">
            <span className="text-sm">Periodo</span>
            <input className="w-full border rounded-xl p-2" value={periodo} onChange={(e) => setPeriodo(e.target.value)} />
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
            <span className="text-sm">Sobrescribir si existe</span>
          </label>
          <div className="flex items-end">
            <button className="px-4 py-2 rounded-xl shadow" disabled={submitting} type="submit">
              {submitting ? "Subiendo…" : "Subir dataset"}
            </button>
          </div>
        </div>
      </form>

      {error && <div className="p-3 rounded-xl bg-red-100">{error}</div>}

      {result && (
        <div className="p-4 rounded-xl border space-y-2">
          <h2 className="font-semibold">Resultado de carga</h2>
          <div className="text-sm">
            <div><strong>dataset_id:</strong> {result.dataset_id}</div>
            <div><strong>rows_ingested:</strong> {result.rows_ingested}</div>
            <div><strong>stored_as:</strong> {result.stored_as}</div>
          </div>
          {result.warnings?.length > 0 && (
            <div className="text-sm">
              <strong>Warnings:</strong>
              <ul className="list-disc ml-6">
                {result.warnings.map((w, i) => (<li key={i}>{w}</li>))}
              </ul>
            </div>
          )}
        </div>
      )}

      <section className="space-y-2">
        <h2 className="font-semibold">Columnas esperadas (plantilla)</h2>
        <div className="overflow-auto border rounded-xl">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <th className="text-left p-2">Columna</th>
                <th className="text-left p-2">Tipo</th>
                <th className="text-left p-2">Requerida</th>
                <th className="text-left p-2">Descripción</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((c) => (
                <tr key={c.name} className="border-t">
                  <td className="p-2">{c.name}</td>
                  <td className="p-2">{c.dtype}</td>
                  <td className="p-2">{c.required ? "Sí" : "No"}</td>
                  <td className="p-2">{c.description || "-"}</td>
                </tr>
              ))}
              {columns.length === 0 && (
                <tr><td className="p-2" colSpan={4}>Sin datos (verifica backend).</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-xs opacity-70">*Los campos de PLN NO van en la plantilla (se calculan más adelante).</p>
      </section>
    </div>
  );
}
