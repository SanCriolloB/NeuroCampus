// src/pages/DataUpload.tsx
// Pantalla de Datos — Día 2 base + Integración de validación (Día 3)
// - GET /datos/esquema (tabla de columnas esperadas)
// - POST /datos/upload (subir dataset)
// - POST /datos/validar (validación en seco con reporte)
// - Un ÚNICO UploadDropzone (no hay <input type="file"> adicional)

import React, { useEffect, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import ValidationReport from "../components/ValidationReport";

import {
  // esquema + utilidades que ya usabas
  getEsquema,
  EsquemaCol,
  UploadResponse,
  uploadDatos,
  dtypePrincipal,
  describeRestricciones,
  // validación v0.3.0
  validarDatos,
  ValidacionResponse,
  inferFormatFromFilename,
} from "../services/datos";

export default function DataUpload() {
  // --- Esquema / metadatos
  const [columns, setColumns] = useState<EsquemaCol[]>([]);
  const [version, setVersion] = useState<string>("");

  // --- Formulario
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState<string>("2024-2");
  const [overwrite, setOverwrite] = useState<boolean>(false);

  // --- Estados de UI
  const [fetching, setFetching] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Validación (nuevo Día 3)
  const [validRes, setValidRes] = useState<ValidacionResponse | null>(null);
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valError, setValError] = useState<string | null>(null);

  // Cargar esquema al montar
  useEffect(() => {
    (async () => {
      setFetching(true);
      try {
        const schema = await getEsquema();
        setColumns(Array.isArray(schema?.columns) ? schema.columns : []);
        setVersion(schema?.version ?? "");
      } catch {
        setError("No se pudo obtener el esquema.");
      } finally {
        setFetching(false);
      }
    })();
  }, []);

  // Subir dataset (NO valida; para eso está el botón aparte)
  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Selecciona un archivo CSV/XLSX/Parquet.");
      return;
    }
    if (!periodo) {
      setError("Ingresa un periodo (p. ej. 2024-2).");
      return;
    }

    setSubmitting(true);
    try {
      const r = await uploadDatos({ file, periodo, overwrite });
      setResult(r);
    } catch {
      setError("Error al subir el archivo (verifica backend y CORS).");
    } finally {
      setSubmitting(false);
    }
  };

  // Validar sin guardar (usa /datos/validar)
  const onValidate = async () => {
    setValError(null);
    setValidRes(null);

    if (!file) {
      setValError("Selecciona un archivo primero.");
      return;
    }

    setValLoading(true);
    try {
      const fmt = inferFormatFromFilename(file.name); // csv|xlsx|parquet|undefined
      const res = await validarDatos(file, fmt);
      setValidRes(res);
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        "Error al validar el archivo.";
      setValError(msg);
    } finally {
      setValLoading(false);
    }
  };

  const onClear = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setValidRes(null);
    setValError(null);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Encabezado */}
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Datos — Carga de Evaluaciones</h1>
        <p className="text-sm opacity-80">
          Esquema versión <strong>{version || "…"}</strong>{" "}
          <span className="opacity-60">(GET /datos/esquema)</span>
        </p>
      </header>

      {/* Formulario */}
      <form onSubmit={onSubmit} className="space-y-4">
        {/* ⚠️ IMPORTANTE: ÚNICO selector de archivo */}
        {/* Tu Dropzone actual usa onFileSelected; lo respetamos */}
        <UploadDropzone onFileSelected={setFile} accept=".csv,.xlsx,.parquet" />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <label className="block">
            <span className="text-sm">Periodo</span>
            <input
              className="w-full border rounded-xl p-2"
              value={periodo}
              onChange={(e) => setPeriodo(e.target.value)}
              placeholder="2024-2"
            />
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(e) => setOverwrite(e.target.checked)}
            />
            <span className="text-sm">Sobrescribir si existe</span>
          </label>

          <div className="flex gap-2 justify-start md:justify-end">
            <button
              className="px-4 py-2 rounded-xl shadow"
              disabled={submitting}
              type="submit"
            >
              {submitting ? "Subiendo…" : "Subir dataset"}
            </button>

            {/* Validación en seco */}
            <button
              type="button"
              className="px-4 py-2 rounded-xl border"
              disabled={valLoading || !file} // se habilita cuando hay archivo
              onClick={onValidate}
              title="Valida el archivo sin almacenarlo"
            >
              {valLoading ? "Validando…" : "Validar sin guardar"}
            </button>

            <button
              type="button"
              className="px-4 py-2 rounded-xl border"
              onClick={onClear}
            >
              Limpiar
            </button>
          </div>
        </div>
      </form>

      {/* Errores de carga */}
      {error && <div className="p-3 rounded-xl bg-red-100">{error}</div>}

      {/* Resultado de carga */}
      {result && (
        <div className="p-4 rounded-xl border space-y-2">
          <h2 className="font-semibold">Resultado de carga</h2>
          <div className="text-sm">
            <div><strong>dataset_id:</strong> {result.dataset_id}</div>
            <div><strong>rows_ingested:</strong> {result.rows_ingested}</div>
            <div><strong>stored_as:</strong> {result.stored_as}</div>
          </div>
          {Array.isArray(result.warnings) && result.warnings.length > 0 && (
            <div className="text-sm">
              <strong>Warnings:</strong>
              <ul className="list-disc ml-6">
                {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Errores de validación */}
      {valError && (
        <div className="p-3 rounded-xl bg-red-50 text-red-700">{valError}</div>
      )}

      {/* Reporte de validación (KPIs + tabla filtrable) */}
      <ValidationReport data={validRes} />

      {/* Tabla de esquema */}
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
              {!fetching && columns.map((c) => (
                <tr key={c.name} className="border-t">
                  <td className="p-2">{c.name}</td>
                  <td className="p-2">
                    {typeof dtypePrincipal === "function"
                      ? dtypePrincipal(c.dtype as any)
                      : (c as any).dtype}
                  </td>
                  <td className="p-2">{c.required ? "Sí" : "No"}</td>
                  <td className="p-2">
                    {typeof describeRestricciones === "function"
                      ? describeRestricciones(c)
                      : c.description || "-"}
                  </td>
                </tr>
              ))}
              {fetching && (
                <tr><td className="p-2" colSpan={4}>Cargando esquema…</td></tr>
              )}
              {!fetching && columns.length === 0 && (
                <tr><td className="p-2" colSpan={4}>Sin datos (verifica backend).</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="text-xs opacity-80 space-y-1">
          <p>*Los campos de PLN no van en la plantilla (se calcularán más adelante).</p>
          <p>*Los encabezados con espacios, acentos o “:” se aceptan y se normalizan automáticamente.</p>
          <p>*Se aplica coerción de tipos previa y se pueden exigir patrones (regex) por columna.</p>
        </div>
      </section>
    </div>
  );
}
