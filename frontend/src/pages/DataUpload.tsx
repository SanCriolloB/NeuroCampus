// frontend/src/pages/DataUpload.tsx
// Carga/validación de datasets:
// - GET  /datos/esquema  -> muestra la plantilla esperada
// - POST /datos/validar   -> validación en seco (no guarda)
// - POST /datos/upload    -> guarda el dataset con periodo (dataset_id) y overwrite

import React, { useEffect, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import ResultsTable from "../components/ResultsTable";
import {
  esquema as getEsquema,
  validar as validarDatos,
  upload as uploadDatos,
  type EsquemaResp,
  type ValidarResp,
  type UploadResp,
} from "../services/datos";

/* ===== Helpers locales ===== */

// Inferir formato a partir del nombre del archivo (opcional para logs/UI)
function inferFormatFromFilename(name?: string): "csv" | "xlsx" | "xls" | "parquet" | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (n.endsWith(".csv")) return "csv";
  if (n.endsWith(".xlsx")) return "xlsx";
  if (n.endsWith(".xls")) return "xls";
  if (n.endsWith(".parquet")) return "parquet";
  return undefined;
}

// Normalizar columnas del esquema: puede venir como `required: string[]`,
// o como `fields: {name, dtype, required, desc...}[]`. Rendereamos ambas.
type SchemaRow = { name: string; dtype?: string | null; required?: boolean; desc?: string | null };

function toSchemaRows(schema: EsquemaResp | null): SchemaRow[] {
  if (!schema) return [];
  if (Array.isArray(schema.fields) && schema.fields.length > 0) {
    return schema.fields.map((f) => ({
      name: f.name,
      dtype: typeof f.dtype === "string" ? f.dtype : undefined,
      required: f.required ?? undefined,
      desc: (f as any).desc ?? undefined,
    }));
  }
  // Fallback a listas simples
  const req = Array.isArray(schema.required) ? schema.required : [];
  const opt = Array.isArray(schema.optional) ? schema.optional : [];
  const rows: SchemaRow[] = [];
  req.forEach((name) => rows.push({ name, dtype: undefined, required: true, desc: "" }));
  opt.forEach((name) => rows.push({ name, dtype: undefined, required: false, desc: "" }));
  return rows;
}

export default function DataUpload() {
  // --- Esquema / metadatos
  const [schema, setSchema] = useState<EsquemaResp | null>(null);
  const [rows, setRows] = useState<SchemaRow[]>([]);
  const [version, setVersion] = useState<string>("");

  // --- Formulario
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState<string>("2024-2"); // dataset_id / periodo
  const [overwrite, setOverwrite] = useState<boolean>(false);

  // --- Estados de UI (subida)
  const [fetching, setFetching] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<UploadResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Validación (sin guardar)
  const [validRes, setValidRes] = useState<ValidarResp | null>(null);
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valError, setValError] = useState<string | null>(null);

  // Cargar esquema al montar
  useEffect(() => {
    (async () => {
      setFetching(true);
      try {
        const s = await getEsquema();
        setSchema(s);
        setRows(toSchemaRows(s));
        setVersion(s?.version ?? "");
      } catch (e: any) {
        setError(
          e?.response?.data?.detail ||
            e?.message ||
            "No se pudo obtener el esquema."
        );
      } finally {
        setFetching(false);
      }
    })();
  }, []);

  // Subir dataset (guardar)
  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Selecciona un archivo CSV/XLSX/Parquet.");
      return;
    }
    if (!periodo?.trim()) {
      setError("Ingresa un periodo (p. ej. 2024-2).");
      return;
    }

    setSubmitting(true);
    try {
      // El servicio upload ahora envía `periodo` (además de dataset_id) automáticamente
      const r = await uploadDatos(file, periodo.trim(), overwrite);
      setResult(r);
    } catch (e: any) {
      setError(
        e?.response?.data?.detail ||
          e?.message ||
          "Error al subir el archivo (verifica backend y CORS)."
      );
    } finally {
      setSubmitting(false);
    }
  }

  // Validar sin guardar
  async function onValidate() {
    setValError(null);
    setValidRes(null);

    if (!file) {
      setValError("Selecciona un archivo primero.");
      return;
    }

    setValLoading(true);
    try {
      // La API de validar solo necesita file (FormData), el formato es informativo
      const _fmt = inferFormatFromFilename(file.name);
      const res = await validarDatos(file);
      setValidRes(res);
    } catch (e: any) {
      setValError(
        e?.response?.data?.detail ||
          e?.message ||
          "Error al validar el archivo."
      );
    } finally {
      setValLoading(false);
    }
  }

  function onClear() {
    setFile(null);
    setResult(null);
    setError(null);
    setValidRes(null);
    setValError(null);
  }

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
        <UploadDropzone onFileSelected={setFile} accept=".csv,.xlsx,.xls,.parquet" />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <label className="block">
            <span className="text-sm">Periodo (dataset_id)</span>
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

            <button
              type="button"
              className="px-4 py-2 rounded-xl border"
              disabled={valLoading || !file}
              onClick={onValidate}
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
      {error && <div className="p-3 rounded-xl bg-red-100 text-red-800">{error}</div>}

      {/* Resultado de carga */}
      {result && (
        <div className="p-4 rounded-xl border space-y-2">
          <h2 className="font-semibold">Resultado de carga</h2>
          <div className="text-sm space-y-1">
            <div>
              <strong>dataset_id:</strong> {result.dataset_id ?? periodo}
            </div>
            <div>
              <strong>rows_ingested:</strong> {result.rows_ingested ?? 0}
            </div>
            <div>
              <strong>stored_as:</strong>{" "}
              <span className="mono">{result.stored_as ?? "—"}</span>
            </div>
            {typeof (result as any)?.message === "string" && (
              <div
                className="mono"
                style={{ color: (result as any)?.ok ? "#00c48c" : "#fca5a5" }}
              >
                message: {(result as any).message}
              </div>
            )}
          </div>
          {!result.ok && (
            <div className="mt-2 text-sm text-red-600">
              Ingesta no realizada. Revisa el mensaje del backend o usa “Validar sin guardar”.
            </div>
          )}
        </div>
      )}

      {/* Errores de validación */}
      {valError && (
        <div className="p-3 rounded-xl bg-red-50 text-red-700">{valError}</div>
      )}

      {/* Reporte de validación */}
      {validRes?.sample?.length ? (
        <div className="p-4 rounded-xl border space-y-2">
          <h2 className="font-semibold">Muestra del archivo validado</h2>
          <ResultsTable
            columns={Object.keys(validRes.sample[0]).slice(0, 8).map((k) => ({ key: k, header: k }))}
            rows={validRes.sample}
          />
        </div>
      ) : null}

      {/* Tabla de esquema */}
      <section className="space-y-2">
        <h2 className="font-semibold">Columnas esperadas (plantilla)</h2>
        <div className="overflow-auto border rounded-xl">
          <table className="min-w-full text-sm">
            <thead>
              <tr>
                <th className="text-left p-2">Columna</th>
                <th className="text-left p-2">Tipo</th>
                <th className="text-left p-2">Requerida</th>
                <th className="text-left p-2">Descripción</th>
              </tr>
            </thead>
            <tbody>
              {!fetching && rows.map((r) => (
                <tr key={r.name} className="border-t">
                  <td className="p-2">{r.name}</td>
                  <td className="p-2">{r.dtype ?? "-"}</td>
                  <td className="p-2">{r.required ? "Sí" : "No"}</td>
                  <td className="p-2">{r.desc ?? "-"}</td>
                </tr>
              ))}
              {fetching && (
                <tr>
                  <td className="p-2" colSpan={4}>
                    Cargando esquema…
                  </td>
                </tr>
              )}
              {!fetching && rows.length === 0 && (
                <tr>
                  <td className="p-2" colSpan={4}>
                    Sin datos (verifica backend).
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="text-xs opacity-80 space-y-1">
          <p>*Los campos de PLN no van en la plantilla (se calculan más adelante).</p>
          <p>*Los encabezados con espacios/acentos se normalizan automáticamente.</p>
          <p>*Se aplica coerción de tipos previa y se pueden exigir patrones por columna.</p>
        </div>
      </section>
    </div>
  );
}
