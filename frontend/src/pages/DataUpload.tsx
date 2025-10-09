// src/pages/DataUpload.tsx
// Pantalla de “Datos” — Día 2 (Miembro B) + Integración de validación (Día 3 — Miembro B)
// - Consulta GET /datos/esquema
// - Formulario: archivo + periodo + overwrite
// - POST /datos/upload (mock) y muestra respuesta
// - POST /datos/validar (nuevo) para validar sin guardar y mostrar reporte
// - Usa el componente UploadDropzone (drag&drop + selector nativo)

import React, { useEffect, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import {
  getEsquema,
  uploadDatos,
  EsquemaCol,
  UploadResponse,
  dtypePrincipal,
  describeRestricciones,
  // nuevos imports del servicio de validación:
  validarDatos,
  ValidacionResponse,
  inferFormatFromFilename
} from "../services/datos";

import ValidationReport from "../components/ValidationReport";

export default function DataUpload() {
  const [columns, setColumns] = useState<EsquemaCol[]>([]);
  const [version, setVersion] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState("2024-2");
  const [overwrite, setOverwrite] = useState(false);

  const [fetching, setFetching] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // NUEVO: estado para validación
  const [validRes, setValidRes] = useState<ValidacionResponse | null>(null);
  const [valLoading, setValLoading] = useState(false);
  const [valError, setValError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setFetching(true);
      try {
        const schema = await getEsquema();
        setColumns(Array.isArray(schema.columns) ? schema.columns : []);
        setVersion(schema.version ?? "");
      } catch {
        setError("No se pudo obtener el esquema.");
      } finally {
        setFetching(false);
      }
    })();
  }, []);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    // La validación "en seco" es separada; aquí solo subimos el dataset.

    if (!file) {
      setError("Selecciona un archivo CSV/XLSX.");
      return;
    }
    if (!periodo) {
      setError("Ingresa un periodo (ej. 2024-2).");
      return;
    }

    setSubmitting(true);
    try {
      const r = await uploadDatos({ file, periodo, overwrite });
      setResult(r);
      // Si sube bien, no modificamos el resultado de validación previa (si existía).
    } catch {
      setError("Error al subir el archivo (revisa backend y/o CORS).");
    } finally {
      setSubmitting(false);
    }
  };

  // NUEVO: validar sin guardar
  const onValidate = async () => {
    setValError(null);
    setValidRes(null);

    if (!file) {
      setValError("Selecciona un archivo primero.");
      return;
    }

    setValLoading(true);
    try {
      // Inferimos formato por extensión (opcional)
      const fmt = inferFormatFromFilename(file?.name);
      const res = await validarDatos(file, fmt);
      setValidRes(res);
    } catch (e: any) {
      // Intentamos extraer mensaje de error del backend, si existe
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
    setValidRes(null);
    setValError(null);
    setError(null);
    setResult(null);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Encabezado */}
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Datos — Carga de Evaluaciones</h1>
        <p className="text-sm opacity-80">
          Esquema versión <strong>{version || "…"}</strong> (GET <code>/datos/esquema</code>)
        </p>
      </header>

      {/* Formulario de carga */}
      <form onSubmit={onSubmit} className="space-y-4">
        {/* Nota: en tu versión el dropzone usa onFileSelected (no onPick) */}
        <UploadDropzone onFileSelected={setFile} accept=".csv,.xlsx,.parquet" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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

          <div className="flex items-end gap-2">
            <button
              className="px-4 py-2 rounded-xl shadow"
              disabled={submitting}
              type="submit"
            >
              {submitting ? "Subiendo…" : "Subir dataset"}
            </button>

            {/* NUEVO: validar sin guardar */}
            <button
              type="button"
              className="px-4 py-2 rounded-xl border"
              disabled={valLoading || !file}
              onClick={onValidate}
              title="Valida el archivo sin almacenarlo"
            >
              {valLoading ? "Validando…" : "Validar sin guardar"}
            </button>

            {/* NUEVO: limpiar selección y reporte */}
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

      {/* NUEVO: errores de validación */}
      {valError && <div className="p-3 rounded-xl bg-red-50 text-red-700">{valError}</div>}

      {/* NUEVO: reporte de validación (KPIs + tabla de issues con filtros) */}
      <ValidationReport data={validRes} />

      {/* Tabla de esquema (ya estaba) */}
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
                  {/* Si tu servicio expone dtype como string|string[], mostramos el principal */}
                  <td className="p-2">
                    {typeof dtypePrincipal === "function"
                      ? dtypePrincipal(c.dtype as any)
                      : (c as any).dtype}
                  </td>
                  <td className="p-2">{c.required ? "Sí" : "No"}</td>
                  <td className="p-2">
                    {/* Mostrar restricciones si el servicio las provee (domain/range/pattern/max_len) */}
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
        {/* Notas de UX mínimas (Día 2) alineadas con las validaciones backend */}
        <div className="text-xs opacity-80 space-y-1">
          <p>
            *Los campos de PLN <em>no</em> van en la plantilla (se calcularán más adelante).
          </p>
          <p>
            *Los encabezados con espacios, acentos o “:” se aceptan y se{" "}
            <strong>normalizan automáticamente</strong> (p. ej.{" "}
            <code>“Código Materia”</code> → <code>codigo_materia</code>).
          </p>
          <p>
            *Se aplica <strong>coerción de tipos</strong> previa y se pueden exigir{" "}
            <strong>patrones (regex)</strong> por columna (p. ej.{" "}
            <code>periodo</code> = <code>^[0-9]{4}-(1|2)$</code>).
          </p>
        </div>
      </section>
    </div>
  );
}
