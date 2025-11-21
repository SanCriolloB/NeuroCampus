// frontend/src/pages/DataUpload.tsx
//
// Pestaña «Datos» del frontend NeuroCampus.
// Responsabilidades principales:
//
//  - Ingesta de nuevos datasets (carga y validación básica).
//  - Resumen estructural del dataset cargado.
//  - Análisis de sentimientos con BETO y visualizaciones asociadas.
//

import React, { useEffect, useMemo, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import ResultsTable from "../components/ResultsTable";
import {
  esquema as getEsquema,
  validar as validarDatos,
  upload as uploadDatos,
  resumen as getResumen,
  sentimientos as getSentimientos,
  type EsquemaResp,
  type ValidarResp,
  type UploadResp,
  type DatasetResumen,
  type DatasetSentimientos,
} from "../services/datos";
import {
  launchBetoPreproc,
  type BetoPreprocJob,
} from "../services/jobs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
} from "recharts";

/* ==========================================================================
 * 1. Configuración y helpers
 * ========================================================================== */

// Límite de tamaño del archivo de entrada (en MB)
const MAX_UPLOAD_MB =
  Number((import.meta as any).env?.VITE_MAX_UPLOAD_MB ?? 10) || 10;
const MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

// Extensiones soportadas en el frontend (deben coincidir con el backend)
const ALLOWED_EXT = [".csv", ".xlsx", ".xls", ".parquet"] as const;

/**
 * Inferir el formato a partir del nombre de archivo.
 * Solo se usa para mostrar información contextual en la UI.
 */
function inferFormatFromFilename(
  name?: string
): "csv" | "xlsx" | "xls" | "parquet" | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (n.endsWith(".csv")) return "csv";
  if (n.endsWith(".xlsx")) return "xlsx";
  if (n.endsWith(".xls")) return "xls";
  if (n.endsWith(".parquet")) return "parquet";
  return undefined;
}

/**
 * Validaciones mínimas en cliente:
 *  - extensión del archivo,
 *  - tamaño máximo permitido.
 */
function preflightChecks(file: File | null): { ok: boolean; message?: string } {
  if (!file)
    return { ok: false, message: "Selecciona un archivo CSV/XLSX/Parquet." };

  const lower = (file.name || "").toLowerCase();
  const hasAllowedExt = ALLOWED_EXT.some((ext) => lower.endsWith(ext));
  if (!hasAllowedExt) {
    return {
      ok: false,
      message: `Formato no soportado. Usa: ${ALLOWED_EXT.join(", ")}.`,
    };
  }

  if (file.size > MAX_BYTES) {
    return {
      ok: false,
      message: `El archivo pesa ${(file.size / (1024 * 1024)).toFixed(
        2
      )} MB y supera el límite de ${MAX_UPLOAD_MB} MB.`,
    };
  }
  return { ok: true };
}

/** Pequeño helper para usar esperas en el polling de BETO. */
const sleep = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

/* ==========================================================================
 * 2. Tipos locales y helpers de esquema (plantilla)
 * ========================================================================== */

/** Fila normalizada para la tabla de plantilla de columnas. */
type SchemaRow = {
  name: string;
  dtype?: string | null;
  required?: boolean;
  desc?: string | null;
};

/** Normaliza EsquemaResp (varias formas) a una lista de SchemaRow. */
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

  const req = Array.isArray(schema.required) ? schema.required : [];
  const opt = Array.isArray(schema.optional) ? schema.optional : [];
  const rows: SchemaRow[] = [];
  req.forEach((name) =>
    rows.push({ name, dtype: undefined, required: true, desc: "" })
  );
  opt.forEach((name) =>
    rows.push({ name, dtype: undefined, required: false, desc: "" })
  );
  return rows;
}

/* ==========================================================================
 * 3. Componente principal: DataUpload
 * ========================================================================== */

export default function DataUpload() {
  // --- Estado: esquema / plantilla ---
  const [schema, setSchema] = useState<EsquemaResp | null>(null);
  const [rows, setRows] = useState<SchemaRow[]>([]);
  const [version, setVersion] = useState<string>("");

  // --- Estado: formulario de ingesta ---
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState<string>("2024-2");
  const [overwrite, setOverwrite] = useState<boolean>(false);
  const [applyPreproc, setApplyPreproc] = useState<boolean>(true);
  const [runSentiment, setRunSentiment] = useState<boolean>(true);

  // --- Estado: respuestas de subida ---
  const [fetching, setFetching] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<UploadResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Estado: validación sin guardar (vista previa de filas) ---
  const [validRes, setValidRes] = useState<ValidarResp | null>(null);
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valError, setValError] = useState<string | null>(null);

  // --- Estado: resumen de dataset ---
  const [resumen, setResumen] = useState<DatasetResumen | null>(null);
  const [loadingResumen, setLoadingResumen] = useState<boolean>(false);

  // --- Estado: análisis de sentimientos (BETO) ---
  const [sentimientos, setSentimientos] =
    useState<DatasetSentimientos | null>(null);
  const [loadingSent, setLoadingSent] = useState<boolean>(false);
  const [sentError, setSentError] = useState<string | null>(null);
  const [betoJob, setBetoJob] = useState<BetoPreprocJob | null>(null);
  const [betoLaunching, setBetoLaunching] = useState<boolean>(false);

  // --- Estado: plantilla de columnas visible/oculta ---
  const [showSchemaDetails, setShowSchemaDetails] =
    useState<boolean>(false);

  /* ------------------------------------------------------------------------
   * Efectos
   * ------------------------------------------------------------------------ */

  // Cargar esquema (plantilla) al montar la página
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
            "No se pudo obtener el esquema de columnas."
        );
      } finally {
        setFetching(false);
      }
    })();
  }, []);

  // Polling de sentimientos tras lanzar BETO
  useEffect(() => {
    if (!betoJob?.id) return;
    let cancelled = false;

    async function pollSentimientos() {
      // Pequeña indicación de progreso local
      setBetoJob((prev) =>
        prev
          ? { ...prev, status: prev.status === "created" ? "running" : prev.status }
          : prev
      );

      setLoadingSent(true);
      setSentError(null);

      try {
        const dataset = periodo.trim();
        for (let attempt = 0; attempt < 10 && !cancelled; attempt++) {
          try {
            const sent = await getSentimientos({ dataset });
            if (cancelled) return;
            setSentimientos(sent);
            setBetoJob((prev) =>
              prev ? { ...prev, status: "done" as any } : prev
            );
            return;
          } catch (_err) {
            await sleep(3000);
          }
        }

        if (!cancelled) {
          setSentError(
            "No se pudo obtener el análisis de sentimientos tras ejecutar BETO."
          );
          setBetoJob((prev) =>
            prev ? { ...prev, status: "failed" as any } : prev
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingSent(false);
        }
      }
    }

    void pollSentimientos();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [betoJob?.id]);

  /* ------------------------------------------------------------------------
   * Datos derivados para gráficas de sentimientos
   * ------------------------------------------------------------------------ */

  const globalChartData = useMemo(
    () =>
      Array.isArray(sentimientos?.global_counts)
        ? sentimientos!.global_counts.map((it) => ({
            label:
              it.label === "pos"
                ? "Positivo"
                : it.label === "neu"
                ? "Neutro"
                : "Negativo",
            count: it.count,
            porcentaje: Math.round(it.proportion * 100),
          }))
        : [],
    [sentimientos]
  );

  const docentesChartData = useMemo(() => {
    if (!Array.isArray(sentimientos?.por_docente)) return [];
    return sentimientos!.por_docente
      .map((g) => {
        const counts = Array.isArray(g.counts) ? g.counts : [];
        const find = (lab: "neg" | "neu" | "pos") =>
          counts.find((c) => c.label === lab)?.count ?? 0;
        const neg = find("neg");
        const neu = find("neu");
        const pos = find("pos");
        const total = neg + neu + pos;
        return { group: g.group, neg, neu, pos, total };
      })
      .sort((a, b) => b.total - a.total)
      .slice(0, 10);
  }, [sentimientos]);

  /* ------------------------------------------------------------------------
   * Handlers de backend (ingesta, resumen, BETO)
   * ------------------------------------------------------------------------ */

  async function fetchResumen(datasetId: string) {
    const trimmed = datasetId.trim();
    if (!trimmed) return;

    setLoadingResumen(true);
    try {
      const res = await getResumen({ dataset: trimmed });
      setResumen(res);
    } catch (e) {
      console.error("Error obteniendo resumen de dataset:", e);
    } finally {
      setLoadingResumen(false);
    }
  }

  async function fetchResumenYSentimientos(datasetId: string) {
    await fetchResumen(datasetId);
    setLoadingSent(true);
    setSentError(null);
    try {
      const sent = await getSentimientos({ dataset: datasetId.trim() });
      setSentimientos(sent);
    } catch (e: any) {
      console.error("Error obteniendo sentimientos:", e);
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        "No se pudo obtener el análisis de sentimientos.";
      setSentError(msg);
      setSentimientos(null);
    } finally {
      setLoadingSent(false);
    }
  }

  async function runBeto(datasetId: string) {
    const trimmed = datasetId.trim();
    if (!trimmed) {
      setSentError(
        "Ingresa primero un periodo/dataset válido antes de lanzar BETO."
      );
      return;
    }
    setBetoLaunching(true);
    setSentError(null);
    try {
      const job = await launchBetoPreproc({ dataset: trimmed });
      setBetoJob(job);
    } catch (e: any) {
      console.error("Error lanzando BETO:", e);
      setSentError(
        e?.response?.data?.detail ||
          e?.message ||
          "Error al lanzar el análisis de sentimientos (BETO)."
      );
    } finally {
      setBetoLaunching(false);
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const pre = preflightChecks(file);
    if (!pre.ok) {
      setError(pre.message || "Archivo inválido.");
      return;
    }
    if (!periodo?.trim()) {
      setError("Ingresa un periodo (p. ej. 2024-2).");
      return;
    }

    const trimmed = periodo.trim();
    setSubmitting(true);
    try {
      const r = await uploadDatos(file as File, trimmed, overwrite);
      setResult(r);

      if (applyPreproc) {
        void fetchResumen(trimmed);
      }

      if (runSentiment) {
        void runBeto(trimmed);
      }
    } catch (e: any) {
      setError(
        e?.message ||
          e?.response?.data?.detail ||
          "Error al subir el archivo (verifica backend y CORS)."
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function onValidate() {
    setValError(null);
    setValidRes(null);

    const pre = preflightChecks(file);
    if (!pre.ok) {
      setValError(pre.message || "Archivo inválido para validar.");
      return;
    }

    setValLoading(true);
    try {
      const inferred = inferFormatFromFilename((file as File).name);
      const fmtNarrow =
        inferred === "csv" || inferred === "xlsx" || inferred === "parquet"
          ? inferred
          : inferred === "xls"
          ? "xlsx"
          : undefined;

      const res = await validarDatos(
        file as File,
        periodo.trim(),
        fmtNarrow ? { fmt: fmtNarrow } : undefined
      );
      setValidRes(res);
    } catch (e: any) {
      setValError(
        e?.message ||
          e?.response?.data?.detail ||
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
    setResumen(null);
    setSentimientos(null);
    setSentError(null);
    setBetoJob(null);
  }

  /* ------------------------------------------------------------------------
   * Render
   * ------------------------------------------------------------------------ */

  return (
    <div className="p-6 space-y-6">
      {/* Encabezado general */}
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">
          Datos <span className="opacity-60">/ Ingesta y análisis</span>
        </h1>
        <p className="text-xs opacity-70">
          Esquema de datos{" "}
          <span className="inline-flex items-center rounded-full border px-2 py-0.5 text-[11px]">
            v{version || "…"}
          </span>
        </p>
      </header>

      {/* Layout principal: 2 columnas (≈40% / 60%) */}
      <div className="grid gap-6 md:grid-cols-[minmax(0,0.4fr)_minmax(0,0.6fr)]">
        {/* COLUMNA IZQUIERDA: Ingreso de dataset */}
        <div className="space-y-4">
          <section className="space-y-4 rounded-2xl bg-white/5 p-4 shadow">
            <h2 className="text-lg font-semibold">Ingreso de dataset</h2>

            <UploadDropzone
              onFileSelected={setFile}
              accept=".csv,.xlsx,.xls,.parquet"
            />

            {file && (
              <div className="text-xs opacity-80">
                Archivo: <b>{file.name}</b> —{" "}
                {(file.size / (1024 * 1024)).toFixed(2)} MB — formato:{" "}
                <code>{inferFormatFromFilename(file.name) ?? "desconocido"}</code>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="block">
                <span className="text-sm">Periodo (dataset_id)</span>
                <input
                  aria-label="dataset id"
                  className="w-full border rounded-xl p-2 bg-transparent"
                  value={periodo}
                  onChange={(e) => setPeriodo(e.target.value)}
                  placeholder="2024-2"
                />
              </label>

              <label className="flex items-center gap-2 text-sm justify-end md:justify-start">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                />
                <span>Sobrescribir si el dataset ya existe</span>
              </label>
            </div>

            <div className="space-y-2 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={applyPreproc}
                  onChange={(e) => setApplyPreproc(e.target.checked)}
                />
                <span>Aplicar preprocesamiento y actualizar resumen</span>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={runSentiment}
                  onChange={(e) => setRunSentiment(e.target.checked)}
                />
                <span>Ejecutar análisis de sentimientos con BETO</span>
              </label>
            </div>

            <div className="flex flex-wrap gap-2 justify-between items-center">
              <button
                className="px-4 py-2 rounded-xl shadow bg-blue-600 text-white disabled:opacity-60"
                disabled={submitting}
                type="button"
                onClick={onSubmit}
              >
                {submitting ? "Procesando…" : "Cargar y procesar"}
              </button>

              <div className="flex gap-2">
                <button
                  type="button"
                  className="px-3 py-2 rounded-xl border text-xs"
                  disabled={valLoading || !file}
                  onClick={onValidate}
                >
                  {valLoading ? "Validando…" : "Validar sin guardar"}
                </button>
                <button
                  type="button"
                  className="px-3 py-2 rounded-xl border text-xs"
                  onClick={onClear}
                >
                  Limpiar
                </button>
              </div>
            </div>

            {error && (
              <div className="mt-2 p-3 rounded-xl bg-red-100 text-red-800 text-sm">
                {error}
              </div>
            )}

            {result && (
              <div className="mt-2 p-3 rounded-2xl border text-sm space-y-1">
                <div>
                  <strong>dataset_id:</strong> {result.dataset_id ?? periodo}
                </div>
                <div>
                  <strong>rows_ingested:</strong>{" "}
                  {(result as any).rows_ingested ?? 0}
                </div>
                <div>
                  <strong>stored_as:</strong>{" "}
                  <span className="mono">{result.stored_as ?? "—"}</span>
                </div>
                {typeof (result as any)?.message === "string" && (
                  <div className="mono">
                    {((result as any)?.ok ?? false)
                      ? "Ingesta realizada correctamente."
                      : (result as any).message}
                  </div>
                )}
              </div>
            )}
          </section>
        </div>

        {/* COLUMNA DERECHA: Resumen + tablas + BETO */}
        <div className="space-y-4">
          {/* Resumen del dataset (arriba derecha) */}
          <section className="space-y-3 rounded-2xl bg-white/5 p-4 shadow">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold">Resumen del dataset</h2>
                <p className="text-xs opacity-70">
                  Filas, columnas, periodos y principales indicadores.
                </p>
              </div>
              <button
                type="button"
                className="px-3 py-1 text-xs rounded-xl border"
                onClick={() => fetchResumenYSentimientos(periodo)}
                disabled={!periodo || loadingResumen || loadingSent}
              >
                {loadingResumen || loadingSent ? "Actualizando…" : "Actualizar"}
              </button>
            </div>

            {resumen ? (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div className="p-3 rounded-2xl bg-slate-900/40">
                    <div className="text-[10px] uppercase opacity-60">Filas</div>
                    <div className="text-xl font-semibold">
                      {resumen.n_rows}
                    </div>
                  </div>
                  <div className="p-3 rounded-2xl bg-slate-900/40">
                    <div className="text-[10px] uppercase opacity-60">
                      Columnas
                    </div>
                    <div className="text-xl font-semibold">
                      {resumen.n_cols}
                    </div>
                  </div>
                  {resumen.n_docentes != null && (
                    <div className="p-3 rounded-2xl bg-slate-900/40">
                      <div className="text-[10px] uppercase opacity-60">
                        Docentes
                      </div>
                      <div className="text-xl font-semibold">
                        {resumen.n_docentes}
                      </div>
                    </div>
                  )}
                  {resumen.n_asignaturas != null && (
                    <div className="p-3 rounded-2xl bg-slate-900/40">
                      <div className="text-[10px] uppercase opacity-60">
                        Asignaturas
                      </div>
                      <div className="text-xl font-semibold">
                        {resumen.n_asignaturas}
                      </div>
                    </div>
                  )}
                </div>

                <div className="text-xs opacity-75 space-y-1">
                  {Array.isArray(resumen.periodos) &&
                    resumen.periodos.length > 0 && (
                      <p>
                        <span className="font-medium">Periodos:</span>{" "}
                        {resumen.periodos.join(", ")}
                      </p>
                    )}
                  {(resumen.fecha_min || resumen.fecha_max) && (
                    <p>
                      <span className="font-medium">Rango de fechas:</span>{" "}
                      {resumen.fecha_min || "?"} — {resumen.fecha_max || "?"}
                    </p>
                  )}
                </div>

                {/* Tabla de columnas como resumen estructural */}
                <div className="rounded-2xl border overflow-auto max-h-64">
                  <table className="min-w-full text-xs">
                    <thead className="bg-slate-900/60">
                      <tr>
                        <th className="px-3 py-2 text-left">Columna</th>
                        <th className="px-3 py-2 text-left">Tipo</th>
                        <th className="px-3 py-2 text-right">No nulos</th>
                        <th className="px-3 py-2 text-left">Ejemplos</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(resumen.columns ?? []).map((col) => (
                        <tr
                          key={col.name}
                          className="border-t border-slate-800"
                        >
                          <td className="px-3 py-1 font-mono text-[11px]">
                            {col.name}
                          </td>
                          <td className="px-3 py-1">{col.dtype}</td>
                          <td className="px-3 py-1 text-right">
                            {col.non_nulls}
                          </td>
                          <td className="px-3 py-1">
                            {col.sample_values?.slice(0, 3).join(" · ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="text-sm opacity-70">
                Aún no hay resumen para este dataset. Sube un archivo y pulsa
                «Cargar y procesar» o «Actualizar».
              </p>
            )}
          </section>

          {/* Vista previa de filas (usa la muestra de /datos/validar) */}
          {validRes?.sample?.length ? (
            <section className="p-4 rounded-2xl bg-white/5 shadow space-y-2">
              <h3 className="font-semibold text-sm">
                Vista previa del dataset (primeras filas)
              </h3>
              <ResultsTable
                columns={Object.keys(validRes.sample[0])
                  .slice(0, 8)
                  .map((k) => ({ key: k, header: k }))}
                rows={validRes.sample}
              />
            </section>
          ) : null}

          {/* Panel de análisis de sentimientos (parte baja) */}
          <section className="space-y-3 rounded-2xl bg-white/5 p-4 shadow">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold">
                  Análisis de sentimientos (BETO)
                </h2>
                <p className="text-xs opacity-70">
                  Distribución global y por docente de comentarios positivos,
                  neutros y negativos.
                </p>
              </div>
              <div className="flex flex-col items-end gap-1">
                {betoJob && (
                  <div className="text-[10px] opacity-70 text-right">
                    <div className="font-semibold">Job BETO</div>
                    <div className="font-mono break-all">{betoJob.id}</div>
                    <div>estado: {betoJob.status}</div>
                  </div>
                )}
                <button
                  type="button"
                  className="px-3 py-1 text-xs rounded-xl border"
                  onClick={() => runBeto(periodo)}
                  disabled={betoLaunching || !periodo}
                >
                  {betoLaunching ? "Lanzando BETO…" : "Ejecutar BETO"}
                </button>
              </div>
            </div>

            {sentError && (
              <div className="p-2 text-xs rounded-xl bg-red-50 text-red-700">
                {sentError}
              </div>
            )}

            {sentimientos && (
              <p className="text-xs opacity-80">
                Total de comentarios analizados:{" "}
                <strong>{sentimientos.total_comentarios}</strong>
              </p>
            )}

            {/* Layout interno de gráficas: 2 cards horizontales */}
            <div className="grid gap-3 lg:grid-cols-2">
              {/* Distribución global (barras horizontales) */}
              <div className="h-56 rounded-2xl bg-slate-900/40 p-3">
                <h3 className="text-xs font-semibold mb-2">
                  Distribución global
                </h3>
                {globalChartData.length > 0 ? (
                  <ResponsiveContainer>
                    <BarChart
                      data={globalChartData}
                      layout="vertical"
                      margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis
                        type="category"
                        dataKey="label"
                        width={80}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip />
                      {/* Verde para positivo, gris para neutro, rojo para negativo.
                         Como aquí hay una sola barra por fila (count), usamos verde
                         como color base para la serie. */}
                      <Bar dataKey="count" name="Comentarios" fill="#22c55e" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-xs opacity-60 text-center">
                    {loadingSent
                      ? "Cargando análisis de sentimientos…"
                      : "Sin datos de sentimientos aún. Ejecuta BETO o pulsa Actualizar."}
                  </div>
                )}
              </div>

              {/* Distribución por docente (barras apiladas horizontales) */}
              <div className="h-64 rounded-2xl bg-slate-900/40 p-3">
                <h3 className="text-xs font-semibold mb-2">
                  Por docente (top 10 por número de comentarios)
                </h3>
                {docentesChartData.length > 0 ? (
                  <ResponsiveContainer>
                    <BarChart
                      data={docentesChartData}
                      layout="vertical"
                      margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis
                        type="category"
                        dataKey="group"
                        width={100}
                        tick={{ fontSize: 10 }}
                      />
                      <Tooltip />
                      <Legend />
                      {/* Colores consistentes para sentimientos */}
                      <Bar
                        dataKey="neg"
                        name="Negativo"
                        stackId="a"
                        fill="#ef4444"
                      />
                      <Bar
                        dataKey="neu"
                        name="Neutro"
                        stackId="a"
                        fill="#9ca3af"
                      />
                      <Bar
                        dataKey="pos"
                        name="Positivo"
                        stackId="a"
                        fill="#22c55e"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-xs opacity-60 text-center">
                    {loadingSent
                      ? "Cargando análisis de sentimientos…"
                      : "Sin datos agregados por docente. Ejecuta BETO o pulsa Actualizar."}
                  </div>
                )}
              </div>
            </div>

            {!sentimientos && !loadingSent && !sentError && (
              <p className="text-xs opacity-70">
                Para ver el análisis de sentimientos: 1) Asegúrate de que existe
                un dataset procesado para el periodo indicado. 2) Marca
                «Ejecutar análisis de sentimientos» al cargar, o usa el botón
                «Ejecutar BETO». 3) Usa «Actualizar» para refrescar los
                resultados.
              </p>
            )}
          </section>
        </div>
      </div>

      {/* Plantilla de columnas (para referencia, plegable) */}
      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-sm">
            Plantilla de columnas (para referencia)
          </h2>
          <button
            type="button"
            className="text-xs underline"
            onClick={() => setShowSchemaDetails((v) => !v)}
          >
            {showSchemaDetails ? "Ocultar plantilla" : "Ver plantilla"}
          </button>
        </div>

        {showSchemaDetails && (
          <>
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
                  {!fetching &&
                    rows.map((r) => (
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
                        No se pudo cargar la plantilla de columnas desde el
                        backend.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="text-xs opacity-80 space-y-1">
              <p>
                · Esta plantilla describe las columnas esperadas en los
                datasets de evaluación docente.
              </p>
              <p>
                · Algunos campos derivados de PLN se calculan en etapas
                posteriores del pipeline.
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
