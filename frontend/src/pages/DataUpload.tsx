// frontend/src/pages/DataUpload.tsx
// Pestaña «Datos»: carga/validación de datasets y resumen básico +
// conexión con el job de análisis de sentimientos BETO.

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

/* ===== Helpers locales ===== */

// Límite de tamaño (MB) desde .env, por defecto 10 MB
const MAX_UPLOAD_MB =
  Number((import.meta as any).env?.VITE_MAX_UPLOAD_MB ?? 10) || 10;
const MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

// Extensiones soportadas en FE (deben coincidir con el BE)
const ALLOWED_EXT = [".csv", ".xlsx", ".xls", ".parquet"] as const;

// Inferir formato a partir del nombre del archivo (solo informativo/UI)
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

// Normalizar columnas del esquema: puede venir como `required: string[]`,
// o como `fields: {name, dtype, required, desc...}[]`.
type SchemaRow = {
  name: string;
  dtype?: string | null;
  required?: boolean;
  desc?: string | null;
};

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

// Validaciones mínimas en cliente: extensión y tamaño.
function preflightChecks(file: File | null): { ok: boolean; message?: string } {
  if (!file) return { ok: false, message: "Selecciona un archivo CSV/XLSX/Parquet." };

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
      )} MB y supera el límite de ${MAX_UPLOAD_MB} MB. Reduce el archivo o contacta a soporte.`,
    };
  }
  return { ok: true };
}

export default function DataUpload() {
  // Esquema / plantilla
  const [schema, setSchema] = useState<EsquemaResp | null>(null);
  const [rows, setRows] = useState<SchemaRow[]>([]);
  const [version, setVersion] = useState<string>("");

  // Formulario
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState<string>("2024-2"); // dataset_id / periodo
  const [overwrite, setOverwrite] = useState<boolean>(false);

  // Estados de subida
  const [fetching, setFetching] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<UploadResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Validación (sin guardar)
  const [validRes, setValidRes] = useState<ValidarResp | null>(null);
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valError, setValError] = useState<string | null>(null);

  // Resumen de dataset (panel derecho)
  const [resumen, setResumen] = useState<DatasetResumen | null>(null);
  const [loadingResumen, setLoadingResumen] = useState<boolean>(false);

  // Análisis de sentimientos (BETO) + job asociado
  const [sentimientos, setSentimientos] = useState<DatasetSentimientos | null>(null);
  const [loadingSent, setLoadingSent] = useState<boolean>(false);
  const [sentError, setSentError] = useState<string | null>(null);
  const [betoJob, setBetoJob] = useState<BetoPreprocJob | null>(null);
  const [betoLaunching, setBetoLaunching] = useState<boolean>(false);

  // Permite lanzar BETO automáticamente tras una carga exitosa
  const [autoLaunchBeto, setAutoLaunchBeto] = useState<boolean>(true);

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

  // Datos derivados para gráficas de sentimientos
  const globalChartData = useMemo(
    () =>
      sentimientos
        ? sentimientos.global_counts.map((it) => ({
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

  const docentesChartData = useMemo(
    () =>
      sentimientos
        ? sentimientos.por_docente
            .map((g) => {
              const find = (lab: "neg" | "neu" | "pos") =>
                g.counts.find((c) => c.label === lab)?.count ?? 0;
              const neg = find("neg");
              const neu = find("neu");
              const pos = find("pos");
              const total = neg + neu + pos;
              return { group: g.group, neg, neu, pos, total };
            })
            .sort((a, b) => b.total - a.total)
            .slice(0, 10)
        : [],
    [sentimientos]
  );

  async function fetchResumenYSentimientos(datasetId: string) {
    const trimmed = datasetId.trim();
    if (!trimmed) return;
    setSentError(null);

    setLoadingResumen(true);
    try {
      const res = await getResumen({ dataset: trimmed });
      setResumen(res);
    } catch (e) {
      console.error("Error obteniendo resumen de dataset:", e);
    } finally {
      setLoadingResumen(false);
    }

    setLoadingSent(true);
    try {
      const sent = await getSentimientos({ dataset: trimmed });
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
      setSentError("Ingresa primero un periodo/dataset válido antes de lanzar BETO.");
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

  // Subir dataset (guardar)
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

      // Refrescamos resumen + sentimientos para el dataset cargado
      void fetchResumenYSentimientos(trimmed);

      // Lanzamos BETO si está activado el modo automático
      if (autoLaunchBeto) {
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

  // Validar sin guardar
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

  return (
    <div className="p-6 space-y-6">
      {/* Encabezado principal */}
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Datos — Ingesta y análisis</h1>
        <p className="text-sm opacity-80">
          Esquema versión <strong>{version || "…"}</strong>{" "}
          <span className="opacity-60">(GET /datos/esquema)</span>
        </p>
        <p className="text-xs opacity-60">
          Ruta: Datos / Ingesta y análisis · Límite de archivo en cliente:{" "}
          <b>{MAX_UPLOAD_MB} MB</b>
        </p>
      </header>

      {/* Layout de dos columnas: izquierda (ingesta), derecha (resumen + sentimientos) */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.45fr)_minmax(0,0.55fr)]">
        {/* Columna izquierda: carga, validación y resultados básicos */}
        <div className="space-y-4">
          <form onSubmit={onSubmit} className="space-y-4">
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

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
              <label className="block">
                <span className="text-sm">Periodo (dataset_id)</span>
                <input
                  aria-label="dataset id"
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

            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={autoLaunchBeto}
                onChange={(e) => setAutoLaunchBeto(e.target.checked)}
              />
              <span>Lanzar análisis de sentimientos (BETO) tras la carga</span>
            </label>
          </form>

          {error && (
            <div className="p-3 rounded-xl bg-red-100 text-red-800">{error}</div>
          )}

          {result && (
            <div className="p-4 rounded-xl border space-y-2">
              <h2 className="font-semibold">Resultado de carga</h2>

              <div className="text-sm space-y-1">
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
                  <div
                    className="mono"
                    style={{ color: (result as any)?.ok ? "#00c48c" : "#fca5a5" }}
                  >
                    message: {(result as any).message}
                  </div>
                )}

                {Array.isArray((result as any)?.warnings) &&
                  (result as any).warnings.length > 0 && (
                    <div className="text-amber-400 mono">
                      warnings: {(result as any).warnings.join(", ")}
                    </div>
                  )}
              </div>

              {(() => {
                const ok = (result as any)?.ok ?? false;
                const ing = Number((result as any).rows_ingested ?? 0);

                if (ing > 0 || ok === true) {
                  return (
                    <div className="mt-2 text-sm text-green-500">
                      Ingesta realizada correctamente. Ya puedes continuar con el flujo.
                    </div>
                  );
                }

                return (
                  <div className="mt-2 text-sm text-yellow-400">
                    Ingesta no confirmada. Revisa el mensaje del backend o usa
                    «Validar sin guardar».
                  </div>
                );
              })()}
            </div>
          )}

          {valError && (
            <div className="p-3 rounded-xl bg-red-50 text-red-700">{valError}</div>
          )}

          {validRes?.sample?.length ? (
            <div className="p-4 rounded-xl border space-y-2">
              <h2 className="font-semibold">Muestra del archivo validado</h2>
              <ResultsTable
                columns={Object.keys(validRes.sample[0])
                  .slice(0, 8)
                  .map((k) => ({ key: k, header: k }))}
                rows={validRes.sample}
              />
            </div>
          ) : null}
        </div>

        {/* Columna derecha: resumen del dataset + análisis de sentimientos */}
        <div className="space-y-4">
          <section className="space-y-3 rounded-2xl bg-white shadow p-4">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold">Resumen del dataset</h2>
                <p className="text-xs opacity-70">
                  Filas, columnas, periodos y principales columnas detectadas.
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
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="p-3 rounded-2xl bg-slate-50">
                    <div className="text-[10px] uppercase opacity-60">Filas</div>
                    <div className="text-xl font-semibold">{resumen.n_rows}</div>
                  </div>
                  <div className="p-3 rounded-2xl bg-slate-50">
                    <div className="text-[10px] uppercase opacity-60">Columnas</div>
                    <div className="text-xl font-semibold">{resumen.n_cols}</div>
                  </div>
                  {resumen.n_docentes != null && (
                    <div className="p-3 rounded-2xl bg-slate-50">
                      <div className="text-[10px] uppercase opacity-60">Docentes</div>
                      <div className="text-xl font-semibold">
                        {resumen.n_docentes}
                      </div>
                    </div>
                  )}
                  {resumen.n_asignaturas != null && (
                    <div className="p-3 rounded-2xl bg-slate-50">
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

                <div className="rounded-2xl border overflow-auto max-h-64">
                  <table className="min-w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>
                        <th className="px-3 py-2 text-left">Columna</th>
                        <th className="px-3 py-2 text-left">Tipo</th>
                        <th className="px-3 py-2 text-right">No nulos</th>
                        <th className="px-3 py-2 text-left">Ejemplos</th>
                      </tr>
                    </thead>
                    <tbody>
                      {resumen.columns.map((col) => (
                        <tr key={col.name} className="border-t">
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
                «Actualizar».
              </p>
            )}
          </section>

          <section className="space-y-3 rounded-2xl bg-white shadow p-4">
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
                    <div className="font-semibold">Job BETO lanzado</div>
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

            {loadingSent ? (
              <p className="text-sm opacity-70">
                Cargando análisis de sentimientos…
              </p>
            ) : sentimientos ? (
              <>
                <p className="text-xs opacity-80">
                  Total de comentarios analizados:{" "}
                  <strong>{sentimientos.total_comentarios}</strong>
                </p>

                <div className="h-48 rounded-2xl bg-slate-50 p-3">
                  <h3 className="text-xs font-semibold mb-2">
                    Distribución global
                  </h3>
                  <ResponsiveContainer>
                    <BarChart data={globalChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" name="Comentarios" fill="#0f766e" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="h-64 rounded-2xl bg-slate-50 p-3">
                  <h3 className="text-xs font-semibold mb-2">
                    Por docente (top 10 por número de comentarios)
                  </h3>
                  <ResponsiveContainer>
                    <BarChart data={docentesChartData} stackOffset="none">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="group" tick={{ fontSize: 10 }} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="neg" name="Negativo" stackId="a" fill="#ef4444" />
                      <Bar dataKey="neu" name="Neutro" stackId="a" fill="#9ca3af" />
                      <Bar dataKey="pos" name="Positivo" stackId="a" fill="#22c55e" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            ) : (
              <p className="text-sm opacity-70">
                Para ver el análisis de sentimientos:
                <br />
                1) Asegúrate de que existe un dataset procesado para el periodo
                indicado.
                <br />
                2) Lanza BETO desde aquí o desde la pestaña Jobs.
                <br />
                3) Pulsa «Actualizar» en el panel de resumen.
              </p>
            )}
          </section>
        </div>
      </div>

      {/* Tabla de esquema (plantilla) a ancho completo */}
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
