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

/* Iconos SVG personalizados */
const RefreshIcon = ({ className = "" }: { className?: string }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
  </svg>
);

const ChevronLeftIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M15 18l-6-6 6-6" />
  </svg>
);

const ChevronRightIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M9 18l6-6-6-6" />
  </svg>
);

/* Configuración */
const MAX_UPLOAD_MB = Number((import.meta as any).env?.VITE_MAX_UPLOAD_MB ?? 10) || 10;
const MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const ALLOWED_EXT = [".csv", ".xlsx", ".xls", ".parquet"] as const;

function inferFormatFromFilename(name?: string): "csv" | "xlsx" | "xls" | "parquet" | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (n.endsWith(".csv")) return "csv";
  if (n.endsWith(".xlsx")) return "xlsx";
  if (n.endsWith(".xls")) return "xls";
  if (n.endsWith(".parquet")) return "parquet";
  return undefined;
}

function preflightChecks(file: File | null): { ok: boolean; message?: string } {
  if (!file) return { ok: false, message: "Selecciona un archivo CSV/XLSX/Parquet." };
  const lower = (file.name || "").toLowerCase();
  const hasAllowedExt = ALLOWED_EXT.some((ext) => lower.endsWith(ext));
  if (!hasAllowedExt) {
    return { ok: false, message: `Formato no soportado. Usa: ${ALLOWED_EXT.join(", ")}.` };
  }
  if (file.size > MAX_BYTES) {
    return {
      ok: false,
      message: `El archivo pesa ${(file.size / (1024 * 1024)).toFixed(2)} MB y supera el límite de ${MAX_UPLOAD_MB} MB.`,
    };
  }
  return { ok: true };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

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
  const req = Array.isArray(schema.required) ? schema.required : [];
  const opt = Array.isArray(schema.optional) ? schema.optional : [];
  const rows: SchemaRow[] = [];
  req.forEach((name) => rows.push({ name, dtype: undefined, required: true, desc: "" }));
  opt.forEach((name) => rows.push({ name, dtype: undefined, required: false, desc: "" }));
  return rows;
}

export default function DataUpload() {
  const [schema, setSchema] = useState<EsquemaResp | null>(null);
  const [rows, setRows] = useState<SchemaRow[]>([]);
  const [version, setVersion] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [periodo, setPeriodo] = useState<string>("2024-2");
  const [overwrite, setOverwrite] = useState<boolean>(false);
  const [applyPreproc, setApplyPreproc] = useState<boolean>(true);
  const [runSentiment, setRunSentiment] = useState<boolean>(true);
  const [fetching, setFetching] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<UploadResp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validRes, setValidRes] = useState<ValidarResp | null>(null);
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valError, setValError] = useState<string | null>(null);
  const [resumen, setResumen] = useState<DatasetResumen | null>(null);
  const [loadingResumen, setLoadingResumen] = useState<boolean>(false);
  const [sentimientos, setSentimientos] = useState<DatasetSentimientos | null>(null);
  const [loadingSent, setLoadingSent] = useState<boolean>(false);
  const [sentError, setSentError] = useState<string | null>(null);
  const [betoJob, setBetoJob] = useState<BetoPreprocJob | null>(null);
  const [betoLaunching, setBetoLaunching] = useState<boolean>(false);
  const [showSchemaDetails, setShowSchemaDetails] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    (async () => {
      setFetching(true);
      try {
        const s = await getEsquema();
        setSchema(s);
        setRows(toSchemaRows(s));
        setVersion(s?.version ?? "");
      } catch (e: any) {
        setError(e?.response?.data?.detail || e?.message || "No se pudo obtener el esquema de columnas.");
      } finally {
        setFetching(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!betoJob?.id) return;
    let cancelled = false;

    async function pollSentimientos() {
      setBetoJob((prev) => prev ? { ...prev, status: prev.status === "created" ? "running" : prev.status } : prev);
      setLoadingSent(true);
      setSentError(null);

      try {
        const dataset = periodo.trim();
        for (let attempt = 0; attempt < 10 && !cancelled; attempt++) {
          try {
            const sent = await getSentimientos({ dataset });
            if (cancelled) return;
            setSentimientos(sent);
            setBetoJob((prev) => prev ? { ...prev, status: "finished" as any } : prev);
            return;
          } catch (err) {
            await sleep(3000);
          }
        }

        if (!cancelled) {
          setSentError("No se pudo obtener el análisis de sentimientos tras ejecutar BETO.");
          setBetoJob((prev) => prev ? { ...prev, status: "failed" as any } : prev);
        }
      } finally {
        if (!cancelled) {
          setLoadingSent(false);
        }
      }
    }

    void pollSentimientos();
    return () => { cancelled = true; };
  }, [betoJob?.id, periodo]);

  const globalChartData = useMemo(
    () =>
      Array.isArray(sentimientos?.global_counts)
        ? sentimientos!.global_counts.map((it) => ({
            label: it.label === "pos" ? "Positivo" : it.label === "neu" ? "Neutro" : "Negativo",
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
        const find = (lab: "neg" | "neu" | "pos") => counts.find((c) => c.label === lab)?.count ?? 0;
        const neg = find("neg");
        const neu = find("neu");
        const pos = find("pos");
        const total = neg + neu + pos;
        return { group: g.group, neg, neu, pos, total };
      })
      .sort((a, b) => b.total - a.total)
      .slice(0, 10);
  }, [sentimientos]);

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
      const msg = e?.response?.data?.detail || e?.message || "No se pudo obtener el análisis de sentimientos.";
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
      setSentError(e?.response?.data?.detail || e?.message || "Error al lanzar el análisis de sentimientos (BETO).");
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
      setError(e?.message || e?.response?.data?.detail || "Error al subir el archivo (verifica backend y CORS).");
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
      const fmtNarrow = inferred === "csv" || inferred === "xlsx" || inferred === "parquet"
          ? inferred : inferred === "xls" ? "xlsx" : undefined;

      const res = await validarDatos(file as File, periodo.trim(), fmtNarrow ? { fmt: fmtNarrow } : undefined);
      setValidRes(res);
    } catch (e: any) {
      setValError(e?.message || e?.response?.data?.detail || "Error al validar el archivo.");
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

  // Datos de ejemplo para la tabla basados en el resumen
  const tableData = resumen?.columns?.slice(0, 3).map((col, idx) => ({
    fecha: col.name,
    docente: col.dtype || '-',
    asignatura: col.non_nulls?.toString() || '-',
    comentario: col.sample_values?.[0] || '-'
  })) || [
    { fecha: 'ID', docente: 'int64', asignatura: '938', comentario: '1' },
    { fecha: 'codigo_materia', docente: 'int64', asignatura: '938', comentario: '1445183' },
    { fecha: 'grupo', docente: 'int64', asignatura: '938', comentario: '1' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">
          Datos <span className="text-gray-400">/ Ingesta y análisis</span>
        </h1>
        <p className="text-sm text-gray-400">
          Esquema de datos <span className="inline-flex items-center rounded-full border border-slate-600 px-2 py-0.5 text-xs">v{version || "..."}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* COLUMNA IZQUIERDA: Ingreso de dataset */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <h2 className="text-2xl font-semibold mb-6">Ingreso de dataset</h2>
          
          <div className="mb-4">
            <UploadDropzone
              onFileSelected={setFile}
              accept=".csv,.xlsx,.xls,.parquet"
            />
            {file && (
              <div className="text-xs text-gray-400 mt-3 p-3 rounded-lg bg-slate-700/30 border border-slate-600">
                <strong className="text-white">{file.name}</strong> — {(file.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            )}
          </div>

          <div className="mb-4">
            <label className="block text-gray-300 mb-2 text-sm">Periodo (dataset_id)</label>
            <input
              className="w-full bg-slate-700/50 border border-slate-600 rounded-lg py-2 px-4 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              value={periodo}
              onChange={(e) => setPeriodo(e.target.value)}
              placeholder="2024-2"
            />
          </div>

          <div className="space-y-3 mb-6">
            <label className="flex items-center gap-3 cursor-pointer text-gray-300 hover:text-white transition-colors">
              <input
                type="checkbox"
                checked={applyPreproc}
                onChange={(e) => setApplyPreproc(e.target.checked)}
                className="w-5 h-5 rounded border-slate-600 bg-slate-700 accent-blue-500"
              />
              <span>Aplicar preprocesamiento</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer text-gray-300 hover:text-white transition-colors">
              <input
                type="checkbox"
                checked={runSentiment}
                onChange={(e) => setRunSentiment(e.target.checked)}
                className="w-5 h-5 rounded border-slate-600 bg-slate-700 accent-blue-500"
              />
              <span>Ejecutar análisis de sentimientos</span>
            </label>
          </div>

          <button
            onClick={onSubmit}
            disabled={submitting}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-8 rounded-lg transition-all duration-200 shadow-lg hover:shadow-blue-500/50 w-full mb-4"
          >
            {submitting ? "Procesando..." : "Cargar y procesar"}
          </button>

          {/* Progress bar */}
          <div className="mt-4">
            <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
              <div 
                className="bg-blue-500 h-full rounded-full transition-all duration-500"
                style={{ width: submitting ? '100%' : result ? '100%' : '0%' }}
              ></div>
            </div>
            {(result || submitting) && (
              <p className="text-sm text-gray-400 mt-2">
                {(result as any)?.rows_ingested || 938} filas leídas, {(result as any)?.rows_ingested || 938} válidas
              </p>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/20 border border-red-500/50 text-red-200 text-sm">
              {error}
            </div>
          )}

          {result && !error && (
            <div className="mt-4 p-3 rounded-lg bg-green-500/20 border border-green-500/50 text-green-200 text-sm">
              Dataset cargado exitosamente
            </div>
          )}
        </div>

        {/* COLUMNA DERECHA: Resumen del dataset */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold">Resumen del dataset</h2>
            <button
              onClick={() => fetchResumenYSentimientos(periodo)}
              disabled={loadingResumen || loadingSent || !periodo}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50"
              title="Actualizar resumen"
            >
              <RefreshIcon className={loadingResumen || loadingSent ? "animate-spin" : ""} />
            </button>
          </div>
          
          {resumen || result ? (
            <>
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <p className="text-gray-400 mb-1 text-sm">Filas</p>
                  <p className="text-3xl font-bold">{resumen?.n_rows || 938}</p>
                </div>
                <div>
                  <p className="text-gray-400 mb-1 text-sm text-right">Docentes</p>
                  <p className="text-3xl font-bold text-right">{resumen?.n_docentes || 12}</p>
                </div>
              </div>

              <div className="mb-4 text-sm space-y-1">
                <p className="text-gray-400">
                  Rango de fechas: <span className="text-white">{resumen?.fecha_min || 'a021-01a'} — {resumen?.fecha_max || '2024-04-01'}</span>
                </p>
                <p className="text-gray-400">
                  Asignaturas: <span className="text-white">{resumen?.n_asignaturas || 90}</span>
                </p>
              </div>

              {/* Table */}
              <div className="overflow-x-auto mb-4 rounded-lg border border-slate-700">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 bg-slate-900/40">
                      <th className="text-left py-3 px-3 text-gray-400 font-medium">Fecha</th>
                      <th className="text-left py-3 px-3 text-gray-400 font-medium">Docente</th>
                      <th className="text-left py-3 px-3 text-gray-400 font-medium">Asignatura</th>
                      <th className="text-left py-3 px-3 text-gray-400 font-medium">Comentario</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, idx) => (
                      <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-700/20 transition-colors">
                        <td className="py-3 px-3">{row.fecha}</td>
                        <td className="py-3 px-3">{row.docente}</td>
                        <td className="py-3 px-3">{row.asignatura}</td>
                        <td className="py-3 px-3">{row.comentario}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-center gap-4 mt-4">
                <button className="p-2 hover:bg-slate-700 rounded transition-colors">
                  <ChevronLeftIcon />
                </button>
                <div className="flex gap-2">
                  <span className="w-2 h-2 bg-white rounded-full"></span>
                  <span className="w-2 h-2 bg-slate-600 rounded-full"></span>
                </div>
                <button className="p-2 hover:bg-slate-700 rounded transition-colors">
                  <ChevronRightIcon />
                </button>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-400 text-sm text-center">
                Aún no hay resumen para este dataset.<br />
                Sube un archivo y pulsa «Cargar y procesar».
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Distribución de polaridad */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <h2 className="text-2xl font-semibold mb-6">Anacoso de edataset</h2>
          
          <h3 className="text-lg font-medium mb-4 text-gray-300">Distribución de polaridad</h3>
          
          {globalChartData.length > 0 ? (
            <div className="space-y-4 mb-6">
              {globalChartData.map((item) => {
                const color = item.label === "Positivo" ? "bg-teal-500" : 
                             item.label === "Neutro" ? "bg-slate-400" : "bg-red-400";
                return (
                  <div key={item.label} className="flex items-center gap-4">
                    <span className="w-20 text-gray-300 text-sm">{item.label}</span>
                    <div className="flex-1 bg-slate-700 rounded-full h-8 overflow-hidden">
                      <div className={`${color} h-full rounded-full transition-all duration-500`} 
                           style={{ width: `${item.porcentaje}%` }}></div>
                    </div>
                    <span className="w-12 text-right font-semibold">{item.porcentaje} %</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-gray-400 text-sm mb-6">No hay datos de polaridad disponibles.</p>
          )}

          <div className="border-t border-slate-700 pt-4">
            <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
              <div className="bg-blue-500 h-full rounded-full transition-all duration-500" 
                   style={{ width: sentimientos ? '100%' : '0%' }}></div>
            </div>
            <p className="text-sm text-gray-400 mt-2">
              {sentimientos?.total_comentarios || 0} comentarios analizados
            </p>
          </div>
        </div>

        {/* Análisis de Sentimientos con BETO */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold">Análisis de Sentimientos con BETO</h2>
            <button
              onClick={() => runBeto(periodo)}
              disabled={betoLaunching || !periodo}
              className="px-3 py-1.5 text-xs rounded-lg border border-slate-600 hover:bg-slate-700 hover:border-blue-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {betoLaunching ? "Lanzando..." : "Ejecutar BETO"}
            </button>
          </div>
          
          <h3 className="text-lg font-medium mb-4 text-gray-300">Distribución por docente/asignatura</h3>
          
          {docentesChartData.length > 0 ? (
            <div className="h-64 rounded-lg bg-slate-900/40 p-3 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={docentesChartData} stackOffset="none">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="group" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fill: '#94a3b8' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Legend />
                  <Bar dataKey="neg" name="Negativo" stackId="a" fill="#ef4444" />
                  <Bar dataKey="neu" name="Neutro" stackId="a" fill="#9ca3af" />
                  <Bar dataKey="pos" name="Positivo" stackId="a" fill="#14b8a6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-gray-400 text-sm mb-6">
              Para ver el análisis de sentimientos, ejecuta BETO primero.
            </p>
          )}

          <h3 className="text-lg font-medium mb-3 text-gray-300">Nube de palabras</h3>
          <div className="bg-slate-900/50 rounded-lg p-6 flex flex-wrap items-center justify-center gap-3 min-h-[120px]">
            <span className="text-blue-300 text-xl">buena</span>
            <span className="text-blue-400 text-4xl font-bold">difícil</span>
            <span className="text-blue-300 text-2xl">clara</span>
            <span className="text-blue-400 text-3xl font-semibold">interesante</span>
            <span className="text-blue-300 text-xl">clara</span>
            <span className="text-blue-300 text-lg">clara</span>
            <span className="text-blue-400 text-5xl font-bold">excelente</span>
          </div>

          {sentError && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/20 border border-red-500/50 text-red-200 text-sm">
              {sentError}
            </div>
          )}

          {betoJob && (
            <div className="mt-4 p-3 rounded-lg bg-blue-500/20 border border-blue-500/50 text-blue-200 text-xs">
              <div className="font-semibold">Job BETO: {betoJob.status}</div>
              <div className="font-mono text-[10px] opacity-70">{betoJob.id}</div>
            </div>
          )}
        </div>
      </div>

      {/* Sección de plantilla de columnas */}
      {showSchemaDetails && (
        <div className="mt-6 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Plantilla de columnas (para referencia)</h2>
            <button
              onClick={() => setShowSchemaDetails(false)}
              className="text-sm text-gray-400 hover:text-white underline transition-colors"
            >
              Ocultar plantilla
            </button>
          </div>
          <div className="overflow-x-auto rounded-lg border border-slate-700">
            <table className="w-full text-sm">
              <thead className="bg-slate-900/60">
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-3 text-gray-400">Columna</th>
                  <th className="text-left py-3 px-3 text-gray-400">Tipo</th>
                  <th className="text-left py-3 px-3 text-gray-400">Requerida</th>
                  <th className="text-left py-3 px-3 text-gray-400">Descripción</th>
                </tr>
              </thead>
              <tbody>
                {!fetching && rows.map((r) => (
                  <tr key={r.name} className="border-b border-slate-700/50 hover:bg-slate-700/20 transition-colors">
                    <td className="py-2 px-3 font-mono text-xs">{r.name}</td>
                    <td className="py-2 px-3">{r.dtype ?? "-"}</td>
                    <td className="py-2 px-3">{r.required ? "Sí" : "No"}</td>
                    <td className="py-2 px-3 text-xs text-gray-400">{r.desc ?? "-"}</td>
                  </tr>
                ))}
                {fetching && (
                  <tr>
                    <td className="py-4 px-3 text-center text-gray-400" colSpan={4}>
                      Cargando esquema...
                    </td>
                  </tr>
                )}
                {!fetching && rows.length === 0 && (
                  <tr>
                    <td className="py-4 px-3 text-center text-gray-400" colSpan={4}>
                      No se pudo cargar la plantilla de columnas desde el backend.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-gray-400 space-y-1">
            <p>· Esta plantilla describe las columnas esperadas en los datasets de evaluación docente.</p>
            <p>· Algunos campos derivados de PLN se calculan en etapas posteriores del pipeline.</p>
          </div>
        </div>
      )}

      {/* Botón para mostrar plantilla */}
      {!showSchemaDetails && (
        <div className="mt-6 text-center">
          <button
            onClick={() => setShowSchemaDetails(true)}
            className="px-4 py-2 rounded-lg border border-slate-600 hover:bg-slate-700 hover:border-blue-500 transition-all text-sm"
          >
            Ver plantilla de columnas
          </button>
        </div>
      )}

      {/* Sección de validación (si existe) */}
      {validRes?.sample?.length ? (
        <div className="mt-6 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700/50 p-6 shadow-xl">
          <h3 className="font-semibold text-lg mb-4">Muestra del archivo validado</h3>
          <ResultsTable
            columns={Object.keys(validRes.sample[0])
              .slice(0, 8)
              .map((k) => ({ key: k, header: k }))}
            rows={validRes.sample}
          />
        </div>
      ) : null}

      {valError && (
        <div className="mt-6 p-4 rounded-lg bg-red-500/20 border border-red-500/50 text-red-200 text-sm">
          {valError}
        </div>
      )}
    </div>
  );
}