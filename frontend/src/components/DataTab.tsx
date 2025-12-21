import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Checkbox } from "./ui/checkbox";
import { Progress } from "./ui/progress";
import { Upload, CheckCircle2 } from "lucide-react";

import { useAppFilters, setAppFilters } from "@/state/appFilters.store";

import { useValidateDataset } from "@/features/datos/hooks/useValidateDataset";
import { useUploadDataset } from "@/features/datos/hooks/useUploadDataset";
import { useDatasetResumen } from "@/features/datos/hooks/useDatasetResumen";
import { useDatasetSentimientos } from "@/features/datos/hooks/useDatasetSentimientos";
import { useBetoPreprocJob } from "@/features/datos/hooks/useBetoPreprocJob";
import { jobsApi } from "@/features/datos/api";

import {
  mapGlobalSentiment,
  mapSampleRowsToPreview,
  mapTeacherSentiment,
  rowsReadValidFromValidation,
  UiPreviewRow,
} from "@/features/datos/mappers";

const DEFAULT_SENTIMENT_DISTRIBUTION = [
  { name: "Positive", value: 450, percentage: 45 },
  { name: "Neutral", value: 350, percentage: 35 },
  { name: "Negative", value: 200, percentage: 20 },
];

const DEFAULT_SENTIMENT_BY_TEACHER = [
  { teacher: "Dr. García", positive: 45, neutral: 35, negative: 20 },
  { teacher: "Prof. Martínez", positive: 40, neutral: 40, negative: 20 },
  { teacher: "Dr. López", positive: 50, neutral: 30, negative: 20 },
  { teacher: "Prof. Rodríguez", positive: 35, neutral: 45, negative: 20 },
  { teacher: "Dr. Fernández", positive: 42, neutral: 38, negative: 20 },
];

const DEFAULT_SAMPLE_DATA: UiPreviewRow[] = [
  { id: 1, teacher: "Dr. García", subject: "Calculus I", rating: 4.5, comment: "Excellent methodology" },
  { id: 2, teacher: "Prof. Martínez", subject: "Physics II", rating: 4.2, comment: "Clear explanations" },
  { id: 3, teacher: "Dr. López", subject: "Programming", rating: 4.7, comment: "Very helpful" },
  { id: 4, teacher: "Prof. Rodríguez", subject: "Chemistry", rating: 3.8, comment: "Good class" },
  { id: 5, teacher: "Dr. Fernández", subject: "Mathematics", rating: 4.0, comment: "Well organized" },
];

const COLORS = {
  positive: "#10B981",
  neutral: "#F59E0B",
  negative: "#EF4444",
};

export function DataTab() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Global filters (store)
  const activePeriodo = useAppFilters((s) => s.activePeriodo) ?? "2025-1";
  const programa = useAppFilters((s) => s.programa) ?? "Engineering";
  const activeDatasetId = useAppFilters((s) => s.activeDatasetId);

  // UI local state (mantiene la dinámica del prototipo)
  const [isProcessing, setIsProcessing] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [applyPreprocessing, setApplyPreprocessing] = useState(true);
  const [runSentiment, setRunSentiment] = useState(true);

  const [datasetName, setDatasetName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [betoJobId, setBetoJobId] = useState<string | null>(null);

  // Hooks de backend
  const validate = useValidateDataset();
  const upload = useUploadDataset();

  // Dataset “activo” para consultas (si ya existe de una sesión previa, debe cargar)
  const datasetForQueries = activeDatasetId ?? (dataLoaded ? activePeriodo : null);

  const resumen = useDatasetResumen(datasetForQueries);
  const sentimientos = useDatasetSentimientos(datasetForQueries);
  const betoJob = useBetoPreprocJob(betoJobId);

  useEffect(() => {
    if (activeDatasetId) setDataLoaded(true);
  }, [activeDatasetId]);

  // Cuando BETO termina, refrescamos sentimientos
  useEffect(() => {
    if (betoJob.job?.status === "done" && datasetForQueries) {
      void sentimientos.refetch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [betoJob.job?.status, datasetForQueries]);

  const previewRows = useMemo(() => {
    const mapped = mapSampleRowsToPreview(validate.data?.sample);
    return mapped.length ? mapped : DEFAULT_SAMPLE_DATA;
  }, [validate.data?.sample]);

  const { rowsRead, rowsValid } = useMemo(() => rowsReadValidFromValidation(validate.data), [validate.data]);

  const kpiRows = resumen.data?.n_rows ?? validate.data?.n_rows ?? 1000;
  const kpiCols = resumen.data?.n_cols ?? validate.data?.n_cols ?? 15;
  const kpiTeachers = resumen.data?.n_docentes ?? 45;

  const sentimentDistribution =
    (sentimientos.data && mapGlobalSentiment(sentimientos.data).length)
      ? mapGlobalSentiment(sentimientos.data)
      : DEFAULT_SENTIMENT_DISTRIBUTION;

  const sentimentByTeacher =
    (sentimientos.data && mapTeacherSentiment(sentimientos.data).length)
      ? mapTeacherSentiment(sentimientos.data)
      : DEFAULT_SENTIMENT_BY_TEACHER;

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function onFileSelected(f: File | null) {
    setFile(f);
    setErrorMsg(null);
  }

  async function handleProcess() {
    setErrorMsg(null);

    if (!file) {
      setErrorMsg("Select a file first.");
      return;
    }

    setIsProcessing(true);

    try {
      // 1) Validación previa (no cambia UI)
      const validateId = (datasetName.trim() || activePeriodo).trim();
      const v = await validate.run(file, validateId);

      // Si el backend marca ok=false o hay errores severos, detenemos.
      const hasSevere =
        Array.isArray(v.issues) && v.issues.some((i) => i.level === "error");

      if (v.ok === false || hasSevere) {
        setErrorMsg("Validation failed. Please check the dataset format.");
        setIsProcessing(false);
        return;
      }

      // 2) Upload real (con progreso)
      const periodo = activePeriodo;
      let up: any;

      try {
        up = await upload.run(file, periodo, false);
      } catch (e: any) {
        const status = e?.response?.status;
        const msg = String(e?.message ?? "");
        const is409 = status === 409 || msg.startsWith("HTTP 409");

        if (!is409) throw e;

        const ok = window.confirm(
          `El dataset '${periodo}' ya existe. ¿Deseas reemplazarlo (overwrite)?`,
        );
        if (!ok) throw e;

        up = await upload.run(file, periodo, true);
      }

      // 3) Setear contexto global (clave para cross-tab futuro)
      setAppFilters({
        activeDatasetId: up.dataset_id ?? periodo,
        activePeriodo: periodo,
        programa,
      });

      setDataLoaded(true);

      // 4) Si el usuario pidió sentimientos, lanzar BETO (si aplica) y/o pedir sentimientos
      if (runSentiment) {
        try {
          const job = await jobsApi.launchBetoPreproc(periodo);
          setBetoJobId(job.id);
        } catch {
          // Si el job no existe en backend, igual intentamos leer /datos/sentimientos
          await sentimientos.refetch();
        }
      }

      // 5) Refrescar resumen
      await resumen.refetch();
    } catch (e) {
      setErrorMsg((e as Error)?.message ?? "Processing failed.");
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-white mb-2">Data</h2>
        <p className="text-gray-400">Data Ingestion and Analysis</p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - Data Ingestion */}
        <div className="space-y-6">
          {/* Upload Section */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Dataset Upload</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Select File</label>

                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={(e) => onFileSelected(e.target.files?.[0] ?? null)}
                />

                <div
                  className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-gray-600 transition-colors"
                  onClick={openFilePicker}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const dropped = e.dataTransfer.files?.[0] ?? null;
                    onFileSelected(dropped);
                  }}
                >
                  <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-400">Click to upload or drag and drop</p>
                  <p className="text-gray-500 text-sm mt-1">CSV, XLSX (Max 10MB)</p>
                  {file && <p className="text-gray-300 text-sm mt-2">{file.name}</p>}
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Dataset Name</label>
                <Input
                  placeholder="e.g., Evaluations_2025_1"
                  className="bg-[#0f1419] border-gray-700"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Semester</label>
                <Select
                  value={activePeriodo}
                  onValueChange={(v) => setAppFilters({ activePeriodo: v })}
                >
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="2025-1">2025-1</SelectItem>
                    <SelectItem value="2024-2">2024-2</SelectItem>
                    <SelectItem value="2024-1">2024-1</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Program</label>
                <Select
                  value={programa}
                  onValueChange={(v) => setAppFilters({ programa: v })}
                >
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="Engineering">Engineering</SelectItem>
                    <SelectItem value="Business">Business</SelectItem>
                    <SelectItem value="Medicine">Medicine</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    checked={applyPreprocessing}
                    onCheckedChange={(checked) => setApplyPreprocessing(checked as boolean)}
                  />
                  <label className="text-sm text-gray-400">Apply preprocessing</label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    checked={runSentiment}
                    onCheckedChange={(checked) => setRunSentiment(checked as boolean)}
                  />
                  <label className="text-sm text-gray-400">Run sentiment analysis (BETO)</label>
                </div>
              </div>

              <Button
                className="w-full bg-blue-600 hover:bg-blue-700"
                onClick={handleProcess}
                disabled={isProcessing || upload.uploading}
              >
                {isProcessing || upload.uploading ? "Processing..." : "Load and Process"}
              </Button>

              {(isProcessing || upload.uploading) && (
                <div className="space-y-2">
                  <Progress value={upload.progress} className="w-full" />
                  <p className="text-sm text-gray-400 text-center">{upload.progress}%</p>
                </div>
              )}

              {errorMsg && <p className="text-sm text-red-400">{errorMsg}</p>}
              {validate.error && <p className="text-sm text-red-400">{validate.error}</p>}
              {upload.error && <p className="text-sm text-red-400">{upload.error}</p>}

              {dataLoaded && (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>
                    {rowsRead ?? kpiRows} rows read, {rowsValid ?? kpiRows} valid
                  </span>
                </div>
              )}

              {runSentiment && betoJob.job?.status === "running" && (
                <p className="text-sm text-gray-400">Running sentiment analysis (BETO)...</p>
              )}
              {runSentiment && betoJob.job?.status === "failed" && (
                <p className="text-sm text-red-400">BETO job failed: {betoJob.job?.error ?? "unknown"}</p>
              )}
            </div>
          </Card>
        </div>

        {/* Right Column - Data Preview */}
        <div className="col-span-2 space-y-6">
          {/* Dataset Summary */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Dataset Summary</h3>

            {dataLoaded ? (
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Total Rows</p>
                  <p className="text-white text-2xl mt-1">{Number(kpiRows).toLocaleString()}</p>
                </div>
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Columns</p>
                  <p className="text-white text-2xl mt-1">{Number(kpiCols).toLocaleString()}</p>
                </div>
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Teachers</p>
                  <p className="text-white text-2xl mt-1">{Number(kpiTeachers).toLocaleString()}</p>
                </div>
              </div>
            ) : (
              <div className="text-gray-400 text-sm">Upload a dataset to see summary.</div>
            )}

            {/* Data Table Preview */}
            {dataLoaded && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="text-left py-3 px-4">ID</th>
                      <th className="text-left py-3 px-4">Teacher</th>
                      <th className="text-left py-3 px-4">Subject</th>
                      <th className="text-left py-3 px-4">Rating</th>
                      <th className="text-left py-3 px-4">Comment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row) => (
                      <tr key={String(row.id)} className="text-white border-b border-gray-800/50">
                        <td className="py-3 px-4">{row.id}</td>
                        <td className="py-3 px-4">{row.teacher}</td>
                        <td className="py-3 px-4">{row.subject}</td>
                        <td className="py-3 px-4">{row.rating}</td>
                        <td className="py-3 px-4">{row.comment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Sentiment Analysis Section */}
      {dataLoaded && runSentiment && (
        <div>
          <h3 className="text-white mb-4">Sentiment Analysis with BETO</h3>
          <div className="grid grid-cols-3 gap-6">
            {/* Sentiment Distribution */}
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h4 className="text-white mb-4">Polarity Distribution</h4>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={sentimentDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }: any) => `${name}: ${percentage}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {sentimentDistribution.map((entry: any, index: number) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1a1f2e", border: "1px solid #374151" }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Sentiment by Teacher */}
            <Card className="bg-[#1a1f2e] border-gray-800 p-6 col-span-2">
              <h4 className="text-white mb-4">Sentiment Distribution by Teacher</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={sentimentByTeacher}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="teacher" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1a1f2e", border: "1px solid #374151" }}
                    labelStyle={{ color: "#fff" }}
                  />
                  <Legend />
                  <Bar dataKey="positive" stackId="a" fill={COLORS.positive} name="Positive" />
                  <Bar dataKey="neutral" stackId="a" fill={COLORS.neutral} name="Neutral" />
                  <Bar dataKey="negative" stackId="a" fill={COLORS.negative} name="Negative" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Errores no intrusivos (sin romper layout) */}
          {sentimientos.error && (
            <p className="text-sm text-gray-400 mt-3">
              Sentiments endpoint not available yet: {sentimientos.error}
            </p>
          )}
          {resumen.error && (
            <p className="text-sm text-gray-400 mt-1">
              Summary endpoint not available yet: {resumen.error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default DataTab;
