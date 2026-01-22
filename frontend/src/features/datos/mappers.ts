// frontend/src/features/datos/mappers.ts
import type { DatasetSentimientos, ValidarResp } from "@/types/neurocampus";

export type UiPreviewRow = {
  id: number | string;
  teacher: string;
  subject: string;
  rating: number | string;
  comment: string;
};

function pick(obj: any, keys: string[]): any {
  for (const k of keys) {
    if (obj && obj[k] !== undefined && obj[k] !== null) return obj[k];
  }
  return undefined;
}

function asNumber(v: any): number | string {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const n = Number(v.replace(",", "."));
    return Number.isFinite(n) ? n : v;
  }
  return "";
}

function normLabel(raw: unknown): "pos" | "neu" | "neg" | null {
  if (raw == null) return null;
  const s = String(raw).trim().toLowerCase();
  if (s === "pos" || s === "positive" || s === "positivo") return "pos";
  if (s === "neu" || s === "neutral" || s === "neutro") return "neu";
  if (s === "neg" || s === "negative" || s === "negativo") return "neg";
  return null;
}

export function mapSampleRowsToPreview(sample?: Array<Record<string, any>>): UiPreviewRow[] {
  if (!Array.isArray(sample) || sample.length === 0) return [];

  return sample.slice(0, 8).map((row, idx) => {
    const teacher =
  pick(row, [
    "teacher", "Teacher",
    "docente", "Docente", "DOCENTE",
    "profesor", "Profesor", "PROFESOR",
    "nombre_docente", "NOMBRE_DOCENTE",
    "nombreProfesor", "NOMBRE_PROFESOR",
  ]) ?? "—";

const subject =
  pick(row, [
    "subject", "Subject",
    "asignatura", "Asignatura", "ASIGNATURA",
    "materia", "Materia", "MATERIA",
    "nombre_asignatura", "NOMBRE_ASIGNATURA",
    "curso", "CURSO",
  ]) ?? "—";

const rating =
  asNumber(
    pick(row, [
      "rating", "Rating",
      "calificacion", "Calificacion", "CALIFICACION",
      "nota", "NOTA",
      "score", "SCORE",
      "promedio", "PROMEDIO",
      "calificacion_final", "CALIFICACION_FINAL",
      "calif_promedio", "CALIF_PROMEDIO",
      "calif_1", "CALIF_1",
      "pregunta_1", "PREGUNTA_1",
    ]),
  ) || "—";

const comment =
  pick(row, [
    "comment", "Comment",
    "comentario", "Comentario", "COMENTARIO",
    "observaciones", "OBSERVACIONES",
    "observacion", "Observacion", "OBSERVACION",
    "feedback", "FEEDBACK",
    "opinion", "OPINION",
    "texto", "TEXTO",
  ]) ?? "—";
    const id = pick(row, ["id", "ID", "codigo", "codigo_docente", "codigo_estudiante"]) ?? (idx + 1);

    return { id, teacher: String(teacher), subject: String(subject), rating, comment: String(comment) };
  });
}

const labelToName: Record<string, string> = {
  pos: "Positive",
  neu: "Neutral",
  neg: "Negative",
};

export function mapGlobalSentiment(ds: DatasetSentimientos | null | undefined) {
  if (!ds) return [];
  return (ds.global_counts ?? []).map((x) => ({
    name: labelToName[x.label] ?? x.label,
    value: x.count,
    percentage: Math.round((x.proportion ?? 0) * 100),
  }));
}

export function mapTeacherSentiment(api: any): Array<{
  teacher: string;
  pos: number;
  neu: number;
  neg: number;
  total: number;
}> {
  const rows = Array.isArray(api?.por_docente) ? api.por_docente : [];

  return rows
    .map((r: any) => {
      const teacher = String(r?.docente ?? r?.teacher ?? r?.profesor ?? r?.group ?? "").trim();
      if (!teacher) return null;

      // Caso A: backend ya manda campos pos/neu/neg
      const posA = r?.pos;
      const neuA = r?.neu;
      const negA = r?.neg;

      if (
        typeof posA === "number" &&
        typeof neuA === "number" &&
        typeof negA === "number"
      ) {
        const total = Number(r?.total ?? (posA + neuA + negA));
        return { teacher, pos: posA, neu: neuA, neg: negA, total };
      }

      // Caso B: backend manda "counts" como lista [{label,count,proportion}]
      const counts = Array.isArray(r?.counts) ? r.counts : [];

      let pos = 0;
      let neu = 0;
      let neg = 0;

      for (const c of counts) {
        const label = normLabel(c?.label);
        const count = Number(c?.count ?? 0);
        if (!Number.isFinite(count)) continue;

        if (label === "pos") pos += count;
        if (label === "neu") neu += count;
        if (label === "neg") neg += count;
      }

      const total = Number(r?.total ?? (pos + neu + neg));
      return { teacher, pos, neu, neg, total };
    })
    .filter(Boolean) as Array<{
      teacher: string;
      pos: number;
      neu: number;
      neg: number;
      total: number;
    }>;
}

export function rowsReadValidFromValidation(v: ValidarResp | null | undefined) {
  if (!v) return { rowsRead: null as number | null, rowsValid: null as number | null };
  const rowsRead = typeof v.n_rows === "number" ? v.n_rows : null;

  const hasError =
    Array.isArray(v.issues) && v.issues.some((i) => i.level === "error");

  const rowsValid = hasError ? null : rowsRead;
  return { rowsRead, rowsValid };
}
