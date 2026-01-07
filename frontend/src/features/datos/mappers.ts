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
    ]),
  ) || "—";

const comment =
  pick(row, [
    "comment", "Comment",
    "comentario", "Comentario", "COMENTARIO",
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

export function mapTeacherSentiment(ds: DatasetSentimientos | null | undefined) {
  if (!ds) return [];
  const byTeacher = ds.por_docente ?? [];

  return byTeacher.slice(0, 10).map((t) => {
    const counts = new Map((t.counts ?? []).map((c) => [c.label, c.count]));
    return {
      teacher: t.group,
      positive: counts.get("pos") ?? 0,
      neutral: counts.get("neu") ?? 0,
      negative: counts.get("neg") ?? 0,
    };
  });
}

export function rowsReadValidFromValidation(v: ValidarResp | null | undefined) {
  if (!v) return { rowsRead: null as number | null, rowsValid: null as number | null };
  const rowsRead = typeof v.n_rows === "number" ? v.n_rows : null;

  const hasError =
    Array.isArray(v.issues) && v.issues.some((i) => i.level === "error");

  const rowsValid = hasError ? null : rowsRead;
  return { rowsRead, rowsValid };
}
