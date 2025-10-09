import React, { useMemo, useState } from "react";
import type { ValidacionResponse, IssueSeverity } from "../services/datos";

type Props = { data: ValidacionResponse | null };

export default function ValidationReport({ data }: Props) {
  const [severity, setSeverity] = useState<IssueSeverity | "all">("all");
  const [colFilter, setColFilter] = useState("");

  if (!data) return null;

  const { rows, errors, warnings, engine } = data.summary;

  const filtered = useMemo(() => {
    return data.issues.filter((it) => {
      const okSeverity = severity === "all" ? true : it.severity === severity;
      const okCol = colFilter.trim()
        ? (it.column || "").toLowerCase().includes(colFilter.toLowerCase())
        : true;
      return okSeverity && okCol;
    });
  }, [data.issues, severity, colFilter]);

  return (
    <div className="space-y-4">
      {/* KPIs resumidos */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="p-4 rounded-2xl shadow bg-white">
          <div className="text-sm text-gray-500">Filas</div>
          <div className="text-2xl font-semibold">{rows}</div>
        </div>
        <div className="p-4 rounded-2xl shadow bg-white">
          <div className="text-sm text-gray-500">Errores</div>
          <div className="text-2xl font-semibold text-red-600">{errors}</div>
        </div>
        <div className="p-4 rounded-2xl shadow bg-white">
          <div className="text-sm text-gray-500">Warnings</div>
          <div className="text-2xl font-semibold text-amber-600">{warnings}</div>
        </div>
        <div className="p-4 rounded-2xl shadow bg-white">
          <div className="text-sm text-gray-500">Engine</div>
          <div className="text-2xl font-semibold">{engine}</div>
        </div>
      </div>

      {/* Filtros */}
      <div className="flex gap-3 items-end">
        <div>
          <label className="block text-sm text-gray-600">Severidad</label>
          <select
            className="border rounded-lg p-2"
            value={severity}
            onChange={(e) => setSeverity(e.target.value as any)}
          >
            <option value="all">Todas</option>
            <option value="error">Errores</option>
            <option value="warning">Warnings</option>
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-sm text-gray-600">Filtrar por columna</label>
          <input
            className="border rounded-lg p-2 w-full"
            placeholder="p. ej. periodo, codigo_materia…"
            value={colFilter}
            onChange={(e) => setColFilter(e.target.value)}
          />
        </div>
      </div>

      {/* Tabla de issues */}
      <div className="overflow-auto rounded-2xl border">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left p-2">Severidad</th>
              <th className="text-left p-2">Código</th>
              <th className="text-left p-2">Columna</th>
              <th className="text-left p-2">Fila</th>
              <th className="text-left p-2">Mensaje</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((it, idx) => (
              <tr key={idx} className="border-t">
                <td className={`p-2 ${it.severity === "error" ? "text-red-600" : "text-amber-700"}`}>
                  {it.severity}
                </td>
                <td className="p-2">{it.code}</td>
                <td className="p-2">{it.column ?? "-"}</td>
                <td className="p-2">{it.row ?? "-"}</td>
                <td className="p-2">{it.message}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td className="p-3 text-gray-500" colSpan={5}>Sin resultados con los filtros actuales.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}