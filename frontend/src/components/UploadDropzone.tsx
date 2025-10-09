// frontend/src/components/UploadDropzone.tsx
/**
 * UploadDropzone — Componente funcional (Día 2 + ajustes Día 3).
 * - Arrastrar/soltar y selector nativo.
 * - Validación mínima: extensión y tamaño.
 * - Accesible (teclado y ARIA).
 *
 * Uso:
 *   <UploadDropzone onFileSelected={(f) => setFile(f)} />
 */

import React, { useCallback, useRef, useState } from "react";

type Props = {
  onFileSelected: (file: File) => void;
  /** Extensiones permitidas, separadas por coma (coinciden con atributo accept del input) */
  accept?: string; // ".csv,.xlsx,.parquet"
  /** Tamaño máximo en MB (por defecto 10MB) */
  maxSizeMB?: number;
  /** Desactivar interacción (ej. mientras se envía el formulario) */
  disabled?: boolean;
  /** Texto opcional a mostrar */
  label?: string;
};

function extFromName(name: string) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

export default function UploadDropzone({
  onFileSelected,
  accept = ".csv,.xlsx,.parquet",  // ← 🆕 incluye parquet por defecto
  maxSizeMB = 10,
  disabled = false,
  label = "Arrastra tu archivo CSV/XLSX aquí o selecciónalo:",
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isOver, setIsOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");

  const allowedExts = accept
    .split(",")
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean);

  const validate = useCallback(
    (f: File): string | null => {
      // Extensión
      const ext = extFromName(f.name);
      if (allowedExts.length > 0 && !allowedExts.includes(ext)) {
        return `Formato no permitido (${ext || "sin extensión"}). Usa: ${allowedExts.join(", ")}`;
      }
      // Tamaño
      const maxBytes = maxSizeMB * 1024 * 1024;
      if (f.size > maxBytes) {
        return `El archivo supera ${maxSizeMB}MB (${(f.size / (1024 * 1024)).toFixed(2)}MB).`;
      }
      return null;
    },
    [allowedExts, maxSizeMB]
  );

  const handleFile = useCallback(
    (f: File) => {
      const v = validate(f);
      if (v) {
        setError(v);
        setFileName("");
        return;
      }
      setError(null);
      setFileName(f.name);
      onFileSelected(f);
    },
    [onFileSelected, validate]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOver(false);
    if (disabled) return;
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!disabled) setIsOver(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsOver(false);
  };

  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !disabled && inputRef.current?.click()}
        className={[
          "border-2 border-dashed rounded-xl p-6 text-center select-none transition-shadow",
          disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
          isOver ? "shadow-lg" : "hover:shadow",
        ].join(" ")}
      >
        <p className="mb-3">{label}</p>
        <div className="inline-flex items-center gap-2">
          <button
            type="button"
            className="px-4 py-2 rounded-xl shadow"
            disabled={disabled}
            // ← 🆕 Click explícito al input; evita depender solo del contenedor
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
          >
            Seleccionar archivo
          </button>
          <span className="text-sm opacity-70">({allowedExts.join(", ")}, máx. {maxSizeMB}MB)</span>
        </div>
        <input
            ref={inputRef}
            type="file"
            accept={accept}
            multiple={false}
            className="hidden"
            style={{ display: "none" }}  // ← fuerza que no se vea aunque falle Tailwind
            onChange={onInputChange}
            disabled={disabled}
        />
      </div>

      {/* Estado / errores */}
      {fileName && !error && (
        <div className="text-sm">
          <strong>Seleccionado:</strong> {fileName}
        </div>
      )}
      {error && (
        <div className="p-2 rounded-lg bg-red-100 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
