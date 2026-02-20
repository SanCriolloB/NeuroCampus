// ============================================================
// NeuroCampus — Shared Badges & Small Components
// ============================================================
import { Badge } from '../ui/badge';
import {
  CheckCircle2, AlertTriangle, XCircle, Flame, FileText,
  Copy, ExternalLink
} from 'lucide-react';
import type { BundleStatus, WarmStartFrom, RunStatus } from './mockData';

// ---------- Warm Start Badge ----------
export function WarmStartBadge({
  warmed,
  from,
  result,
}: {
  warmed: boolean;
  from: WarmStartFrom;
  result: 'ok' | 'skipped' | 'error' | null;
}) {
  if (!warmed) {
    return (
      <Badge className="bg-gray-700/60 text-gray-400 border-none gap-1 text-xs">
        No WS
      </Badge>
    );
  }
  const color =
    result === 'ok'
      ? 'bg-green-500/20 text-green-400 border-green-500/40'
      : result === 'error'
      ? 'bg-red-500/20 text-red-400 border-red-500/40'
      : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40';

  return (
    <Badge className={`${color} gap-1 text-xs`}>
      <Flame className="w-3 h-3" />
      WS:{from === 'champion' ? 'Champ' : 'Run'}
      {result && <span className="opacity-70">({result})</span>}
    </Badge>
  );
}

// ---------- Text Features Badge ----------
export function TextFeaturesBadge({ count }: { count: number }) {
  if (count === 0) {
    return (
      <Badge className="bg-gray-700/60 text-gray-400 border-none gap-1 text-xs">
        <FileText className="w-3 h-3" />0 txt
      </Badge>
    );
  }
  return (
    <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/40 gap-1 text-xs">
      <FileText className="w-3 h-3" />{count} txt
    </Badge>
  );
}

// ---------- Bundle Status Badge ----------
export function BundleStatusBadge({ status }: { status: BundleStatus }) {
  if (status === 'complete') {
    return (
      <Badge className="bg-green-500/20 text-green-400 border-green-500/40 gap-1 text-xs">
        <CheckCircle2 className="w-3 h-3" /> Completo
      </Badge>
    );
  }
  return (
    <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/40 gap-1 text-xs">
      <AlertTriangle className="w-3 h-3" /> Incompleto
    </Badge>
  );
}

// ---------- Run Status Badge ----------
export function RunStatusBadge({ status }: { status: RunStatus }) {
  const map: Record<RunStatus, { color: string; icon: React.ReactNode }> = {
    completed: { color: 'bg-green-500/20 text-green-400 border-green-500/40', icon: <CheckCircle2 className="w-3 h-3" /> },
    running: { color: 'bg-blue-500/20 text-blue-400 border-blue-500/40', icon: <span className="w-3 h-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin inline-block" /> },
    queued: { color: 'bg-gray-500/20 text-gray-400 border-gray-500/40', icon: null },
    failed: { color: 'bg-red-500/20 text-red-400 border-red-500/40', icon: <XCircle className="w-3 h-3" /> },
  };
  const { color, icon } = map[status];
  return (
    <Badge className={`${color} gap-1 text-xs capitalize`}>
      {icon}{status}
    </Badge>
  );
}

// ---------- Diagnostic Status Icon ----------
export function DiagnosticIcon({ status }: { status: 'pass' | 'warn' | 'fail' }) {
  if (status === 'pass') return <CheckCircle2 className="w-4 h-4 text-green-400" />;
  if (status === 'warn') return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
  return <XCircle className="w-4 h-4 text-red-400" />;
}

// ---------- Copy Button ----------
export function CopyButton({ text }: { text: string }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(text).catch(() => {});
  };
  return (
    <button
      onClick={handleCopy}
      className="p-1 hover:bg-gray-700 rounded transition-colors"
      title="Copiar"
    >
      <Copy className="w-3.5 h-3.5 text-gray-400" />
    </button>
  );
}

// ---------- Metric Chip ----------
export function MetricChip({ label, value, mode }: { label: string; value: string; mode?: string }) {
  return (
    <div className="bg-[#0f1419] border border-gray-700 rounded-lg px-3 py-1.5 text-xs flex items-center gap-2">
      <span className="text-gray-400">{label}:</span>
      <span className="text-cyan-400">{value}</span>
      {mode && <span className="text-gray-500">({mode})</span>}
    </div>
  );
}
