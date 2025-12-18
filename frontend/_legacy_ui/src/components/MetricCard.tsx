type Props = {
title: string;
value: string | number;
subtitle?: string;
};
export default function MetricCard({ title, value, subtitle }: Props) {
return (
<div className="card" style={{ display: "grid", gap: 6 }}>
<div style={{ color: "#64748b", fontSize: 12 }}>{title}</div>
<div style={{ fontSize: 22, fontWeight: 600 }}>{String(value)}</div>
{subtitle && <div className="badge">{subtitle}</div>}
</div>
);
}