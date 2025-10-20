type Col = { key: string; header: string };


export default function ResultsTable({
    columns,
    rows,
}: {
    columns: Col[];
    rows: Array<Record<string, any>>;
}) {
    if (!rows?.length) return <div className="badge">Sin filas para mostrar</div>;
    return (
        <div className="card" style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", fontSize: 14 }}>
                <thead>
                    <tr style={{ textAlign: "left", color: "#64748b" }}>
                        {columns.map((c) => (
                            <th key={c.key} style={{ padding: "8px 6px" }}>{c.header}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((r, i) => (
                        <tr key={i} style={{ borderTop: "1px solid #e5e7eb" }}>
                            {columns.map((c) => (
                                <td key={c.key} style={{ padding: "8px 6px" }}>
                                    {String(r[c.key] ?? "")}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}