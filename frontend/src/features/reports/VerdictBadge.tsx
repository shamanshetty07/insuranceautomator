import type { ClaimResult } from "../../types/api";

export function VerdictBadge({ result }: { result: ClaimResult | null }) {
  const [label, tone] = !result
    ? ["Not verified", "bg-slate-100 text-slate-700"]
    : result.verdict === "supported"
      ? ["Supported", "bg-emerald-100 text-emerald-800"]
      : result.verdict === "contradicted"
        ? ["Contradicted", "bg-red-100 text-red-800"]
        : result.verdict === "insufficient_evidence"
          ? ["Insufficient evidence", "bg-amber-100 text-amber-900"]
          : ["No verdict", "bg-slate-100 text-slate-700"];
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${tone}`}>{label}</span>;
}
