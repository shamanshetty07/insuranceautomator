import { statusLabel } from "../lib/format";
import type { InspectionStatus } from "../types/api";

const tone: Record<InspectionStatus, string> = {
  created: "bg-slate-100 text-slate-700",
  images_pending: "bg-amber-100 text-amber-800",
  ready_for_analysis: "bg-blue-100 text-blue-800",
  processing: "bg-blue-100 text-blue-800",
  review_required: "bg-amber-100 text-amber-800",
  completed: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-800",
};

export function StatusBadge({ status }: { status: InspectionStatus }) {
  return (
    <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${tone[status]}`}>
      {statusLabel(status)}
    </span>
  );
}
