import { Link } from "react-router-dom";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDateTime } from "../../lib/format";
import type { Inspection } from "../../types/api";

export function InspectionEmptyState({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white p-10 text-center">
      <p className="font-medium">No inspections yet</p>
      <p className="mt-1 text-sm text-slate-600">Create an inspection to start collecting vehicle photos.</p>
      <button
        type="button"
        onClick={onCreate}
        className="mt-4 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
      >
        Create inspection
      </button>
    </div>
  );
}

export function InspectionList({ items }: { items: Inspection[] }) {
  return (
    <ul className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
      {items.map((i) => (
        <li key={i.id}>
          <Link to={`/inspections/${i.id}`} className="flex items-center justify-between gap-4 p-4 hover:bg-slate-50">
            <div className="min-w-0">
              <p className="truncate font-medium">{i.title ?? "Untitled inspection"}</p>
              <p className="text-xs text-slate-500">Created {formatDateTime(i.created_at)}</p>
            </div>
            <StatusBadge status={i.status} />
          </Link>
        </li>
      ))}
    </ul>
  );
}
