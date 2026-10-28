import { Link, useParams } from "react-router-dom";
import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { StatusBadge } from "../../components/StatusBadge";
import { ApiError, getInspection } from "../../lib/api";
import { formatDateTime } from "../../lib/format";
import { useAsync } from "../../lib/useAsync";
import { ImagesSection } from "../images/ImagesSection";

export function InspectionDetailPage() {
  const { inspectionId = "" } = useParams();
  const { state, reload } = useAsync(() => getInspection(inspectionId), [inspectionId]);

  return (
    <div className="space-y-6">
      <Link to="/inspections" className="text-sm text-slate-600 hover:underline">
        ← All inspections
      </Link>

      {state.status === "loading" && <Spinner label="Loading inspection…" />}
      {state.status === "error" &&
        (state.error instanceof ApiError && state.error.status === 404 ? (
          <ErrorState message="This inspection does not exist." />
        ) : (
          <ErrorState message={state.error.message} onRetry={reload} />
        ))}
      {state.status === "success" && (
        <>
          <div className="flex items-start justify-between gap-4">
            <h1 className="text-2xl font-semibold">{state.data.title ?? "Untitled inspection"}</h1>
            <StatusBadge status={state.data.status} />
          </div>
          <dl className="grid gap-4 rounded-lg border border-slate-200 bg-white p-5 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-slate-500">Created</dt>
              <dd>{formatDateTime(state.data.created_at)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Last updated</dt>
              <dd>{formatDateTime(state.data.updated_at)}</dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-slate-500">Notes</dt>
              <dd className="whitespace-pre-wrap">{state.data.notes ?? "No notes."}</dd>
            </div>
          </dl>
          <ImagesSection inspectionId={state.data.id} />
          <p className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-600">
            Damage detection, reports and verification will appear here in later phases.
          </p>
        </>
      )}
    </div>
  );
}
