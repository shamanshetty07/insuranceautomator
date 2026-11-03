import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { StatusBadge } from "../../components/StatusBadge";
import { ApiError, getInspection, runInspection } from "../../lib/api";
import { formatDateTime } from "../../lib/format";
import { useAsync } from "../../lib/useAsync";
import { DetectionSection } from "../detections/DetectionSection";
import { ImagesSection } from "../images/ImagesSection";
import { ReportSection } from "../reports/ReportSection";

export function InspectionDetailPage() {
  const { inspectionId = "" } = useParams();
  const { state, reload } = useAsync(() => getInspection(inspectionId), [inspectionId]);
  const [imagesVersion, setImagesVersion] = useState(0);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  async function runCompleteInspection() {
    if (running) return;
    setRunning(true);
    setRunError(null);
    try {
      await runInspection(inspectionId);
      await reload();
      setImagesVersion((version) => version + 1);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Inspection processing failed.");
      await reload();
    } finally {
      setRunning(false);
    }
  }

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
          <section className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white p-4">
            <div>
              <h2 className="font-medium">Run complete inspection</h2>
              <p className="text-sm text-slate-600">
                Runs the configured local detector, generates a report, and verifies its claims. It can take several minutes.
              </p>
            </div>
            <button
              type="button"
              onClick={runCompleteInspection}
              disabled={running || state.data.status === "processing"}
              className="rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
            >
              {running || state.data.status === "processing" ? "Processing inspection…" : "Run inspection"}
            </button>
            {runError && <p className="w-full text-sm text-red-700" role="alert">{runError}</p>}
          </section>
          <ImagesSection inspectionId={state.data.id} onChange={() => setImagesVersion((v) => v + 1)} />
          <DetectionSection inspectionId={state.data.id} refreshKey={imagesVersion} />
          <ReportSection inspectionId={state.data.id} refreshKey={imagesVersion} />
        </>
      )}
    </div>
  );
}
