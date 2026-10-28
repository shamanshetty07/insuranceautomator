import { useRef, useState } from "react";
import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { getDetectorStatus, listDetections, listImages, runDetection } from "../../lib/api";
import { useAsync } from "../../lib/useAsync";
import type { DetectionRun } from "../../types/api";
import { DetectionResultCard } from "./DetectionResultCard";

interface Props {
  inspectionId: string;
  /** Changes whenever images are uploaded or deleted, so results are refetched. */
  refreshKey?: number;
}

export function DetectionSection({ inspectionId, refreshKey = 0 }: Props) {
  const detector = useAsync(getDetectorStatus, []);
  const results = useAsync(
    () => Promise.all([listImages(inspectionId), listDetections(inspectionId, true)]),
    [inspectionId, refreshKey],
  );
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  // Guards double submits before React re-renders the disabled button.
  const inFlight = useRef(false);

  const images = results.state.status === "success" ? results.state.data[0].items : [];
  const runsByImage = new Map<string, DetectionRun>();
  if (results.state.status === "success") {
    for (const run of results.state.data[1].items) runsByImage.set(run.inspection_image_id, run);
  }
  const available = detector.state.status === "success" && detector.state.data.available;
  const canRun = available && results.state.status === "success" && images.length > 0 && !running;

  async function run() {
    if (inFlight.current) return;
    inFlight.current = true;
    setRunning(true);
    setRunError(null);
    try {
      await runDetection(inspectionId);
      results.reload();
    } catch (e) {
      setRunError(e instanceof Error ? e.message : "Detection failed.");
    } finally {
      inFlight.current = false;
      setRunning(false);
    }
  }

  return (
    <section className="space-y-4" aria-label="Damage detection">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-lg font-semibold">Damage detection</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={results.reload}
            disabled={running}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
          >
            Refresh results
          </button>
          <button
            type="button"
            onClick={run}
            disabled={!canRun}
            className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {running ? "Running detection…" : "Run detection"}
          </button>
        </div>
      </div>

      {detector.state.status === "loading" && <p className="text-sm text-slate-600">Checking detector…</p>}
      {detector.state.status === "error" && (
        <ErrorState message={`Could not check the detector: ${detector.state.error.message}`} onRetry={detector.reload} />
      )}
      {detector.state.status === "success" &&
        (detector.state.data.available ? (
          <p className="text-sm text-slate-600">
            Model: {detector.state.data.model_name} {detector.state.data.model_version}
          </p>
        ) : (
          <p role="note" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            Damage detection is unavailable
            {detector.state.data.reason ? `: ${detector.state.data.reason}` : "."}
          </p>
        ))}

      {running && <Spinner label="Running detection on all images…" />}
      {runError && <ErrorState message={runError} />}

      {results.state.status === "loading" && <Spinner label="Loading detection results…" />}
      {results.state.status === "error" && (
        <ErrorState message={results.state.error.message} onRetry={results.reload} />
      )}
      {results.state.status === "success" &&
        (images.length === 0 ? (
          <p className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600">
            Add images before running damage detection.
          </p>
        ) : (
          <>
            {runsByImage.size === 0 && <p className="text-sm text-slate-600">No detection runs yet.</p>}
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-label="Detection results">
              {images.map((image) => (
                <DetectionResultCard key={image.id} image={image} run={runsByImage.get(image.id)} />
              ))}
            </ul>
          </>
        ))}
    </section>
  );
}
