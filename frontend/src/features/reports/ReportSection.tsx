import { useRef, useState } from "react";
import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { ApiError, generateReport, getReport, getVlmStatus, listImages, listReports, verifyReport } from "../../lib/api";
import { formatDateTime, statusLabel } from "../../lib/format";
import { useAsync } from "../../lib/useAsync";
import type { InspectionImage, ReportDetail } from "../../types/api";
import { ClaimRow } from "./ClaimRow";

interface Props {
  inspectionId: string;
  refreshKey?: number;
}

export function actionErrorMessage(e: unknown, action: string): string {
  if (!(e instanceof ApiError)) return e instanceof Error ? e.message : `${action} failed.`;
  if (e.status === 504) return `${action} timed out: the model did not finish in time. Try again later.`;
  if (e.status === 503 && e.code === "vlm_busy") {
    return "The model is busy with another request. Try again shortly.";
  }
  if (e.status === 503) return `The report model is unavailable: ${e.message}`;
  return e.message;
}

async function loadReportView(inspectionId: string) {
  const [images, reports] = await Promise.all([listImages(inspectionId), listReports(inspectionId)]);
  const newest = reports.items[0];
  const report = newest ? await getReport(newest.id) : null;
  return { images: images.items, report };
}

export function ReportSection({ inspectionId, refreshKey = 0 }: Props) {
  const vlm = useAsync(getVlmStatus, []);
  const view = useAsync(() => loadReportView(inspectionId), [inspectionId, refreshKey]);
  const [busy, setBusy] = useState<null | "generate" | "verify">(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const inFlight = useRef(false);

  const available = vlm.state.status === "success" && vlm.state.data.available;
  const loaded = view.state.status === "success" ? view.state.data : null;
  const report: ReportDetail | null = loaded?.report ?? null;
  const images = new Map<string, InspectionImage>((loaded?.images ?? []).map((i) => [i.id, i]));
  const canGenerate = available && loaded !== null && loaded.images.length > 0 && busy === null;
  const canVerify = available && report !== null && report.claims.length > 0 && busy === null;

  async function act(kind: "generate" | "verify") {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(kind);
    setActionError(null);
    try {
      if (kind === "generate") await generateReport(inspectionId);
      else if (report) await verifyReport(report.id);
      view.reload();
    } catch (e) {
      setActionError(actionErrorMessage(e, kind === "generate" ? "Report generation" : "Verification"));
    } finally {
      inFlight.current = false;
      setBusy(null);
    }
  }

  const reviewCount = report?.claims.filter((c) => c.result?.needs_human_review).length ?? 0;

  return (
    <section className="space-y-4" aria-label="Damage report">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-lg font-semibold">Damage report</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => act("verify")}
            disabled={!canVerify}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
          >
            {busy === "verify" ? "Verifying claims…" : "Verify claims"}
          </button>
          <button
            type="button"
            onClick={() => act("generate")}
            disabled={!canGenerate}
            className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {busy === "generate" ? "Generating report…" : "Generate report"}
          </button>
        </div>
      </div>

      <p role="note" className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
        AI-generated predictions, not ground truth; claims flagged for review need a human.
      </p>

      {vlm.state.status === "error" && (
        <ErrorState message={`Could not check the report model: ${vlm.state.error.message}`} onRetry={vlm.reload} />
      )}
      {vlm.state.status === "success" &&
        (vlm.state.data.available ? (
          <p className="text-sm text-slate-600">
            Report model: {vlm.state.data.model_name} {vlm.state.data.model_version}
          </p>
        ) : (
          <p role="note" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            Report generation is unavailable{vlm.state.data.reason ? `: ${vlm.state.data.reason}` : "."}
          </p>
        ))}

      {busy && (
        <Spinner
          label={
            busy === "generate"
              ? "Generating report with the local model; this can take several minutes…"
              : "Verifying claims with the local model; this can take several minutes…"
          }
        />
      )}
      {actionError && <ErrorState message={actionError} />}

      {view.state.status === "loading" && <Spinner label="Loading report…" />}
      {view.state.status === "error" && <ErrorState message={view.state.error.message} onRetry={view.reload} />}
      {loaded &&
        (report === null ? (
          <p className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600">
            {loaded.images.length === 0 ? "Add images before generating a report." : "No report generated yet."}
          </p>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-slate-600">
              Generated {formatDateTime(report.created_at)} by {report.model_name} {report.model_version} ·{" "}
              {statusLabel(report.evidence_strategy)}
            </p>
            {report.verification ? (
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-700">
                <p>
                  Verification {statusLabel(report.verification.status)} · verifier {report.verification.verifier_model}{" "}
                  {report.verification.verifier_version}
                  {report.verification.policy_version ? ` · policy ${report.verification.policy_version}` : ""} ·{" "}
                  {reviewCount} claim(s) need human review
                </p>
                {report.verification.error_message && (
                  <p className="mt-1 text-red-700">{report.verification.error_message}</p>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-600">Claims have not been verified yet.</p>
            )}
            {report.claims.length === 0 ? (
              <p className="rounded-lg border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-600">
                The model described no visible damage; this is not a finding that the vehicle is undamaged.
              </p>
            ) : (
              <ul className="space-y-3" aria-label="Report claims">
                {report.claims.map((claim) => (
                  <ClaimRow key={claim.id} claim={claim} images={images} />
                ))}
              </ul>
            )}
          </div>
        ))}
    </section>
  );
}
