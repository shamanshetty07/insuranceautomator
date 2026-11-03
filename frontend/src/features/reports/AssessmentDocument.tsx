import type { ReactNode } from "react";
import { apiUrl } from "../../lib/api";
import { formatDateTime, statusLabel } from "../../lib/format";
import type {
  AssessmentClaimStatus,
  AssessmentFinding,
  AssessmentReport,
  InspectionImage,
} from "../../types/api";
import { boxStyle, formatConfidence } from "../detections/geometry";

interface Props {
  assessment: AssessmentReport;
  images: Map<string, InspectionImage>;
  claimsSection: ReactNode;
}

const STATUS_TONE: Record<AssessmentClaimStatus, string> = {
  supported: "bg-emerald-100 text-emerald-800",
  uncertain: "bg-amber-100 text-amber-900",
  unsupported: "bg-red-100 text-red-800",
};

const STATE_LABEL: Record<AssessmentReport["header"]["assessment_status"], string> = {
  no_findings: "No findings",
  awaiting_verification: "Awaiting verification",
  verification_failed: "Verification failed",
  pending_human_review: "Pending human review",
  ready: "Assessment ready",
};

const REVIEW_LABEL: Record<AssessmentReport["human_review"]["review_status"], string> = {
  not_started: "Not started",
  not_required: "Not required",
  pending: "Pending",
  complete: "Complete",
};

export function ClaimStatusBadge({ status }: { status: AssessmentClaimStatus }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_TONE[status]}`}>
      {statusLabel(status).replace(/^./, (c) => c.toUpperCase())}
    </span>
  );
}

export function ConfidenceBadge({ value }: { value: number }) {
  const tone =
    value >= 0.7 ? "bg-emerald-100 text-emerald-800" : value >= 0.5 ? "bg-amber-100 text-amber-900" : "bg-red-100 text-red-800";
  const level = value >= 0.7 ? "high" : value >= 0.5 ? "medium" : "low";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${tone}`} title="Detection confidence, not the probability the assessment is correct">
      {formatConfidence(value)} ({level})
    </span>
  );
}

function Section({ letter, title, children }: { letter: string; title: string; children: ReactNode }) {
  const id = `assessment-${letter.toLowerCase()}`;
  return (
    <section aria-labelledby={id} className="space-y-2">
      <h3 id={id} className="border-b border-slate-300 pb-1 text-base font-semibold text-slate-900">
        {letter}. {title}
      </h3>
      {children}
    </section>
  );
}

function EvidenceImage({ finding, image }: { finding: AssessmentFinding; image: InspectionImage | undefined }) {
  if (!image) {
    return <p className="rounded border border-dashed border-slate-300 p-2 text-xs text-slate-500">Image unavailable</p>;
  }
  const src = apiUrl(image.content_url);
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <figure>
        <img src={src} alt={`Original ${finding.image_label}`} loading="lazy" className="block h-auto w-full rounded" />
        <figcaption className="text-xs text-slate-500">Original · {finding.image_label}</figcaption>
      </figure>
      <figure>
        <div className="relative overflow-hidden rounded bg-slate-100">
          <img
            src={src}
            alt={`Annotated ${finding.image_label} for ${finding.damage_id}`}
            loading="lazy"
            className="block h-auto w-full"
          />
          {finding.bounding_box && (
            <div
              data-testid={`annotation-${finding.damage_id}`}
              className="pointer-events-none absolute border-2 border-red-500"
              style={boxStyle(finding.bounding_box)}
            >
              <span className="absolute left-0 top-0 bg-red-600 px-1 text-[10px] font-semibold text-white">
                {finding.damage_id}
              </span>
            </div>
          )}
        </div>
        <figcaption className="text-xs text-slate-500">Annotated · {finding.damage_id}</figcaption>
      </figure>
    </div>
  );
}

const coords = (b: AssessmentFinding["bounding_box"]) =>
  b ? `x ${b.x_min.toFixed(3)}–${b.x_max.toFixed(3)}, y ${b.y_min.toFixed(3)}–${b.y_max.toFixed(3)} (relative)` : "Not available";

export function AssessmentDocument({ assessment, images, claimsSection }: Props) {
  const { header, vehicle, findings, human_review: review } = assessment;
  const vehicleRows: [string, string][] = [
    ["Make", vehicle.make],
    ["Model", vehicle.model],
    ["Manufacturing year", vehicle.year],
    ["Registration number", vehicle.registration_number],
    ["Colour", vehicle.color],
  ];

  return (
    <article
      aria-label="Vehicle damage assessment report"
      className="space-y-6 rounded-lg border border-slate-300 bg-white p-4 shadow-sm sm:p-6"
    >
      <header className="space-y-1">
        <h2 className="text-xl font-bold text-slate-900">Vehicle Damage Assessment Report</h2>
        <p className="text-xs font-medium uppercase tracking-wide text-amber-800">
          AI-assisted preliminary assessment
        </p>
      </header>

      <Section letter="A" title="Report header">
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {[
            ["Report ID", header.report_id],
            ["Inspection ID", header.inspection_id],
            ["Generated", formatDateTime(header.generated_at)],
            ["Inspection type", header.inspection_type],
            ["Assessment status", STATE_LABEL[header.assessment_status]],
            ["Report model", header.report_model],
          ].map(([k, v]) => (
            <div key={k} className="min-w-0">
              <dt className="inline text-slate-500">{k}: </dt>
              <dd className="inline break-all">{v}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section letter="B" title="Vehicle information">
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {vehicleRows.map(([k, v]) => (
            <div key={k}>
              <dt className="inline text-slate-500">{k}: </dt>
              <dd className={v === "Not Provided" ? "inline italic text-slate-500" : "inline"}>{v}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section letter="C" title="Executive summary">
        <p className="text-sm text-slate-800">{assessment.executive_summary}</p>
      </Section>

      <Section letter="D" title="Damage detection summary">
        {findings.length === 0 ? (
          <p className="text-sm text-slate-600">No damage was detected. This is not a finding that the vehicle is undamaged.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-600">
                <tr>
                  {["ID", "Component", "Damage type", "Severity", "Confidence", "Image"].map((h) => (
                    <th key={h} scope="col" className="px-2 py-1.5 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {findings.map((f) => (
                  <tr key={f.prediction_id}>
                    <td className="px-2 py-1.5 font-medium">{f.damage_id}</td>
                    <td className="px-2 py-1.5 text-slate-600">{f.component}</td>
                    <td className="px-2 py-1.5">{f.damage_type}</td>
                    <td className="px-2 py-1.5 text-slate-600">{f.severity}</td>
                    <td className="px-2 py-1.5">
                      <ConfidenceBadge value={f.detection_confidence} />
                    </td>
                    <td className="px-2 py-1.5">{f.image_label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section letter="E" title="Detailed damage assessment">
        {findings.length === 0 ? (
          <p className="text-sm text-slate-600">No detected damage instances to assess.</p>
        ) : (
          <ul className="space-y-4">
            {findings.map((f) => (
              <li key={f.prediction_id} aria-label={`Damage ${f.damage_id}`} className="space-y-2 rounded border border-slate-200 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h4 className="font-semibold">
                    {f.damage_id} · {f.damage_type}
                  </h4>
                  <ConfidenceBadge value={f.detection_confidence} />
                </div>
                <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
                  <div>
                    <dt className="inline text-slate-500">Component: </dt>
                    <dd className="inline">{f.component}</dd>
                  </div>
                  <div>
                    <dt className="inline text-slate-500">Source image: </dt>
                    <dd className="inline">{f.image_label}</dd>
                  </div>
                  <div className="sm:col-span-2">
                    <dt className="inline text-slate-500">Bounding box: </dt>
                    <dd className="inline">{coords(f.bounding_box)}</dd>
                  </div>
                </dl>
                <div className="text-sm">
                  <p className="text-slate-500">AI visual observations (not confirmed facts)</p>
                  {f.observations.length === 0 ? (
                    <p className="text-slate-600">No AI-generated observation is linked to this detection.</p>
                  ) : (
                    <ul className="space-y-1">
                      {f.observations.map((o) => (
                        <li key={o.claim_id} className="flex flex-wrap items-center gap-2">
                          <span>{o.text}</span>
                          <ClaimStatusBadge status={o.status} />
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="text-sm">
                  <p className="text-slate-500">Possible repair considerations</p>
                  {f.repair_considerations.length === 0 ? (
                    <p className="text-slate-600">None supported by the evidence; physical inspection required.</p>
                  ) : (
                    <ul className="list-disc pl-5">
                      {f.repair_considerations.map((r) => (
                        <li key={r}>{r}</li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="text-sm">
                  <p className="text-slate-500">Additional inspection requirements</p>
                  <ul className="list-disc pl-5">
                    {f.additional_inspection.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section letter="F" title="Visual evidence">
        {findings.length === 0 ? (
          <p className="text-sm text-slate-600">No annotated evidence: nothing was detected.</p>
        ) : (
          <ul className="space-y-4">
            {findings.map((f) => (
              <li key={f.prediction_id} aria-label={`Evidence for ${f.damage_id}`} className="space-y-1">
                <p className="text-sm font-medium">
                  {f.damage_id} · {f.damage_type} · {f.image_label}
                </p>
                <EvidenceImage finding={f} image={f.image_id ? images.get(f.image_id) : undefined} />
                <p className="text-xs text-slate-500">Coordinates: {coords(f.bounding_box)}</p>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-slate-500">{assessment.evidence_notice}</p>
      </Section>

      <Section letter="G" title="AI claim verification">{claimsSection}</Section>

      <Section letter="H" title="Repair recommendations">
        <ul className="list-disc pl-5 text-sm">
          {assessment.recommendations.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <p className="text-sm text-slate-600">{assessment.cost_estimate}</p>
      </Section>

      <Section letter="I" title="Human review">
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          <div>
            <dt className="inline text-slate-500">AI assessment status: </dt>
            <dd className="inline">{review.ai_assessment_status}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">Verification status: </dt>
            <dd className="inline">{statusLabel(review.verification_status)}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">Human review status: </dt>
            <dd className="inline font-medium">{REVIEW_LABEL[review.review_status]}</dd>
          </div>
        </dl>
        {review.items.length > 0 && (
          <ul className="space-y-1 text-sm" aria-label="Reviewer decisions">
            {review.items.map((item) => (
              <li key={item.claim_id}>
                Claim {item.claim_id}:{" "}
                {item.outcome ? (
                  <>
                    <strong>{statusLabel(item.outcome)}</strong>
                    {item.reviewer_note ? ` — ${item.reviewer_note}` : " — no reviewer comment"}
                    {item.reviewed_at ? ` (${formatDateTime(item.reviewed_at)})` : ""}
                  </>
                ) : (
                  <span className="text-amber-800">awaiting reviewer decision</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section letter="J" title="Disclaimer">
        <p className="text-xs text-slate-600">{assessment.disclaimer}</p>
      </Section>
    </article>
  );
}
