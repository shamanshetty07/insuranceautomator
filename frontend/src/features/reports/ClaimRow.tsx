import { apiUrl } from "../../lib/api";
import { statusLabel } from "../../lib/format";
import type { InspectionImage, ReportClaim } from "../../types/api";
import { boxStyle } from "../detections/geometry";
import { VerdictBadge } from "./VerdictBadge";

interface Props {
  claim: ReportClaim;
  images: Map<string, InspectionImage>;
}

export function ClaimRow({ claim, images }: Props) {
  const result = claim.result;
  const evidence = [
    ...claim.evidence,
    ...claim.image_ids
      .filter((id) => !claim.evidence.some((e) => e.image_id === id))
      .map((id) => ({ image_id: id, prediction_id: null, crop: null, note: null })),
  ];

  return (
    <li className="space-y-2 rounded-lg border border-slate-200 bg-white p-4" aria-label={`Claim ${claim.claim_id}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <p className="font-medium text-slate-900">{claim.text}</p>
        <VerdictBadge result={result} />
      </div>
      <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        <div>
          <dt className="inline">Claim type: </dt>
          <dd className="inline">{statusLabel(claim.claim_type)}</dd>
        </div>
        <div>
          <dt className="inline">Damage type: </dt>
          <dd className="inline">{claim.damage_type ?? "—"}</dd>
        </div>
      </dl>

      {result?.needs_human_review && (
        <div className="space-y-1">
          <span className="inline-block rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-900">
            Needs human review
          </span>
          {result.policy_reason_codes.length > 0 && (
            <ul className="flex flex-wrap gap-1" aria-label={`Review reasons for claim ${claim.claim_id}`}>
              {result.policy_reason_codes.map((code) => (
                <li key={code}>
                  <code className="rounded bg-slate-100 px-1 text-xs">{code}</code>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {result && (
        <p className="text-xs text-slate-600">
          Verifier: {result.verification_label ? statusLabel(result.verification_label) : "no usable label"}
          {result.explanation ? ` — ${result.explanation}` : ""}
        </p>
      )}

      {evidence.length > 0 && (
        <ul className="flex flex-wrap gap-2" aria-label={`Evidence for claim ${claim.claim_id}`}>
          {evidence.map((ev, i) => {
            const image = images.get(ev.image_id);
            return (
              <li key={`${ev.image_id}-${ev.prediction_id ?? ""}-${i}`} className="w-32">
                {image ? (
                  <div className="relative overflow-hidden rounded bg-slate-100">
                    <img
                      src={apiUrl(image.content_url)}
                      alt={`Evidence for claim ${claim.claim_id}: Photo ${image.image_order + 1}`}
                      loading="lazy"
                      className="block h-auto w-full"
                    />
                    {ev.crop && (
                      <div
                        data-testid="evidence-crop"
                        aria-hidden="true"
                        className="pointer-events-none absolute border-2 border-sky-400"
                        style={boxStyle(ev.crop)}
                      />
                    )}
                  </div>
                ) : (
                  <p className="rounded border border-dashed border-slate-300 p-2 text-xs text-slate-500">
                    Image unavailable
                  </p>
                )}
                {ev.note && <p className="mt-1 text-xs text-slate-600">{ev.note}</p>}
              </li>
            );
          })}
        </ul>
      )}
    </li>
  );
}
