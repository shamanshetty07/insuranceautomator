import { apiUrl } from "../../lib/api";
import type { DetectionRun, InspectionImage } from "../../types/api";
import { boxStyle, formatConfidence, formatDuration } from "./geometry";

interface Props {
  image: InspectionImage;
  run: DetectionRun | undefined;
}

export function DetectionResultCard({ image, run }: Props) {
  const label = `Photo ${image.image_order + 1}`;
  const boxed = run?.status === "completed" ? run.predictions.filter((p) => p.bounding_box) : [];
  const duration = run ? formatDuration(run.duration_ms) : null;

  return (
    <li className="space-y-2 rounded-lg border border-slate-200 bg-white p-3" aria-label={`Detection results for ${label}`}>
      {/* The wrapper hugs the image, so percentage offsets map onto the stored pixels. */}
      <div className="relative overflow-hidden rounded bg-slate-100">
        <img
          src={apiUrl(image.content_url)}
          alt={`${label}: ${image.original_filename}`}
          width={image.width_px ?? undefined}
          height={image.height_px ?? undefined}
          loading="lazy"
          className="block h-auto w-full"
        />
        {boxed.map((p, i) => (
          <div
            key={p.id}
            data-testid="detection-box"
            aria-hidden="true"
            className="pointer-events-none absolute border-2 border-amber-400"
            style={boxStyle(p.bounding_box!)}
          >
            <span className="absolute left-0 top-0 bg-amber-400 px-1 text-[10px] font-semibold text-slate-900">
              {i + 1}
            </span>
          </div>
        ))}
      </div>

      <div className="space-y-1 text-xs text-slate-600">
        <p className="truncate font-medium text-slate-900">
          {label} · {image.original_filename}
        </p>
        {!run && <p>Not analyzed yet.</p>}
        {run && (
          <p>
            <span className="font-medium">Status:</span> {run.status}
            {duration && ` · ${duration}`}
          </p>
        )}
        {run && (
          <p>
            <span className="font-medium">Model:</span> {run.model_name} {run.model_version}
          </p>
        )}
        {run?.status === "failed" && (
          <p role="alert" className="text-red-700">
            Detection failed: {run.error_message ?? "unknown error"}
          </p>
        )}
        {run?.status === "completed" &&
          (run.predictions.length === 0 ? (
            <p>No damage detected by this model.</p>
          ) : (
            <ol className="space-y-0.5" aria-label={`Detections for ${label}`}>
              {run.predictions.map((p) => (
                <li key={p.id} className="flex justify-between gap-2">
                  <span>
                    {boxed.includes(p) && <span className="mr-1 font-semibold">{boxed.indexOf(p) + 1}.</span>}
                    {p.damage_type}
                  </span>
                  <span className="tabular-nums">{formatConfidence(p.confidence)}</span>
                </li>
              ))}
            </ol>
          ))}
      </div>
    </li>
  );
}
