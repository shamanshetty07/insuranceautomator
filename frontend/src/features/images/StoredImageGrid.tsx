import { useState } from "react";
import { apiUrl, deleteImage } from "../../lib/api";
import type { InspectionImage } from "../../types/api";
import { formatBytes } from "./validation";

interface Props {
  images: InspectionImage[];
  onDeleted: () => void;
}

function StoredImage({ image, onDeleted }: { image: InspectionImage; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasSize = image.width_px && image.height_px;

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await deleteImage(image.inspection_id, image.id);
      onDeleted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete the image.");
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="space-y-2 rounded-lg border border-slate-200 bg-white p-3">
      <div
        className="relative overflow-hidden rounded bg-slate-100"
        style={hasSize ? { aspectRatio: `${image.width_px} / ${image.height_px}` } : undefined}
      >
        {/* Already EXIF-normalized: never rotate with CSS. */}
        <img
          src={apiUrl(image.content_url)}
          alt={`Photo ${image.image_order + 1}: ${image.original_filename}`}
          width={image.width_px ?? undefined}
          height={image.height_px ?? undefined}
          loading="lazy"
          className="h-full w-full object-contain"
        />
        <span className="absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-xs font-medium text-white">
          #{image.image_order + 1}
        </span>
      </div>
      <div className="text-xs text-slate-600">
        <p className="truncate font-medium text-slate-900">{image.original_filename}</p>
        <p>
          {hasSize ? `${image.width_px} × ${image.height_px} px · ` : ""}
          {formatBytes(image.file_size_bytes)}
        </p>
      </div>
      {error && (
        <p role="alert" className="text-xs text-red-700">
          {error}
        </p>
      )}
      {confirming ? (
        <div className="flex gap-3 text-xs">
          <button type="button" disabled={busy} onClick={remove} className="font-medium text-red-700 underline">
            {busy ? "Deleting…" : "Confirm delete"}
          </button>
          <button type="button" disabled={busy} onClick={() => setConfirming(false)} className="underline">
            Cancel
          </button>
        </div>
      ) : (
        <button type="button" onClick={() => setConfirming(true)} className="text-xs text-slate-600 underline">
          Delete
        </button>
      )}
    </li>
  );
}

export function StoredImageGrid({ images, onDeleted }: Props) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-label="Uploaded images">
      {images.map((image) => (
        <StoredImage key={image.id} image={image} onDeleted={onDeleted} />
      ))}
    </ul>
  );
}
