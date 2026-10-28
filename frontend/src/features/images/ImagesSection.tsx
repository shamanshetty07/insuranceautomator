import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { listImages } from "../../lib/api";
import { IMAGE_LIMITS } from "../../lib/config";
import { useAsync } from "../../lib/useAsync";
import { ImageUploader } from "./ImageUploader";
import { StoredImageGrid } from "./StoredImageGrid";

interface Props {
  inspectionId: string;
  /** Called after an upload or delete changes the stored images. */
  onChange?: () => void;
}

export function ImagesSection({ inspectionId, onChange }: Props) {
  const { state, reload: reloadImages } = useAsync(() => listImages(inspectionId), [inspectionId]);
  const reload = () => {
    reloadImages();
    onChange?.();
  };
  const count = state.status === "success" ? state.data.total : 0;

  return (
    <section className="space-y-4" aria-label="Images">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">Images</h2>
        {state.status === "success" && (
          <span className="text-sm text-slate-500">
            {count} of {IMAGE_LIMITS.maxCount}
          </span>
        )}
      </div>

      {state.status === "success" && (
        <ImageUploader inspectionId={inspectionId} storedCount={count} onUploaded={reload} />
      )}
      {state.status === "loading" && <Spinner label="Loading images…" />}
      {state.status === "error" && <ErrorState message={state.error.message} onRetry={reload} />}
      {state.status === "success" &&
        (state.data.items.length === 0 ? (
          <p className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600">
            No images yet. Take a photo or choose photos from your device.
          </p>
        ) : (
          <StoredImageGrid images={state.data.items} onDeleted={reload} />
        ))}
    </section>
  );
}
