import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { Button } from "../../components/Button";
import { uploadImage } from "../../lib/api";
import { IMAGE_LIMITS } from "../../lib/config";
import { CameraCapture } from "./CameraCapture";
import { formatBytes, validateImageFile } from "./validation";

interface Staged {
  key: string;
  file: File;
  previewUrl: string;
  status: "ready" | "uploading" | "error";
  progress: number;
  error?: string;
}

interface Props {
  inspectionId: string;
  storedCount: number;
  onUploaded: () => void;
}

const fingerprint = (f: File) => `${f.name}:${f.size}:${f.lastModified}`;

export function ImageUploader({ inspectionId, storedCount, onUploaded }: Props) {
  const [staged, setStaged] = useState<Staged[]>([]);
  const [messages, setMessages] = useState<string[]>([]);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inFlight = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const stagedRef = useRef(staged);
  stagedRef.current = staged;

  useEffect(
    () => () => stagedRef.current.forEach((s) => URL.revokeObjectURL(s.previewUrl)),
    [],
  );

  const remaining = IMAGE_LIMITS.maxCount - storedCount - staged.length;

  function addFiles(files: File[]) {
    const problems: string[] = [];
    const accepted: Staged[] = [];
    const known = new Set(stagedRef.current.map((s) => fingerprint(s.file)));
    let room = remaining;
    for (const file of files) {
      const problem = validateImageFile(file);
      if (problem) problems.push(problem);
      else if (known.has(fingerprint(file))) problems.push(`${file.name}: already selected.`);
      else if (room <= 0) problems.push(`${file.name}: an inspection can have at most ${IMAGE_LIMITS.maxCount} images.`);
      else {
        known.add(fingerprint(file));
        room -= 1;
        accepted.push({
          key: crypto.randomUUID(),
          file,
          previewUrl: URL.createObjectURL(file),
          status: "ready",
          progress: 0,
        });
      }
    }
    setMessages(problems);
    if (accepted.length) setStaged((current) => [...current, ...accepted]);
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(e.target.files ?? []));
    e.target.value = "";
  }

  function remove(key: string) {
    setStaged((current) => {
      const target = current.find((s) => s.key === key);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return current.filter((s) => s.key !== key);
    });
  }

  const patch = (key: string, change: Partial<Staged>) =>
    setStaged((current) => current.map((s) => (s.key === key ? { ...s, ...change } : s)));

  async function uploadAll() {
    if (inFlight.current) return;
    inFlight.current = true;
    setUploading(true);
    setMessages([]);
    let anyUploaded = false;
    for (const item of stagedRef.current.filter((s) => s.status !== "uploading")) {
      patch(item.key, { status: "uploading", progress: 0, error: undefined });
      try {
        await uploadImage(inspectionId, item.file, (p) => patch(item.key, { progress: p }));
        anyUploaded = true;
        remove(item.key);
      } catch (e) {
        patch(item.key, { status: "error", error: e instanceof Error ? e.message : "Upload failed." });
      }
    }
    inFlight.current = false;
    setUploading(false);
    if (anyUploaded) onUploaded();
  }

  const full = remaining <= 0;

  return (
    <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-5" aria-label="Add photos">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={IMAGE_LIMITS.acceptedTypes.join(",")}
          onChange={onPick}
          className="sr-only"
          aria-label="Choose photos"
          data-testid="file-input"
        />
        <Button variant="secondary" disabled={uploading || full} onClick={() => fileInput.current?.click()}>
          Choose photos
        </Button>
        <Button variant="secondary" disabled={uploading || full || cameraOpen} onClick={() => setCameraOpen(true)}>
          Use camera
        </Button>
        <p className="text-xs text-slate-500">
          JPEG, PNG or WebP · up to {formatBytes(IMAGE_LIMITS.maxBytes)} each · {Math.max(remaining, 0)} more allowed
        </p>
      </div>

      {cameraOpen && (
        <CameraCapture onCapture={(file) => addFiles([file])} onClose={() => setCameraOpen(false)} />
      )}

      {messages.length > 0 && (
        <ul role="alert" className="list-disc space-y-1 pl-5 text-sm text-red-700">
          {messages.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      )}

      {staged.length > 0 && (
        <>
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3" aria-label="Selected photos">
            {staged.map((s) => (
              <li key={s.key} className="space-y-1 rounded-md border border-slate-200 p-2 text-xs">
                <img src={s.previewUrl} alt={`Preview of ${s.file.name}`} className="aspect-video w-full rounded object-cover" />
                <p className="truncate font-medium">{s.file.name}</p>
                <p className="text-slate-500">{formatBytes(s.file.size)}</p>
                {s.status === "uploading" && (
                  <progress value={s.progress} max={1} className="w-full" aria-label={`Uploading ${s.file.name}`} />
                )}
                {s.status === "error" && <p className="text-red-700">{s.error}</p>}
                <button
                  type="button"
                  disabled={s.status === "uploading"}
                  onClick={() => remove(s.key)}
                  className="text-slate-600 underline disabled:opacity-40"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
          <Button disabled={uploading} onClick={uploadAll}>
            {uploading ? "Uploading…" : `Upload ${staged.length} photo${staged.length === 1 ? "" : "s"}`}
          </Button>
        </>
      )}
    </section>
  );
}
