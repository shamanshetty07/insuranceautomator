import { IMAGE_LIMITS } from "../../lib/config";

export const formatBytes = (bytes: number) =>
  bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

// Client-side check only; the server re-validates the file.
export function validateImageFile(file: File): string | null {
  if (!(IMAGE_LIMITS.acceptedTypes as readonly string[]).includes(file.type)) {
    return `${file.name}: only JPEG, PNG or WebP images are supported.`;
  }
  if (file.size === 0) return `${file.name}: the file is empty.`;
  if (file.size > IMAGE_LIMITS.maxBytes) {
    return `${file.name}: larger than the ${formatBytes(IMAGE_LIMITS.maxBytes)} limit.`;
  }
  return null;
}
