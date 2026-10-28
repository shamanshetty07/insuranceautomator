import type { BoundingBox } from "../../types/api";

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/**
 * CSS placement for a normalized box over an element that exactly covers the stored
 * image (boxes are relative to the stored, EXIF-normalized pixels; origin top-left).
 */
export function boxStyle(box: BoundingBox) {
  const x0 = clamp01(box.x_min);
  const y0 = clamp01(box.y_min);
  const pct = (v: number) => `${+(v * 100).toFixed(4)}%`;
  return {
    left: pct(x0),
    top: pct(y0),
    width: pct(clamp01(box.x_max) - x0),
    height: pct(clamp01(box.y_max) - y0),
  };
}

export const formatConfidence = (c: number) => `${Math.round(c * 100)}%`;

export function formatDuration(ms: number | null) {
  if (ms === null) return null;
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}
