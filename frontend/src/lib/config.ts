const positiveInt = (value: string | undefined, fallback: number) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
};

export const IMAGE_LIMITS = {
  maxBytes: positiveInt(import.meta.env.VITE_MAX_IMAGE_BYTES, 15 * 1024 * 1024),
  maxCount: positiveInt(import.meta.env.VITE_MAX_IMAGES_PER_INSPECTION, 20),
  acceptedTypes: ["image/jpeg", "image/png", "image/webp"],
} as const;
