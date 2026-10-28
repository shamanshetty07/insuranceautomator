export type InspectionStatus =
  | "created"
  | "images_pending"
  | "ready_for_analysis"
  | "processing"
  | "review_required"
  | "completed"
  | "failed";

export interface Inspection {
  id: string;
  title: string | null;
  notes: string | null;
  status: InspectionStatus;
  created_at: string;
  updated_at: string;
}

export interface InspectionList {
  items: Inspection[];
  total: number;
  limit: number;
  offset: number;
}

export interface InspectionCreate {
  title?: string;
  notes?: string;
}

export interface ApiFieldError {
  field: string;
  message: string;
}

export interface InspectionImage {
  id: string;
  inspection_id: string;
  original_filename: string;
  media_type: string;
  file_size_bytes: number;
  sha256: string | null;
  /** Pixel size of the stored, EXIF-normalized image (the space bounding boxes refer to). */
  width_px: number | null;
  height_px: number | null;
  /** Original EXIF orientation; metadata only, already applied to the stored pixels. */
  exif_orientation: number | null;
  image_order: number;
  created_at: string;
  /** Path relative to the API base URL; use apiUrl() to build the full URL. */
  content_url: string;
}

export interface InspectionImageList {
  items: InspectionImage[];
  total: number;
}

export type DetectorBackend = "none" | "ultralytics";

export interface DetectionConfig {
  confidence_threshold: number;
  iou_threshold: number;
  image_size: number;
  max_detections: number;
  class_filter: string[] | null;
  device: string | null;
}

export interface DetectorStatus {
  available: boolean;
  backend: DetectorBackend;
  model_name: string | null;
  model_version: string | null;
  weights_sha256: string | null;
  /** Model class names, verbatim. */
  classes: string[] | null;
  config: DetectionConfig | null;
  /** Human-readable explanation when available is false. */
  reason: string | null;
}

/** Normalized to [0,1] relative to the stored image; origin top-left, y down. */
export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface DamagePrediction {
  id: string;
  damage_type: string;
  confidence: number;
  bounding_box: BoundingBox | null;
}

export type DetectionRunStatus = "pending" | "running" | "completed" | "failed";

export interface DetectionRun {
  id: string;
  inspection_image_id: string;
  model_name: string;
  model_version: string;
  weights_sha256: string | null;
  config: DetectionConfig;
  status: DetectionRunStatus;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  /** Ordered by confidence, descending. */
  predictions: DamagePrediction[];
}

export interface DetectionRunList {
  items: DetectionRun[];
}
