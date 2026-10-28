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
