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
