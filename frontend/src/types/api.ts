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
  vehicle_make?: string | null;
  vehicle_model?: string | null;
  vehicle_year?: number | null;
  vehicle_registration?: string | null;
  vehicle_color?: string | null;
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
  vehicle_make?: string;
  vehicle_model?: string;
  vehicle_year?: number;
  vehicle_registration?: string;
  vehicle_color?: string;
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
  width_px: number | null;
  height_px: number | null;
  exif_orientation: number | null;
  image_order: number;
  created_at: string;
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
  classes: string[] | null;
  config: DetectionConfig | null;
  reason: string | null;
}

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
  predictions: DamagePrediction[];
}

export interface DetectionRunList {
  items: DetectionRun[];
}

/** Result returned after the server runs detection, report generation, and verification. */
export interface InspectionRunResult {
  inspection: Inspection;
  detections: DetectionRun[];
  report: ReportDetail;
}

export type RunStatus = DetectionRunStatus;

export interface VlmStatus {
  available: boolean;
  backend: string;
  reason: string | null;
  model_name: string | null;
  model_version: string | null;
  weights_sha256: string | null;
}

export type ClaimType =
  | "visible_damage_presence"
  | "damage_type"
  | "damage_location"
  | "damage_severity"
  | "hidden_structural_damage"
  | "repair_recommendation"
  | "repair_cost"
  | "damage_cause"
  | "other";

export type Verdict = "supported" | "contradicted" | "insufficient_evidence";

export type VerificationLabel =
  | "supported"
  | "contradicted"
  | "wrong_location"
  | "wrong_damage_type"
  | "unsupported_severity"
  | "unverifiable"
  | "omitted_damage";

export type EvidenceStrategy = "full_image" | "cropped_region" | "image_with_segmentation_mask";

export interface EvidenceReference {
  image_id: string;
  prediction_id: string | null;
  crop: BoundingBox | null;
  note: string | null;
}

export interface ClaimResult {
  /** Present once a claim-level verification row has been persisted. */
  id?: string;
  verdict: Verdict | null;
  needs_human_review: boolean;
  policy_reason_codes: string[];
  verification_label: VerificationLabel | null;
  explanation: string;
  review_decision?: HumanReviewDecision | null;
}

export type HumanReviewOutcome = "approved" | "rejected";

export interface HumanReviewDecision {
  id: string;
  verification_result_id: string;
  outcome: HumanReviewOutcome;
  reviewer_note: string | null;
  reviewed_at: string;
}

export interface ReportClaim {
  id: string;
  claim_id: string;
  position: number;
  text: string;
  claim_type: ClaimType;
  damage_type: string | null;
  image_ids: string[];
  prediction_ids: string[];
  evidence: EvidenceReference[];
  result: ClaimResult | null;
}

export interface VerificationRun {
  id: string;
  report_id: string;
  verifier_model: string;
  verifier_version: string;
  status: RunStatus;
  error_message: string | null;
  policy_version: string | null;
  created_at: string;
  completed_at: string | null;
  review_count: number;
}

export interface ReportSummary {
  id: string;
  inspection_id: string;
  content: string;
  model_name: string;
  model_version: string;
  evidence_strategy: EvidenceStrategy;
  created_at: string;
  claim_count: number;
}

export interface ReportDetail extends ReportSummary {
  claims: ReportClaim[];
  verification: VerificationRun | null;
}

export interface ReportList {
  items: ReportSummary[];
}

export type AssessmentClaimStatus = "supported" | "uncertain" | "unsupported";
export type AssessmentState =
  | "no_findings"
  | "awaiting_verification"
  | "verification_failed"
  | "pending_human_review"
  | "ready";

export interface AssessmentVehicle {
  make: string;
  model: string;
  year: string;
  registration_number: string;
  color: string;
}

export interface AssessmentObservation {
  claim_id: string;
  text: string;
  status: AssessmentClaimStatus;
}

export interface AssessmentFinding {
  damage_id: string;
  prediction_id: string;
  image_id: string | null;
  image_label: string;
  component: string;
  damage_type: string;
  severity: string;
  detection_confidence: number;
  bounding_box: BoundingBox | null;
  observations: AssessmentObservation[];
  repair_considerations: string[];
  additional_inspection: string[];
}

export interface AssessmentClaim {
  claim_id: string;
  claim_text: string;
  claim_type: string;
  damage_ids: string[];
  image_ids: string[];
  status: AssessmentClaimStatus;
  verifier_label: string | null;
  explanation: string;
  needs_human_review: boolean;
  review_reason_codes: string[];
}

export interface AssessmentReviewItem {
  claim_id: string;
  required: boolean;
  outcome: string | null;
  reviewer_note: string | null;
  reviewed_at: string | null;
}

export interface AssessmentReport {
  header: {
    report_id: string;
    inspection_id: string;
    inspection_title: string | null;
    generated_at: string;
    inspection_type: string;
    assessment_status: AssessmentState;
    report_model: string;
  };
  vehicle: AssessmentVehicle;
  executive_summary: string;
  findings: AssessmentFinding[];
  claims: AssessmentClaim[];
  recommendations: string[];
  cost_estimate: string;
  human_review: {
    ai_assessment_status: string;
    verification_status: "not_run" | "running" | "completed" | "failed";
    review_status: "not_started" | "not_required" | "pending" | "complete";
    items: AssessmentReviewItem[];
  };
  evidence_notice: string;
  disclaimer: string;
}
