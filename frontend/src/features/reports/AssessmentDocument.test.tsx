import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AssessmentReport, InspectionImage } from "../../types/api";
import { AssessmentDocument } from "./AssessmentDocument";

const image: InspectionImage = {
  id: "img1", inspection_id: "i1", original_filename: "door.jpg", media_type: "image/jpeg",
  file_size_bytes: 2048, sha256: null, width_px: 1200, height_px: 800, exif_orientation: 1,
  image_order: 0, created_at: "2026-01-01T00:00:00Z", content_url: "/api/v1/inspections/i1/images/img1/content",
};
const images = new Map([[image.id, image]]);

const base = (over: Partial<AssessmentReport> = {}): AssessmentReport => ({
  header: {
    report_id: "r1", inspection_id: "i1", inspection_title: "Door", generated_at: "2026-01-01T00:00:00Z",
    inspection_type: "AI-assisted visual damage assessment", assessment_status: "pending_human_review",
    report_model: "qwen 1",
  },
  vehicle: { make: "Honda", model: "Not Provided", year: "2019", registration_number: "Not Provided", color: "Not Provided" },
  executive_summary: "The detector reported 1 damage instance.",
  findings: [{
    damage_id: "D1", prediction_id: "p1", image_id: "img1", image_label: "Photo 1",
    component: "Not identified by detector", damage_type: "dent", severity: "Not assessed",
    detection_confidence: 0.82, bounding_box: { x_min: 0.1, y_min: 0.2, x_max: 0.5, y_max: 0.6 },
    observations: [{ claim_id: "c1", text: "There is a dent.", status: "uncertain" }],
    repair_considerations: ["Possible panel repair"], additional_inspection: ["Physical inspection of the affected area"],
  }],
  claims: [],
  recommendations: ["Arrange a physical inspection to confirm the detected damage."],
  cost_estimate: "No repair cost estimate is available: this system has no pricing data.",
  human_review: {
    ai_assessment_status: "AI-generated, verified by the verifier model", verification_status: "completed",
    review_status: "pending", items: [{ claim_id: "c1", required: true, outcome: null, reviewer_note: null, reviewed_at: null }],
  },
  evidence_notice: "Detections are model output.",
  disclaimer: "This report is an AI-assisted preliminary assessment. It is not a final insurance claim approval.",
  ...over,
});

const renderDoc = (a: AssessmentReport) =>
  render(<AssessmentDocument assessment={a} images={images} claimsSection={<p>claims go here</p>} />);

describe("AssessmentDocument", () => {
  it("renders every section heading in order", () => {
    renderDoc(base());
    const headings = screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent);
    expect(headings).toEqual([
      "A. Report header", "B. Vehicle information", "C. Executive summary", "D. Damage detection summary",
      "E. Detailed damage assessment", "F. Visual evidence", "G. AI claim verification",
      "H. Repair recommendations", "I. Human review", "J. Disclaimer",
    ]);
    expect(screen.getByText("claims go here")).toBeInTheDocument();
    expect(screen.getByText(/not a final insurance claim approval/)).toBeInTheDocument();
  });

  it("shows missing vehicle information as Not Provided", () => {
    renderDoc(base());
    const section = screen.getByRole("region", { name: /Vehicle information/ });
    expect(within(section).getAllByText("Not Provided")).toHaveLength(3);
    expect(within(section).getByText("Honda")).toBeInTheDocument();
  });

  it("shows the damage table with a confidence badge and the annotated evidence box", () => {
    renderDoc(base());
    const table = screen.getByRole("table");
    expect(within(table).getByText("dent")).toBeInTheDocument();
    expect(within(table).getByText("82% (high)")).toBeInTheDocument();
    expect(screen.getByTestId("annotation-D1")).toBeInTheDocument();
    expect(screen.getByAltText("Original Photo 1")).toBeInTheDocument();
    expect(screen.getByText(/x 0.100–0.500, y 0.200–0.600/, { selector: "dd" })).toBeInTheDocument();
  });

  it("states that nothing was detected without calling the vehicle undamaged", () => {
    renderDoc(base({ findings: [], recommendations: ["No damage was detected."] }));
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.getAllByText(/not a finding that the vehicle is undamaged/)).not.toHaveLength(0);
    expect(screen.queryByTestId("annotation-D1")).not.toBeInTheDocument();
  });

  it("labels AI observations as unconfirmed and shows their verification status", () => {
    renderDoc(base({
      findings: [{ ...base().findings[0], observations: [{ claim_id: "c2", text: "Large dent.", status: "unsupported" }] }],
    }));
    const card = screen.getByRole("listitem", { name: "Damage D1" });
    expect(within(card).getByText(/not confirmed facts/)).toBeInTheDocument();
    expect(within(card).getByText("Unsupported")).toBeInTheDocument();
  });

  it("shows missing repair suggestions honestly and no invented cost", () => {
    renderDoc(base({ findings: [{ ...base().findings[0], repair_considerations: [] }] }));
    expect(screen.getByText(/None supported by the evidence/)).toBeInTheDocument();
    expect(screen.getByText(/no pricing data/)).toBeInTheDocument();
    expect(screen.queryByText(/\$|₹|€/)).not.toBeInTheDocument();
  });

  it("shows pending human review, then the reviewer decision and comment", () => {
    const { unmount } = renderDoc(base());
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getByText(/awaiting reviewer decision/)).toBeInTheDocument();
    unmount();
    renderDoc(base({
      human_review: {
        ai_assessment_status: "AI-generated, verified by the verifier model", verification_status: "completed",
        review_status: "complete",
        items: [{ claim_id: "c1", required: true, outcome: "rejected", reviewer_note: "Not a dent", reviewed_at: "2026-01-02T00:00:00Z" }],
      },
    }));
    const list = screen.getByRole("list", { name: "Reviewer decisions" });
    expect(within(list).getByText("rejected")).toBeInTheDocument();
    expect(within(list).getByText(/Not a dent/)).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
  });

  it("handles an unverified report", () => {
    renderDoc(base({
      header: { ...base().header, assessment_status: "awaiting_verification" },
      human_review: { ai_assessment_status: "AI-generated and not yet verified", verification_status: "not_run", review_status: "not_started", items: [] },
    }));
    expect(screen.getByText("Awaiting verification")).toBeInTheDocument();
    expect(screen.getByText("Not started")).toBeInTheDocument();
    expect(screen.getByText("not run")).toBeInTheDocument();
  });
});
