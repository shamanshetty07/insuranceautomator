import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { InspectionImage, ReportClaim, ReportDetail, VlmStatus } from "../../types/api";
import { ReportSection } from "./ReportSection";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const apiError = (status: number, code: string, message: string) => json({ error: { code, message, details: [] } }, status);

const available: VlmStatus = {
  available: true, backend: "transformers", reason: null, model_name: "qwen2-vl-2b", model_version: "1",
  weights_sha256: null,
};
const unavailable: VlmStatus = {
  available: false, backend: "none", reason: "The vision-language model is disabled (VLM_BACKEND=none).",
  model_name: null, model_version: null, weights_sha256: null,
};
const image: InspectionImage = {
  id: "img1", inspection_id: "i1", original_filename: "door.jpg", media_type: "image/jpeg",
  file_size_bytes: 2048, sha256: "a".repeat(64), width_px: 1200, height_px: 800, exif_orientation: 1,
  image_order: 0, created_at: "2026-01-01T00:00:00Z", content_url: "/api/v1/inspections/i1/images/img1/content",
};
const BOX = { x_min: 0.1, y_min: 0.2, x_max: 0.5, y_max: 0.6 };

const claim = (over: Partial<ReportClaim> = {}): ReportClaim => ({
  id: "rc1", claim_id: "c1", position: 0, text: "There is a dent on the door.", claim_type: "damage_type",
  damage_type: "dent", image_ids: ["img1"], prediction_ids: ["p1"],
  evidence: [{ image_id: "img1", prediction_id: "p1", crop: BOX, note: null }], result: null, ...over,
});
const report = (over: Partial<ReportDetail> = {}): ReportDetail => ({
  id: "r1", inspection_id: "i1", content: "There is a dent on the door.", model_name: "qwen2-vl-2b",
  model_version: "1", evidence_strategy: "full_image", created_at: "2026-01-01T00:00:00Z", claim_count: 1,
  claims: [claim()], verification: null, ...over,
});
const verified = report({
  claims: [claim({
    result: {
      verdict: "supported", needs_human_review: true, policy_reason_codes: ["detection_inconclusive"],
      verification_label: "supported", explanation: "A dent is visible.",
    },
  })],
  verification: {
    id: "v1", report_id: "r1", verifier_model: "qwen2-vl-2b", verifier_version: "1", status: "completed",
    error_message: null, policy_version: "claim-policy-4", created_at: "2026-01-01T00:01:00Z",
    completed_at: "2026-01-01T00:02:00Z", review_count: 1,
  },
});

function mockApi(opts: {
  vlm?: VlmStatus;
  images?: InspectionImage[];
  report?: ReportDetail | null;
  generate?: () => Response | Promise<Response>;
  verify?: () => Response;
  list?: () => Response;
}) {
  const state = { report: opts.report ?? null };
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const post = init?.method === "POST";
    if (url.endsWith("/vlm")) return json(opts.vlm ?? available);
    if (url.includes("/images")) return json({ items: opts.images ?? [image], total: 1 });
    if (url.endsWith("/inspections/i1/reports") && post) return opts.generate ? opts.generate() : json(state.report, 201);
    if (url.endsWith("/inspections/i1/reports")) {
      if (opts.list) return opts.list();
      return json({ items: state.report ? [state.report] : [] });
    }
    if (url.endsWith("/reports/r1/verifications") && post) return opts.verify ? opts.verify() : json(verified, 201);
    if (url.endsWith("/reports/r1")) return json(state.report);
    throw new Error(`unexpected ${init?.method ?? "GET"} ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, state };
}

const posts = (fetchMock: ReturnType<typeof vi.fn>, suffix: string) =>
  fetchMock.mock.calls.filter(([url, init]) => String(url).endsWith(suffix) && init?.method === "POST").length;

afterEach(() => vi.unstubAllGlobals());

describe("ReportSection", () => {
  it("shows the AI disclaimer and an empty state without a report", async () => {
    mockApi({});
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText("No report generated yet.")).toBeInTheDocument();
    expect(screen.getByText(/AI-generated predictions, not ground truth/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Verify claims" })).toBeDisabled();
  });

  it("generates once despite double clicks and then shows the claims", async () => {
    let resolve: (r: Response) => void = () => {};
    const { fetchMock, state } = mockApi({ generate: () => new Promise<Response>((r) => (resolve = r)) });
    render(<ReportSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Generate report" });
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
    await userEvent.click(button);
    expect(screen.getByRole("button", { name: "Generating report…" })).toBeDisabled();
    expect(screen.getByText(/can take several minutes/)).toBeInTheDocument();
    state.report = report();
    resolve(json(state.report, 201));
    expect(await screen.findByText("There is a dent on the door.")).toBeInTheDocument();
    expect(posts(fetchMock, "/inspections/i1/reports")).toBe(1);
    expect(screen.getByText("Not verified")).toBeInTheDocument();
  });

  it("renders claim fields, verdict text, review reasons and the verifier explanation", async () => {
    mockApi({ report: verified });
    render(<ReportSection inspectionId="i1" />);
    const row = await screen.findByRole("listitem", { name: "Claim c1" });
    expect(within(row).getByText("damage type")).toBeInTheDocument();
    expect(within(row).getByText("dent")).toBeInTheDocument();
    expect(within(row).getByText("Supported")).toBeInTheDocument();
    expect(within(row).getByText("Needs human review")).toBeInTheDocument();
    expect(within(row).getByText("detection_inconclusive")).toBeInTheDocument();
    expect(within(row).getByText(/A dent is visible\./)).toBeInTheDocument();
    expect(screen.getByText(/policy claim-policy-4/)).toHaveTextContent("1 claim(s) need human review");
  });

  it("draws evidence thumbnails with the crop box, and none without a crop", async () => {
    mockApi({
      report: report({
        claims: [
          claim(),
          claim({ id: "rc2", claim_id: "c2", position: 1, evidence: [{ image_id: "img1", prediction_id: null, crop: null, note: null }] }),
        ],
      }),
    });
    render(<ReportSection inspectionId="i1" />);
    const img = await screen.findByAltText("Evidence for claim c1: Photo 1");
    expect(img.getAttribute("src")).toMatch(/\/api\/v1\/inspections\/i1\/images\/img1\/content$/);
    const crops = screen.getAllByTestId("evidence-crop");
    expect(crops).toHaveLength(1);
    expect(crops[0]).toHaveStyle({ left: "10%", top: "20%", width: "40%", height: "40%" });
    expect(screen.getByAltText("Evidence for claim c2: Photo 1")).toBeInTheDocument();
  });

  it("explains an unavailable model and a busy model", async () => {
    mockApi({ generate: () => apiError(503, "vlm_unavailable", "The model failed to load.") });
    render(<ReportSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Generate report" });
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("The report model is unavailable: The model failed to load.");
    expect(screen.getByRole("button", { name: "Generate report" })).toBeEnabled();

    vi.unstubAllGlobals();
    mockApi({ generate: () => apiError(503, "vlm_busy", "busy") });
    await userEvent.click(screen.getByRole("button", { name: "Generate report" }));
    expect(await screen.findByText(/The model is busy with another request/)).toBeInTheDocument();
  });

  it("shows timeout copy for a 504", async () => {
    mockApi({ generate: () => apiError(504, "vlm_timeout", "The vision-language model did not finish in time.") });
    render(<ReportSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Generate report" });
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("Report generation timed out");
  });

  it("offers retry when the report list fails to load", async () => {
    let fail = true;
    mockApi({ list: () => (fail ? apiError(500, "internal_error", "Server exploded.") : json({ items: [] })) });
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText("Server exploded.")).toBeInTheDocument();
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("No report generated yet.")).toBeInTheDocument();
  });

  it("never presents zero claims as an all-clear", async () => {
    mockApi({ report: report({ claims: [], claim_count: 0 }) });
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText(/not a finding that the vehicle is undamaged/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Verify claims" })).toBeDisabled();
  });

  it("renders a result with no verdict and an omitted_damage label", async () => {
    mockApi({
      report: report({
        claims: [claim({
          result: {
            verdict: null, needs_human_review: true, policy_reason_codes: [], verification_label: "omitted_damage",
            explanation: "",
          },
        })],
      }),
    });
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText("No verdict")).toBeInTheDocument();
    expect(screen.getByText(/Verifier: omitted damage/)).toBeInTheDocument();
  });

  it("shows a fallback for evidence images that no longer exist", async () => {
    mockApi({ images: [], report: report() });
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText("Image unavailable")).toBeInTheDocument();
  });

  it("disables generation when the model is unavailable", async () => {
    mockApi({ vlm: unavailable });
    render(<ReportSection inspectionId="i1" />);
    expect(await screen.findByText(/Report generation is unavailable: The vision-language model is disabled/)).toBeInTheDocument();
    await screen.findByText("No report generated yet.");
    expect(screen.getByRole("button", { name: "Generate report" })).toBeDisabled();
  });

  it("verifies claims and shows the results with review flags", async () => {
    const { fetchMock, state } = mockApi({ report: report() });
    render(<ReportSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Verify claims" });
    await waitFor(() => expect(button).toBeEnabled());
    state.report = verified;
    await userEvent.click(button);
    expect(await screen.findByText("Needs human review")).toBeInTheDocument();
    expect(posts(fetchMock, "/reports/r1/verifications")).toBe(1);
  });
});
