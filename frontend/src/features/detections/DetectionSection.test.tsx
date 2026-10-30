import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DetectionRun, DetectorStatus, InspectionImage } from "../../types/api";
import { DetectionSection } from "./DetectionSection";
import { boxStyle, formatConfidence, formatDuration } from "./geometry";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const config = {
  confidence_threshold: 0.25, iou_threshold: 0.45, image_size: 640, max_detections: 100,
  class_filter: null, device: "cpu",
};
const available: DetectorStatus = {
  available: true, backend: "ultralytics", model_name: "yolo-damage", model_version: "1.0",
  weights_sha256: "b".repeat(64), classes: ["dent", "scratch"], config, reason: null,
};
const unavailable: DetectorStatus = {
  available: false, backend: "none", model_name: null, model_version: null, weights_sha256: null,
  classes: null, config: null, reason: "No detector is configured on the server.",
};
const image: InspectionImage = {
  id: "img1", inspection_id: "i1", original_filename: "door.jpg", media_type: "image/jpeg",
  file_size_bytes: 2048, sha256: "a".repeat(64), width_px: 1200, height_px: 800, exif_orientation: 1,
  image_order: 0, created_at: "2026-01-01T00:00:00Z", content_url: "/api/v1/inspections/i1/images/img1/content",
};
const makeRun = (over: Partial<DetectionRun> = {}): DetectionRun => ({
  id: "run1", inspection_image_id: "img1", model_name: "yolo-damage", model_version: "1.0",
  weights_sha256: null, config, status: "completed", error_message: null,
  created_at: "2026-01-01T00:00:00Z", completed_at: "2026-01-01T00:00:01Z", duration_ms: 420,
  predictions: [], ...over,
});

function mockApi(opts: {
  detector?: DetectorStatus;
  images?: InspectionImage[];
  detections?: DetectionRun[];
  post?: () => Response;
}) {
  const state = { detections: opts.detections ?? [] };
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/detector")) return json(opts.detector ?? available);
    if (url.includes("/images")) return json({ items: opts.images ?? [image], total: (opts.images ?? [image]).length });
    if (url.includes("/detections") && init?.method === "POST") {
      return opts.post ? opts.post() : json({ items: state.detections });
    }
    if (url.includes("/detections")) return json({ items: state.detections });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, state };
}

afterEach(() => vi.unstubAllGlobals());

describe("DetectionSection", () => {
  it("explains an unavailable detector and disables the run button", async () => {
    mockApi({ detector: unavailable });
    render(<DetectionSection inspectionId="i1" />);
    expect(await screen.findByRole("note")).toHaveTextContent("No detector is configured on the server.");
    await screen.findByText("No detection runs yet.");
    expect(screen.getByRole("button", { name: "Run detection" })).toBeDisabled();
  });

  it("asks for images when the inspection has none", async () => {
    mockApi({ images: [] });
    render(<DetectionSection inspectionId="i1" />);
    expect(await screen.findByText(/Add images before running/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run detection" })).toBeDisabled();
  });

  it("runs detection once and shows predictions with boxes", async () => {
    const { fetchMock, state } = mockApi({});
    let resolvePost: (r: Response) => void = () => {};
    const run = makeRun({
      predictions: [
        { id: "p1", damage_type: "dent", confidence: 0.91, bounding_box: { x_min: 0.1, y_min: 0.2, x_max: 0.5, y_max: 0.6 } },
        { id: "p2", damage_type: "scratch", confidence: 0.4, bounding_box: null },
      ],
    });
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes("/detections") && init?.method === "POST") {
        return new Promise<Response>((r) => {
          resolvePost = (resp) => {
            state.detections = [run];
            r(resp);
          };
        });
      }
      return base(url, init);
    });

    const user = userEvent.setup();
    render(<DetectionSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Run detection" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    expect(screen.getByRole("button", { name: "Running detection…" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Running detection…" }));
    const posts = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
    expect(posts).toHaveLength(1);
    expect(JSON.parse(posts[0][1]!.body as string)).toEqual({ image_ids: null });

    resolvePost(json({ items: [run] }));
    const list = await screen.findByRole("list", { name: "Detections for Photo 1" });
    expect(within(list).getByText("dent")).toBeInTheDocument();
    expect(within(list).getByText("91%")).toBeInTheDocument();
    expect(within(list).getByText("scratch")).toBeInTheDocument();
    expect(screen.getByText(/420 ms/)).toBeInTheDocument();
    const boxes = screen.getAllByTestId("detection-box");
    expect(boxes).toHaveLength(1);
    expect(boxes[0]).toHaveStyle({ left: "10%", top: "20%", width: "40%", height: "40%" });
  });

  it("says the model detected nothing rather than that there is no damage", async () => {
    mockApi({ detections: [makeRun()] });
    render(<DetectionSection inspectionId="i1" />);
    expect(await screen.findByText("No damage detected by this model.")).toBeInTheDocument();
    expect(screen.queryByTestId("detection-box")).not.toBeInTheDocument();
  });

  it("shows a failed run's error message", async () => {
    mockApi({ detections: [makeRun({ status: "failed", error_message: "Inference error", predictions: [] })] });
    render(<DetectionSection inspectionId="i1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Detection failed: Inference error");
  });

  it("shows API errors from running detection", async () => {
    mockApi({
      post: () => json({ error: { code: "detector_unavailable", message: "Detector weights could not be loaded." } }, 503),
    });
    const user = userEvent.setup();
    render(<DetectionSection inspectionId="i1" />);
    const button = await screen.findByRole("button", { name: "Run detection" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("Detector weights could not be loaded.");
    expect(screen.getByRole("button", { name: "Run detection" })).toBeEnabled();
  });

  it("shows an error with retry when results fail to load", async () => {
    const { fetchMock } = mockApi({});
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) =>
      url.includes("/detections") ? json({ error: { code: "x", message: "Boom" } }, 500) : base(url, init),
    );
    render(<DetectionSection inspectionId="i1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Boom");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

describe("geometry", () => {
  it("maps normalized boxes to percentages and clamps out-of-range values", () => {
    expect(boxStyle({ x_min: 0.25, y_min: 0, x_max: 0.75, y_max: 1 })).toEqual({
      left: "25%", top: "0%", width: "50%", height: "100%",
    });
    expect(boxStyle({ x_min: -0.1, y_min: 0.5, x_max: 1.2, y_max: 0.9 })).toEqual({
      left: "0%", top: "50%", width: "100%", height: "40%",
    });
    expect(boxStyle({ x_min: 0.1, y_min: 0.2, x_max: 0.3, y_max: 0.7 }).width).toBe("20%");
  });

  it("formats confidence and duration", () => {
    expect(formatConfidence(0.876)).toBe("88%");
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration(420)).toBe("420 ms");
    expect(formatDuration(2500)).toBe("2.5 s");
  });
});
