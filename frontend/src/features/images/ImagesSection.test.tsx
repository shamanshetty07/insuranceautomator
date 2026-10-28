import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ImagesSection } from "./ImagesSection";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("ImagesSection", () => {
  it("shows loading, then an empty state without touching the camera", async () => {
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", { value: { getUserMedia }, configurable: true });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ items: [], total: 0 })));
    render(<ImagesSection inspectionId="i1" />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading images");
    expect(await screen.findByText(/No images yet/)).toBeInTheDocument();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("shows an error with retry when the API fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: { code: "x", message: "Boom" } }, 500)));
    render(<ImagesSection inspectionId="i1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Boom");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("shows stored images with order and normalized dimensions", async () => {
    const image = {
      id: "img1", inspection_id: "i1", original_filename: "door.jpg", media_type: "image/jpeg",
      file_size_bytes: 2048, sha256: "a".repeat(64), width_px: 1200, height_px: 800, exif_orientation: 6,
      image_order: 0, created_at: "2026-01-01T00:00:00Z", content_url: "/api/v1/inspections/i1/images/img1/content",
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ items: [image], total: 1 })));
    render(<ImagesSection inspectionId="i1" />);
    expect(await screen.findByText("#1")).toBeInTheDocument();
    expect(screen.getByText(/1200 × 800 px/)).toBeInTheDocument();
    expect(screen.getByAltText("Photo 1: door.jpg")).toHaveAttribute("src", expect.stringContaining("/images/img1/content"));
  });
});
