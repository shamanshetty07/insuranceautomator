import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CameraCapture } from "./CameraCapture";

function fakeStream() {
  const track = { stop: vi.fn() };
  return { stream: { getTracks: () => [track] } as unknown as MediaStream, track };
}

function stubMediaDevices(getUserMedia: unknown) {
  Object.defineProperty(navigator, "mediaDevices", { value: getUserMedia ? { getUserMedia } : undefined, configurable: true });
}

afterEach(() => {
  vi.restoreAllMocks();
  stubMediaDevices(undefined);
});

describe("CameraCapture", () => {
  it("explains when the browser has no camera API", async () => {
    stubMediaDevices(undefined);
    render(<CameraCapture onCapture={vi.fn()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/cannot access the camera/i);
  });

  it("shows a useful message when permission is denied", async () => {
    stubMediaDevices(vi.fn().mockRejectedValue(new DOMException("denied", "NotAllowedError")));
    render(<CameraCapture onCapture={vi.fn()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/permission was denied/i);
  });

  it("shows a useful message when no camera exists", async () => {
    stubMediaDevices(vi.fn().mockRejectedValue(new DOMException("none", "NotFoundError")));
    render(<CameraCapture onCapture={vi.fn()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/no camera was found/i);
  });

  it("stops the camera tracks on unmount", async () => {
    const { stream, track } = fakeStream();
    stubMediaDevices(vi.fn().mockResolvedValue(stream));
    const { unmount } = render(<CameraCapture onCapture={vi.fn()} onClose={vi.fn()} />);
    await screen.findByRole("button", { name: "Capture photo" });
    unmount();
    expect(track.stop).toHaveBeenCalled();
  });

  it("stops the tracks when the camera is closed", async () => {
    const { stream, track } = fakeStream();
    stubMediaDevices(vi.fn().mockResolvedValue(stream));
    const onClose = vi.fn();
    render(<CameraCapture onCapture={vi.fn()} onClose={onClose} />);
    await userEvent.click(await screen.findByRole("button", { name: "Close camera" }));
    expect(track.stop).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("stops a stream that arrives after the component was already closed", async () => {
    const { stream, track } = fakeStream();
    let resolve!: (s: MediaStream) => void;
    stubMediaDevices(vi.fn(() => new Promise<MediaStream>((r) => (resolve = r))));
    const { unmount } = render(<CameraCapture onCapture={vi.fn()} onClose={vi.fn()} />);
    unmount();
    resolve(stream);
    await waitFor(() => expect(track.stop).toHaveBeenCalled());
  });

  it("captures, allows retaking, and hands the photo over only when accepted", async () => {
    const { stream } = fakeStream();
    stubMediaDevices(vi.fn().mockResolvedValue(stream));
    Object.defineProperty(HTMLVideoElement.prototype, "videoWidth", { value: 640, configurable: true });
    Object.defineProperty(HTMLVideoElement.prototype, "videoHeight", { value: 480, configurable: true });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({ drawImage: vi.fn() } as never);
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((cb) => cb(new Blob(["x"], { type: "image/jpeg" })));
    const onCapture = vi.fn();
    render(<CameraCapture onCapture={onCapture} onClose={vi.fn()} />);

    await userEvent.click(await screen.findByRole("button", { name: "Capture photo" }));
    expect(screen.getByAltText("Captured photo preview")).toBeInTheDocument();
    expect(onCapture).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: "Retake" }));
    expect(screen.getByRole("button", { name: "Capture photo" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Capture photo" }));
    await userEvent.click(screen.getByRole("button", { name: "Use photo" }));
    expect(onCapture).toHaveBeenCalledOnce();
    expect(onCapture.mock.calls[0][0]).toBeInstanceOf(File);
    expect(onCapture.mock.calls[0][0].type).toBe("image/jpeg");
  });
});
