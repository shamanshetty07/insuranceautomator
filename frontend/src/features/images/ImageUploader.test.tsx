import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../lib/api";
import { IMAGE_LIMITS } from "../../lib/config";
import { ImageUploader } from "./ImageUploader";

const uploadImage = vi.fn();
vi.mock("../../lib/api", async (orig) => ({
  ...(await orig<typeof import("../../lib/api")>()),
  uploadImage: (...args: unknown[]) => uploadImage(...args),
}));

const png = (name = "a.png", size = 10) => new File([new Uint8Array(size)], name, { type: "image/png" });

function setup(storedCount = 0) {
  const onUploaded = vi.fn();
  render(<ImageUploader inspectionId="insp-1" storedCount={storedCount} onUploaded={onUploaded} />);
  return { onUploaded, input: screen.getByTestId("file-input") as HTMLInputElement };
}

beforeEach(() => {
  uploadImage.mockReset();
});

describe("ImageUploader", () => {
  it("previews selected images and allows removing them before upload", async () => {
    const { input } = setup();
    await userEvent.upload(input, [png("a.png"), png("b.png")]);
    expect(screen.getByAltText("Preview of a.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload 2 photos" })).toBeInTheDocument();

    await userEvent.click(screen.getAllByRole("button", { name: "Remove" })[0]);
    expect(screen.queryByAltText("Preview of a.png")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload 1 photo" })).toBeInTheDocument();
  });

  it("rejects wrong types and oversized files with clear messages", async () => {
    const { input } = setup();
    const user = userEvent.setup({ applyAccept: false });
    const gif = new File(["x"], "a.gif", { type: "image/gif" });
    const big = png("big.png", IMAGE_LIMITS.maxBytes + 1);
    await user.upload(input, [gif, big, png("empty.png", 0)]);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("a.gif: only JPEG, PNG or WebP");
    expect(alert).toHaveTextContent("big.png: larger than");
    expect(alert).toHaveTextContent("empty.png: the file is empty");
    expect(screen.queryByRole("button", { name: /^Upload/ })).not.toBeInTheDocument();
  });

  it("skips the same file selected twice", async () => {
    const { input } = setup();
    const file = png("same.png");
    await userEvent.upload(input, file);
    await userEvent.upload(input, file);
    expect(screen.getAllByAltText("Preview of same.png")).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent("already selected");
  });

  it("enforces the maximum image count including stored images", async () => {
    const { input } = setup(IMAGE_LIMITS.maxCount - 1);
    await userEvent.upload(input, [png("a.png"), png("b.png")]);
    expect(screen.getAllByRole("img")).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent("at most");
  });

  it("uploads each file once, ignores a double click, and reports completion", async () => {
    let finish!: () => void;
    uploadImage.mockImplementation(() => new Promise<void>((r) => (finish = r)));
    const { input, onUploaded } = setup();
    await userEvent.upload(input, png("a.png"));

    const button = screen.getByRole("button", { name: "Upload 1 photo" });
    await userEvent.dblClick(button);
    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(uploadImage.mock.calls[0][0]).toBe("insp-1");
    expect(screen.getByRole("button", { name: "Uploading…" })).toBeDisabled();

    finish();
    await waitFor(() => expect(onUploaded).toHaveBeenCalledOnce());
    expect(screen.queryByAltText("Preview of a.png")).not.toBeInTheDocument();
  });

  it("keeps failed files with the server's message so they can be retried or removed", async () => {
    uploadImage.mockRejectedValue(new ApiError("This image has already been uploaded to this inspection.", 409));
    const { input, onUploaded } = setup();
    await userEvent.upload(input, png("a.png"));
    await userEvent.click(screen.getByRole("button", { name: "Upload 1 photo" }));
    expect(await screen.findByText(/already been uploaded/)).toBeInTheDocument();
    expect(screen.getByAltText("Preview of a.png")).toBeInTheDocument();
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it("does not open the camera until the user asks", async () => {
    const getUserMedia = vi.fn().mockRejectedValue(new DOMException("denied", "NotAllowedError"));
    Object.defineProperty(navigator, "mediaDevices", { value: { getUserMedia }, configurable: true });
    setup();
    expect(getUserMedia).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Use camera" }));
    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(await screen.findByText(/permission was denied/i)).toBeInTheDocument();
  });
});
