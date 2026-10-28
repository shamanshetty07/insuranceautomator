import type {
  ApiFieldError,
  DetectionRunList,
  DetectorStatus,
  Inspection,
  InspectionCreate,
  InspectionImage,
  InspectionImageList,
  InspectionList,
} from "../types/api";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly details: ApiFieldError[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const apiUrl = (path: string) => `${BASE_URL}${path}`;

function errorFromBody(status: number, text: string): ApiError {
  let message = `Request failed (HTTP ${status}).`;
  let details: ApiFieldError[] = [];
  try {
    const body = JSON.parse(text);
    message = body.error?.message ?? message;
    details = body.error?.details ?? [];
  } catch {
    // Non-JSON error body: keep the generic message.
  }
  return new ApiError(message, status, details);
}

const unreachable = () =>
  new ApiError(`Cannot reach the API at ${BASE_URL}. Is the backend running?`, null);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/api/v1${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw unreachable();
  }

  if (!response.ok) throw errorFromBody(response.status, await response.text());
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const listInspections = (limit = 20, offset = 0) =>
  request<InspectionList>(`/inspections?limit=${limit}&offset=${offset}`);

export const getInspection = (id: string) =>
  request<Inspection>(`/inspections/${encodeURIComponent(id)}`);

export const createInspection = (payload: InspectionCreate) =>
  request<Inspection>("/inspections", { method: "POST", body: JSON.stringify(payload) });

export const listImages = (inspectionId: string) =>
  request<InspectionImageList>(`/inspections/${encodeURIComponent(inspectionId)}/images`);

export const deleteImage = (inspectionId: string, imageId: string) =>
  request<void>(`/inspections/${encodeURIComponent(inspectionId)}/images/${encodeURIComponent(imageId)}`, {
    method: "DELETE",
  });

export const getDetectorStatus = () => request<DetectorStatus>("/detector");

/** Runs detection synchronously; omit imageIds to run on every image of the inspection. */
export const runDetection = (inspectionId: string, imageIds?: string[]) =>
  request<DetectionRunList>(`/inspections/${encodeURIComponent(inspectionId)}/detections`, {
    method: "POST",
    body: JSON.stringify({ image_ids: imageIds ?? null }),
  });

/** latest=true: newest run per image; false: every run, newest first. */
export const listDetections = (inspectionId: string, latest = true) =>
  request<DetectionRunList>(`/inspections/${encodeURIComponent(inspectionId)}/detections?latest=${latest}`);

/** Uploads one image with progress (fetch cannot report upload progress, XHR can). */
export function uploadImage(
  inspectionId: string,
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<InspectionImage> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/api/v1/inspections/${encodeURIComponent(inspectionId)}/images`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve(JSON.parse(xhr.responseText) as InspectionImage)
        : reject(errorFromBody(xhr.status, xhr.responseText));
    xhr.onerror = () => reject(unreachable());
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}
