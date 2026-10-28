import type { ApiFieldError, Inspection, InspectionCreate, InspectionList } from "../types/api";

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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/api/v1${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(`Cannot reach the API at ${BASE_URL}. Is the backend running?`, null);
  }

  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    let details: ApiFieldError[] = [];
    try {
      const body = await response.json();
      message = body.error?.message ?? message;
      details = body.error?.details ?? [];
    } catch {
      // Non-JSON error body: keep the generic message.
    }
    throw new ApiError(message, response.status, details);
  }
  return (await response.json()) as T;
}

export const listInspections = (limit = 20, offset = 0) =>
  request<InspectionList>(`/inspections?limit=${limit}&offset=${offset}`);

export const getInspection = (id: string) =>
  request<Inspection>(`/inspections/${encodeURIComponent(id)}`);

export const createInspection = (payload: InspectionCreate) =>
  request<Inspection>("/inspections", { method: "POST", body: JSON.stringify(payload) });
