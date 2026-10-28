import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { InspectionsPage } from "./InspectionsPage";

function mockFetch(response: Response | Error) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => (response instanceof Error ? Promise.reject(response) : Promise.resolve(response))),
  );
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const renderPage = () =>
  render(
    <MemoryRouter>
      <InspectionsPage />
    </MemoryRouter>,
  );

afterEach(() => vi.unstubAllGlobals());

describe("InspectionsPage", () => {
  it("shows a loading state, then the empty state when the API returns no inspections", async () => {
    mockFetch(json({ items: [], total: 0, limit: 20, offset: 0 }));
    renderPage();
    expect(screen.getByRole("status")).toHaveTextContent("Loading inspections");
    expect(await screen.findByText("No inspections yet")).toBeInTheDocument();
  });

  it("shows an error state with retry when the API is unreachable", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    renderPage();
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the API");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("lists inspections returned by the API", async () => {
    mockFetch(
      json({
        items: [
          {
            id: "11111111-1111-1111-1111-111111111111",
            title: "Door dent",
            notes: null,
            status: "created",
            created_at: "2026-01-01T10:00:00Z",
            updated_at: "2026-01-01T10:00:00Z",
          },
        ],
        total: 1,
        limit: 20,
        offset: 0,
      }),
    );
    renderPage();
    expect(await screen.findByText("Door dent")).toBeInTheDocument();
  });

  it("opens the creation form from the empty state", async () => {
    mockFetch(json({ items: [], total: 0, limit: 20, offset: 0 }));
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Create inspection" }));
    await waitFor(() => expect(screen.getByLabelText(/Title/)).toBeInTheDocument());
  });
});
