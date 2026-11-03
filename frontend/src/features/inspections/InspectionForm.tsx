import { type FormEvent, useState } from "react";
import { Button } from "../../components/Button";
import { ApiError, createInspection } from "../../lib/api";
import type { Inspection } from "../../types/api";

interface Props {
  onCreated: (inspection: Inspection) => void;
  onCancel: () => void;
}

export function InspectionForm({ onCreated, onCancel }: Props) {
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [vehicle, setVehicle] = useState({ make: "", model: "", year: "", registration: "", color: "" });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const created = await createInspection({
        title: title.trim() || undefined,
        notes: notes.trim() || undefined,
        vehicle_make: vehicle.make.trim() || undefined,
        vehicle_model: vehicle.model.trim() || undefined,
        vehicle_year: vehicle.year.trim() ? Number(vehicle.year) : undefined,
        vehicle_registration: vehicle.registration.trim() || undefined,
        vehicle_color: vehicle.color.trim() || undefined,
      });
      onCreated(created);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setSubmitting(false);
    }
  }

  const field = "mt-1 block w-full rounded-md border border-slate-300 px-3 py-2 text-sm";

  return (
    <form onSubmit={handleSubmit} className="space-y-4 rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-semibold">New inspection</h2>
      <div>
        <label htmlFor="title" className="text-sm font-medium">
          Title <span className="font-normal text-slate-500">(optional)</span>
        </label>
        <input
          id="title"
          value={title}
          maxLength={200}
          onChange={(e) => setTitle(e.target.value)}
          className={field}
          placeholder="e.g. Rear bumper collision, 12 May"
        />
      </div>
      <div>
        <label htmlFor="notes" className="text-sm font-medium">
          Notes <span className="font-normal text-slate-500">(optional)</span>
        </label>
        <textarea
          id="notes"
          value={notes}
          maxLength={5000}
          rows={4}
          onChange={(e) => setNotes(e.target.value)}
          className={field}
        />
      </div>
      <fieldset className="grid gap-3 sm:grid-cols-2">
        <legend className="text-sm font-medium">
          Vehicle <span className="font-normal text-slate-500">(optional)</span>
        </legend>
        {(
          [
            ["make", "Make", 100],
            ["model", "Model", 100],
            ["registration", "Registration number", 32],
            ["color", "Colour", 50],
          ] as const
        ).map(([key, label, max]) => (
          <div key={key}>
            <label htmlFor={`vehicle-${key}`} className="text-sm">
              {label}
            </label>
            <input
              id={`vehicle-${key}`}
              value={vehicle[key]}
              maxLength={max}
              onChange={(e) => setVehicle({ ...vehicle, [key]: e.target.value })}
              className={field}
            />
          </div>
        ))}
        <div>
          <label htmlFor="vehicle-year" className="text-sm">
            Manufacturing year
          </label>
          <input
            id="vehicle-year"
            type="number"
            min={1886}
            max={2100}
            value={vehicle.year}
            onChange={(e) => setVehicle({ ...vehicle, year: e.target.value })}
            className={field}
          />
        </div>
      </fieldset>
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error.message}
        </p>
      )}
      <div className="flex gap-2">
        <Button type="submit" disabled={submitting}>
          {submitting ? "Creating…" : "Create inspection"}
        </Button>
        <Button type="button" variant="secondary" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
