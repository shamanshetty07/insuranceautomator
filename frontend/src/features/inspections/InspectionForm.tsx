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
