import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../../components/Button";
import { ErrorState } from "../../components/ErrorState";
import { Spinner } from "../../components/Spinner";
import { listInspections } from "../../lib/api";
import { useAsync } from "../../lib/useAsync";
import { InspectionForm } from "./InspectionForm";
import { InspectionEmptyState, InspectionList } from "./InspectionList";

export function InspectionsPage() {
  const [showForm, setShowForm] = useState(false);
  const navigate = useNavigate();
  const { state, reload } = useAsync(() => listInspections(), []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Inspections</h1>
        {!showForm && <Button onClick={() => setShowForm(true)}>New inspection</Button>}
      </div>

      {showForm && (
        <InspectionForm
          onCancel={() => setShowForm(false)}
          onCreated={(created) => navigate(`/inspections/${created.id}`)}
        />
      )}

      {state.status === "loading" && <Spinner label="Loading inspections…" />}
      {state.status === "error" && <ErrorState message={state.error.message} onRetry={reload} />}
      {state.status === "success" &&
        (state.data.items.length === 0 ? (
          !showForm && <InspectionEmptyState onCreate={() => setShowForm(true)} />
        ) : (
          <InspectionList items={state.data.items} />
        ))}
    </div>
  );
}
