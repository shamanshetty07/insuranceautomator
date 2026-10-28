import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { InspectionDetailPage } from "./features/inspections/InspectionDetailPage";
import { InspectionsPage } from "./features/inspections/InspectionsPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/inspections" replace />} />
        <Route path="/inspections" element={<InspectionsPage />} />
        <Route path="/inspections/:inspectionId" element={<InspectionDetailPage />} />
        <Route path="*" element={<p>Page not found.</p>} />
      </Route>
    </Routes>
  );
}
