import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 md:flex">
      <aside className="border-b border-slate-200 bg-white md:w-60 md:border-r md:border-b-0">
        <div className="px-5 py-4">
          <p className="text-base font-semibold">Vehicle Damage AI</p>
          <p className="text-xs text-slate-500">Inspection &amp; report verification</p>
        </div>
        <nav className="flex gap-1 px-3 pb-3 md:flex-col" aria-label="Main">
          <NavLink
            to="/inspections"
            className={({ isActive }) =>
              `rounded-md px-3 py-2 text-sm font-medium ${isActive ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"}`
            }
          >
            Inspections
          </NavLink>
        </nav>
      </aside>
      <main className="mx-auto w-full max-w-4xl flex-1 p-4 md:p-8">
        <Outlet />
      </main>
    </div>
  );
}
