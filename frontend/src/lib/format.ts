export const formatDateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

export const statusLabel = (status: string) => status.replace(/_/g, " ");
