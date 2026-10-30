import { useCallback, useEffect, useState } from "react";

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "success"; data: T };

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  const [tick, setTick] = useState(0);

  const run = useCallback(fn, deps);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    run().then(
      (data) => !cancelled && setState({ status: "success", data }),
      (error: Error) => !cancelled && setState({ status: "error", error }),
    );
    return () => {
      cancelled = true;
    };
  }, [run, tick]);

  return { state, reload: () => setTick((t) => t + 1) };
}
