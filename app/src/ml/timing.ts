/**
 * Tiny per stage timer. Each call to timed() adds the elapsed milliseconds to
 * a label. readPlate logs the totals once per photo, so the numbers show up in
 * adb logcat as one PLATE_TIMING line.
 */
const perf = (globalThis as { performance?: { now(): number } }).performance;
const now = (): number => (perf ? perf.now() : Date.now());

let totals: Record<string, number> = {};

export function resetTiming(): void {
  totals = {};
}

export function timed<T>(label: string, fn: () => T): T {
  const start = now();
  try {
    return fn();
  } finally {
    totals[label] = (totals[label] ?? 0) + (now() - start);
  }
}

export async function timedAsync<T>(
  label: string,
  fn: () => Promise<T>,
): Promise<T> {
  const start = now();
  try {
    return await fn();
  } finally {
    totals[label] = (totals[label] ?? 0) + (now() - start);
  }
}

/** Rounded copy of the current totals, in milliseconds. */
export function getTiming(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of Object.keys(totals)) {
    out[key] = Math.round(totals[key]);
  }
  return out;
}
