/**
 * Shared adapter contract checks, independent of any test runner: an adapter package runs these against
 * its source or encoder (with provider mocks, no credentials) and fails its suite on any problem.
 */
import { isTimerMs, MAX_TIMER_MS } from './bytes.ts';
import { encoderIdentityProblems, vectorProblems, type Encoder } from './identity.ts';
import type { ReleaseSource } from './release.ts';

const withTimeout = async <T>(p: Promise<T>, ms: number): Promise<T | 'timeout'> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([p, new Promise<'timeout'>((resolve) => { timer = setTimeout(() => resolve('timeout'), ms); })]);
  } finally {
    clearTimeout(timer);
  }
};

/**
 * Problems with an encoder adapter (empty when it conforms): a valid identity; one finite vector per
 * text, of the declared width and normalisation, in order; deterministic output; and an already-aborted
 * signal makes embed reject promptly rather than return.
 */
export async function encoderContractProblems(encoder: Encoder, options: { texts?: readonly string[]; timeoutMs?: number; tolerance?: number } = {}): Promise<string[]> {
  const texts = options.texts ?? ['refund my order', 'where is my card?', 'ünïcödé text 🙂'];
  const timeoutMs = options.timeoutMs ?? 10_000;
  const tolerance = options.tolerance ?? 1e-6;
  if (!isTimerMs(timeoutMs)) return [`timeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}]`];
  const problems = encoderIdentityProblems(encoder.identity).map((p) => `identity: ${p}`);
  if (problems.length) return problems;
  const run = async (batch: readonly string[], signal = new AbortController().signal) => withTimeout(Promise.resolve().then(() => encoder.embed(batch, { signal })), timeoutMs);
  let rows: Awaited<ReturnType<typeof run>>;
  try {
    rows = await run(texts);
  } catch (e) {
    return [`embed rejected: ${(e as Error).message}`];
  }
  if (rows === 'timeout') return [`embed did not finish within ${timeoutMs} ms`];
  if (!Array.isArray(rows) || rows.length !== texts.length) return [`embed returned ${Array.isArray(rows) ? rows.length : typeof rows} rows for ${texts.length} texts`];
  rows.forEach((r, i) => { const why = vectorProblems(encoder.identity, r); if (why) problems.push(`row ${i}: ${why}`); });
  try {
    const single = await run([texts[texts.length - 1]]);
    const again = await run(texts);
    if (single === 'timeout' || again === 'timeout') problems.push('a repeated embed timed out');
    else {
      const far = (a: readonly number[], b: readonly number[]) => a.length !== b.length || a.some((v, k) => Math.abs(v - b[k]) > tolerance);
      if (far(single[0], rows[rows.length - 1])) problems.push('a text embedded alone differs from the same text in a batch');
      if (again.some((r, i) => far(r, rows[i]))) problems.push('embedding the same texts twice gave different vectors');
    }
  } catch (e) {
    problems.push(`a repeated embed rejected: ${(e as Error).message}`);
  }
  const aborted = new AbortController();
  aborted.abort();
  try {
    const r = await run(texts, aborted.signal);
    problems.push(r === 'timeout' ? 'embed with an aborted signal neither rejected nor returned' : 'embed with an already-aborted signal returned vectors instead of rejecting');
  } catch {
    // Expected: cancellation rejects.
  }
  return problems;
}

/**
 * Problems with a release source adapter: each present key returns exactly its bytes, a missing key
 * rejects, and keys that would escape the source prefix reject.
 */
export async function releaseSourceContractProblems(source: ReleaseSource, options: { present: Readonly<Record<string, Uint8Array>>; missingKey?: string; timeoutMs?: number }): Promise<string[]> {
  const problems: string[] = [];
  const timeoutMs = options.timeoutMs ?? 10_000;
  if (!isTimerMs(timeoutMs)) return [`timeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}]`];
  const read = (key: string) => withTimeout(Promise.resolve().then(() => source.read(key, { signal: new AbortController().signal })), timeoutMs);
  for (const [key, expected] of Object.entries(options.present)) {
    try {
      const got = await read(key);
      if (got === 'timeout') problems.push(`${key}: read timed out`);
      else if (!(got instanceof Uint8Array) || got.byteLength !== expected.byteLength || got.some((b, i) => b !== expected[i])) problems.push(`${key}: returned bytes differ`);
    } catch (e) {
      problems.push(`${key}: read rejected: ${(e as Error).message}`);
    }
  }
  for (const key of [options.missingKey ?? 'definitely-missing.json', '../escape.json', '/absolute.json', 'a/../../escape.json']) {
    try {
      const got = await read(key);
      problems.push(got === 'timeout' ? `${key}: read neither rejected nor returned` : `${key}: read returned bytes instead of rejecting`);
    } catch {
      // Expected.
    }
  }
  return problems;
}
