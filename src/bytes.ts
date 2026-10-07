/**
 * Platform-neutral primitives: Web Crypto SHA-256, strict UTF-8 JSON decoding, deep freezing and a
 * monotonic clock (canonical JSON comes from text-preprocessing). Nothing here touches Node built-ins.
 */

// Canonical JSON and the SHA-256 hex check are text-preprocessing's, so digests agree across the libraries.
export { canonicalJson, isSha256Hex } from '@liquidau/text-preprocessing';

function subtle(): SubtleCrypto {
  const s = (globalThis as { crypto?: { subtle?: SubtleCrypto } }).crypto?.subtle;
  if (!s) throw new Error('Web Crypto SHA-256 (globalThis.crypto.subtle) is unavailable: supply a compatible implementation at the host boundary');
  return s;
}

/** True when the runtime provides what the router needs (Web Crypto, TextEncoder/TextDecoder, AbortController). */
export function platformProblems(): string[] {
  const g = globalThis as Record<string, unknown>;
  const problems: string[] = [];
  if (!(g.crypto as { subtle?: unknown } | undefined)?.subtle) problems.push('globalThis.crypto.subtle');
  for (const name of ['TextEncoder', 'TextDecoder', 'AbortController']) if (typeof g[name] !== 'function') problems.push(`globalThis.${name}`);
  return problems;
}

/** Lowercase hex SHA-256 of exact bytes. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await subtle().digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
  let out = '';
  for (const b of digest) out += b.toString(16).padStart(2, '0');
  return out;
}

const encoder = new TextEncoder();

export const utf8 = (text: string): Uint8Array => encoder.encode(text);

/** UTF-8 length as TextEncoder would produce it (a lone surrogate becomes U+FFFD, 3 bytes), without allocating. */
export function utf8Length(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0x80) n += 1;
    else if (c < 0x800) n += 2;
    else if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length && text.charCodeAt(i + 1) >= 0xdc00 && text.charCodeAt(i + 1) <= 0xdfff) { n += 4; i++; }
    else n += 3;
  }
  return n;
}

/** Parses exact bytes as UTF-8 JSON; invalid UTF-8 is an error, not replacement characters. */
export function decodeJson(bytes: Uint8Array): unknown {
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
}

/** Pretty JSON bytes for release documents (any stable serialisation works: files are hashed as bytes). */
export const encodeJson = (value: unknown): Uint8Array => utf8(`${JSON.stringify(value, null, 2)}\n`);

/** Freezes a parsed JSON value in place, recursively, and returns it. */
export function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
  }
  return value;
}

/** A parsed JSON copy: callers' objects are never retained. */
export const jsonClone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/** Milliseconds from a monotonic clock where available. */
export function now(): number {
  const p = (globalThis as { performance?: { now(): number } }).performance;
  return p ? p.now() : Date.now();
}

/** A uniform number in [0, 1) from the first 48 bits of a SHA-256 hex digest. */
export const unitFromDigest = (hex: string): number => Number.parseInt(hex.slice(0, 12), 16) / 2 ** 48;

/** The largest delay timers honour: Node and browsers clamp longer delays to about 1 ms. */
export const MAX_TIMER_MS = 2_147_483_647;

/** A usable timer delay in milliseconds: finite and within [1, MAX_TIMER_MS]. */
export const isTimerMs = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 1 && v <= MAX_TIMER_MS;
