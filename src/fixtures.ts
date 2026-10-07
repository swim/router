/**
 * In-memory host pieces that need no filesystem, network or credentials: a byte-map release source and
 * a function-backed encoder. Enough to load and route a complete release in a test, a browser or an
 * edge function with bundled release bytes.
 */
import { EncoderError } from './errors.ts';
import type { Encoder, EncoderIdentity } from './identity.ts';
import { isSafeKey } from './manifest.ts';
import type { ReleaseSource } from './release.ts';

/** A release source over a byte map. Keys outside the map (or unsafe keys) reject; bytes are copied out. */
export function memorySource(files: ReadonlyMap<string, Uint8Array> | Readonly<Record<string, Uint8Array>>, options: { maxBytes?: number } = {}): ReleaseSource {
  const map = new Map<string, Uint8Array>(files instanceof Map ? files : Object.entries(files));
  return {
    async read(key, { signal }) {
      if (signal.aborted) throw new Error('aborted');
      if (!isSafeKey(key)) throw new Error(`unsafe key ${JSON.stringify(key)}`);
      const bytes = map.get(key);
      if (!bytes) throw new Error(`no such key ${key}`);
      if (options.maxBytes !== undefined && bytes.byteLength > options.maxBytes) throw new Error(`${key} exceeds ${options.maxBytes} bytes`);
      return bytes.slice();
    },
  };
}

/**
 * An encoder from a synchronous or asynchronous per-text function: for deterministic fakes, or wrapping
 * a local model. Honours the abort signal; a thrown error becomes ENCODER_FAILURE.
 */
export function functionEncoder(identity: EncoderIdentity, embedOne: (text: string, signal: AbortSignal) => readonly number[] | Promise<readonly number[]>): Encoder {
  const frozen = Object.freeze({ ...identity, layers: identity.layers ? Object.freeze([...identity.layers]) as number[] : null });
  return {
    identity: frozen,
    async embed(texts, { signal }) {
      const out: (readonly number[])[] = [];
      for (const t of texts) {
        if (signal.aborted) throw new EncoderError('ENCODER_FAILURE', 'aborted', { retryable: false });
        out.push(await embedOne(t, signal));
      }
      return out;
    },
  };
}
