/**
 * Request handling shared by the single-process router and the two tiers of a split deployment, so
 * every path validates input, selects monitoring samples, shapes results and delivers telemetry the
 * same way: a split deployment decides exactly as one process does.
 */
import { canonicalJson, now, sha256Hex, unitFromDigest, utf8, utf8Length } from './bytes.ts';
import type { AbstainReason, Candidate, ReviewReason } from './decision.ts';
import { RETRYABLE, type RuntimeErrorCode } from './errors.ts';
import { deliver, type DecisionEvent, type RouterEvent, type TelemetryCounters } from './events.ts';
import type { EncoderIdentity } from './identity.ts';
import type { RouterManifest } from './manifest.ts';

export type RouterMode = 'enforce' | 'shadow';

export interface RouteRequest {
  text: string;
  requestId: string;
  signal?: AbortSignal;
}

interface ResultBase {
  releaseId: string;
  requestId: string;
  embedded: boolean;
}

export type RouteResult = ResultBase & (
  | { mode: 'enforce'; actionable: true; outcome: 'route'; routeId: string; destination: string; mechanism: 'rule' | 'classifier' }
  | { mode: 'enforce'; actionable: false; outcome: 'review'; reason: ReviewReason }
  | { mode: 'enforce'; actionable: false; outcome: 'abstain'; reason: AbstainReason }
  | { mode: 'shadow'; actionable: false; outcome: 'shadow'; candidate: Candidate }
  | { mode: RouterMode; actionable: false; outcome: 'unavailable'; code: RuntimeErrorCode; retryable: boolean }
);

export interface EventContext {
  releaseId: string;
  requestId: string;
  mode: RouterMode;
  samplingProbability: number;
}

export const FORWARD_SCHEMA = 'liquidau-router-forward/1';

/**
 * What a rules tier sends its model tier (JSON-safe). 'decide': the rules couldn't settle the request,
 * so the model tier routes it. 'monitor': the rules tier answered it and selected it for monitoring;
 * the model tier only embeds and scores it for telemetry.
 */
export interface ForwardedRequest {
  schema: typeof FORWARD_SCHEMA;
  /** The release the rules tier served: the model tier refuses any other (RELEASE_MISMATCH). */
  manifestSha256: string;
  requestId: string;
  text: string;
  purpose: 'decide' | 'monitor';
  /** The rules tier's monitoring sample rate ('monitor'), recorded on the diagnostic event. */
  inclusionProbability: number;
}

/** Why a value is not a forwarded request (null when it is). */
export function forwardProblem(raw: unknown): string | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return 'not an object';
  const f = raw as Record<string, unknown>;
  for (const k of Object.keys(f)) if (!['schema', 'manifestSha256', 'requestId', 'text', 'purpose', 'inclusionProbability'].includes(k)) return `unknown field ${k}`;
  if (f.schema !== FORWARD_SCHEMA) return `schema must be '${FORWARD_SCHEMA}'`;
  if (typeof f.manifestSha256 !== 'string' || typeof f.requestId !== 'string' || typeof f.text !== 'string') return 'manifestSha256, requestId and text must be strings';
  if (f.purpose !== 'decide' && f.purpose !== 'monitor') return "purpose must be 'decide' or 'monitor'";
  if (!(typeof f.inclusionProbability === 'number' && f.inclusionProbability > 0 && f.inclusionProbability <= 1)) return 'inclusionProbability must be in (0, 1]';
  return null;
}

/** The request id as the router records it (a non-string id becomes its string form). */
export const requestIdOf = (request: RouteRequest | undefined): string => (typeof request?.requestId === 'string' ? request.requestId : String(request?.requestId ?? ''));

/**
 * Why a request can't be routed before any rule runs (null when it can): the text and id, the UTF-8
 * size limit, a legacy release's untruncated character limit, and cancellation.
 */
export function requestProblem(request: RouteRequest | undefined, requestId: string, manifest: RouterManifest, legacyIdentity: Readonly<EncoderIdentity> | null): 'INVALID_INPUT' | 'ABORTED' | null {
  const text = request?.text;
  if (typeof text !== 'string' || !requestId || requestId.length > 256) return 'INVALID_INPUT';
  if (utf8Length(text) > manifest.serving.maxInputUtf8Bytes) return 'INVALID_INPUT';
  // Legacy releases embed the whole text once; a document release applies maxChars per chunk instead.
  if (legacyIdentity && legacyIdentity.truncation === 'none' && legacyIdentity.maxChars !== null && text.length > legacyIdentity.maxChars) return 'INVALID_INPUT';
  if (request!.signal?.aborted) return 'ABORTED';
  return null;
}

/** Deterministic monitoring selection by release, salt and request id (never by text alone). */
export async function selectedForMonitoring(releaseId: string, salt: string, rate: number, requestId: string): Promise<boolean> {
  if (rate <= 0) return false;
  if (rate >= 1) return true;
  const digest = await sha256Hex(utf8(canonicalJson(['liquidau-router-sample/1', releaseId, salt, requestId])));
  return unitFromDigest(digest) < rate;
}

export function wrapCandidate(mode: RouterMode, releaseId: string, requestId: string, c: Candidate, embedded: boolean): RouteResult {
  const base = { releaseId, requestId, embedded };
  if (mode === 'shadow') return { ...base, mode: 'shadow', actionable: false, outcome: 'shadow', candidate: { ...c } };
  if (c.outcome === 'route') return { ...base, mode: 'enforce', actionable: true, outcome: 'route', routeId: c.routeId, destination: c.destination, mechanism: c.mechanism };
  if (c.outcome === 'review') return { ...base, mode: 'enforce', actionable: false, outcome: 'review', reason: c.reason };
  return { ...base, mode: 'enforce', actionable: false, outcome: 'abstain', reason: c.reason };
}

export function unavailableResult(mode: RouterMode, releaseId: string, requestId: string, code: RuntimeErrorCode, embedded: boolean, retryable = RETRYABLE[code]): RouteResult {
  return { releaseId, requestId, embedded, mode, actionable: false, outcome: 'unavailable', code, retryable };
}

/** The decision event for an unavailable result (nothing decided, nothing embedded unless `extra` says so). */
export function unavailableEvent(base: EventContext, code: RuntimeErrorCode, totalMs: number, extra: Partial<DecisionEvent> = {}): DecisionEvent {
  return {
    ...base, type: 'decision', outcome: 'unavailable', routeId: null, mechanism: null, reason: null, rulesSettled: false, sampled: false, embedded: false,
    inclusionProbability: 1, firedRule: null, dismissed: [], errorCode: code, timings: { totalMs, rulesMs: 0, embedMs: null, scoreMs: null }, ...extra,
  };
}

export function decisionEvent(base: EventContext, c: Candidate, rest: Omit<DecisionEvent, 'type' | 'outcome' | 'routeId' | 'mechanism' | 'reason' | keyof EventContext>): DecisionEvent {
  return {
    ...base, type: 'decision', outcome: c.outcome, routeId: c.outcome === 'route' ? c.routeId : null, mechanism: c.outcome === 'route' ? c.mechanism : null,
    reason: c.outcome === 'route' ? null : c.reason, ...rest,
  };
}

/**
 * Work that runs after a result is returned - observer deliveries and monitoring embeddings - tracked so
 * flush() and close() can wait for it. The observer is called in event order; waiting for an
 * asynchronous observer never delays a result.
 */
export class Background {
  readonly #observer: ((event: RouterEvent) => void | Promise<void>) | undefined;
  readonly #observerTimeoutMs: number;
  readonly #counters: TelemetryCounters = { delivered: 0, dropped: 0 };
  readonly #pending = new Set<Promise<void>>();

  constructor(observer: ((event: RouterEvent) => void | Promise<void>) | undefined, observerTimeoutMs: number) {
    this.#observer = observer;
    this.#observerTimeoutMs = observerTimeoutMs;
  }

  telemetry(): TelemetryCounters {
    return { ...this.#counters };
  }

  /** Runs `work` after the response; it must not reject (failures are swallowed). */
  run(work: Promise<void>): void {
    const tracked = work.catch(() => {}).finally(() => { this.#pending.delete(tracked); });
    this.#pending.add(tracked);
  }

  emit(event: RouterEvent): void {
    if (!this.#observer) return;
    this.run(deliver(this.#observer, event, this.#observerTimeoutMs).then((ok) => {
      if (ok === true) this.#counters.delivered++;
      else if (ok === false) this.#counters.dropped++;
    }));
  }

  async flush(): Promise<void> {
    // Settling work can start more (a monitoring sample emits its event), so repeat until none is left.
    while (this.#pending.size) await Promise.all(this.#pending);
  }
}

/** Counts in-flight requests so close() can wait for them to finish. */
export class Drain {
  #closed = false;
  #inflight = 0;
  #drained: (() => void) | null = null;

  get closed(): boolean {
    return this.#closed;
  }

  async track<T>(work: () => Promise<T>): Promise<T> {
    this.#inflight++;
    try {
      return await work();
    } finally {
      if (--this.#inflight === 0 && this.#drained) this.#drained();
    }
  }

  /** Refuses new work, then resolves once in-flight work finishes. */
  async close(): Promise<void> {
    this.#closed = true;
    if (this.#inflight > 0) await new Promise<void>((resolve) => { this.#drained = resolve; });
  }
}

export { now };
