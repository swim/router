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
import type { RouterPolicy } from './policy.ts';
import { callReviewAdapter, reviewAdapterIdentityDigest, reviewAdapterIdentityProblems, type ReviewAnswerFields, type ReviewInput, type ReviewOptions } from './review.ts';

export type RouterMode = 'enforce' | 'shadow';

export interface RouteRequest {
  text: string;
  requestId: string;
  /**
   * Releases with context features (manifest `contextEmbedding`) only, and required by them: the text
   * the classifier's context features were trained on, such as the conversation's next turn or an
   * assistant's reply, embedded by the release's context encoder. Pass it exactly as training did,
   * including whatever stood for "no context". Rules and the edge read `text` only. Releases without
   * context features refuse it (INVALID_INPUT), so a host never believes context is used when it isn't.
   */
  context?: string;
  signal?: AbortSignal;
}

interface ResultBase {
  releaseId: string;
  requestId: string;
  embedded: boolean;
}

export type RouteResult = ResultBase & (
  | { mode: 'enforce'; actionable: true; outcome: 'route'; routeId: string; destination: string; mechanism: 'rule' | 'edge' | 'middle' | 'classifier' }
  | ({ mode: 'enforce'; actionable: false; outcome: 'review'; reason: ReviewReason } & ReviewAnswerFields)
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
  /** The request's context (releases with context features). */
  context?: string;
  purpose: 'decide' | 'monitor';
  /** The rules tier's monitoring sample rate ('monitor'), recorded on the diagnostic event. */
  inclusionProbability: number;
}

/** Why a value is not a forwarded request (null when it is). */
export function forwardProblem(raw: unknown): string | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return 'not an object';
  const f = raw as Record<string, unknown>;
  for (const k of Object.keys(f)) if (!['schema', 'manifestSha256', 'requestId', 'text', 'context', 'purpose', 'inclusionProbability'].includes(k)) return `unknown field ${k}`;
  if (f.context !== undefined && typeof f.context !== 'string') return 'context must be a string'
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
 * size limit, a legacy release's untruncated character limit, the context (present exactly when the
 * release has context features, within the same limits), and cancellation.
 */
export function requestProblem(request: RouteRequest | undefined, requestId: string, manifest: RouterManifest, legacyIdentity: Readonly<EncoderIdentity> | null): 'INVALID_INPUT' | 'ABORTED' | null {
  const text = request?.text;
  if (typeof text !== 'string' || !requestId || requestId.length > 256) return 'INVALID_INPUT';
  if (utf8Length(text) > manifest.serving.maxInputUtf8Bytes) return 'INVALID_INPUT';
  // Legacy releases embed the whole text once; a document release applies maxChars per chunk instead.
  if (legacyIdentity && legacyIdentity.truncation === 'none' && legacyIdentity.maxChars !== null && text.length > legacyIdentity.maxChars) return 'INVALID_INPUT';
  const context = request!.context, contextIdentity = manifest.contextEmbedding;
  if ((context !== undefined) !== !!contextIdentity) return 'INVALID_INPUT';
  if (context !== undefined) {
    if (typeof context !== 'string' || utf8Length(context) > manifest.serving.maxInputUtf8Bytes) return 'INVALID_INPUT';
    if (contextIdentity!.truncation === 'none' && contextIdentity!.maxChars !== null && context.length > contextIdentity!.maxChars) return 'INVALID_INPUT';
  }
  if (request!.signal?.aborted) return 'ABORTED';
  return null;
}

/** Deterministic monitoring selection by release, salt and request id (never by text alone). */
export async function selectedForMonitoring(releaseId: string, salt: string, rate: number, requestId: string): Promise<boolean> {
  return selected('liquidau-router-sample/1', releaseId, salt, rate, requestId);
}

/** The human-review sample of answered reviews: its own domain tag, so it is independent of monitoring. */
export async function selectedForHumanReview(releaseId: string, salt: string, rate: number, requestId: string): Promise<boolean> {
  return selected('liquidau-router-review-sample/1', releaseId, salt, rate, requestId);
}

async function selected(tag: string, releaseId: string, salt: string, rate: number, requestId: string): Promise<boolean> {
  if (rate <= 0) return false;
  if (rate >= 1) return true;
  const digest = await sha256Hex(utf8(canonicalJson([tag, releaseId, salt, requestId])));
  return unitFromDigest(digest) < rate;
}

/** Problems with a review option (empty when valid). */
export function reviewOptionProblems(review: unknown): string[] {
  if (review === undefined) return [];
  const r = review as Partial<ReviewOptions>;
  if (!r || typeof r !== 'object') return ['review must be { adapter, timeoutMs, humanSampleRate }'];
  const p: string[] = [];
  if (!r.adapter || typeof r.adapter.review !== 'function') p.push('review.adapter must implement review(input, { signal })');
  else p.push(...reviewAdapterIdentityProblems(r.adapter.identity).map((e) => `review.adapter identity ${e}`));
  if (!(typeof r.timeoutMs === 'number' && Number.isFinite(r.timeoutMs) && r.timeoutMs >= 1 && r.timeoutMs <= 2_147_483_647)) p.push('review.timeoutMs must be milliseconds in [1, 2147483647]');
  if (!(typeof r.humanSampleRate === 'number' && r.humanSampleRate >= 0 && r.humanSampleRate <= 1)) p.push('review.humanSampleRate must be in [0, 1]');
  if (r.samplingSalt !== undefined && typeof r.samplingSalt !== 'string') p.push('review.samplingSalt must be a string');
  return p;
}

/** A loaded review option: the adapter, its digest, and the policy's routes the answer may name. */
export interface Reviewer {
  options: ReviewOptions;
  salt: string;
  adapterSha256: string;
  routes: ReviewInput['routes'];
  destination: ReadonlyMap<string, string>;
}

export async function makeReviewer(options: ReviewOptions, monitoringSalt: string, policy: RouterPolicy): Promise<Reviewer> {
  return {
    options, salt: options.samplingSalt ?? monitoringSalt, adapterSha256: await reviewAdapterIdentityDigest(options.adapter.identity),
    routes: policy.routes.map((r) => ({ id: r.id, ...(r.description !== undefined ? { description: r.description } : {}), ...(r.examples ? { examples: [...r.examples] } : {}) })),
    destination: new Map(policy.routes.map((r) => [r.id, r.destination])),
  };
}

/**
 * Asks the reviewer about a review candidate and returns the candidate with the answer attached, plus
 * the decision event's review record. The decision itself never changes; failures are recorded.
 */
export async function answerReview(
  reviewer: Reviewer,
  c: Extract<Candidate, { outcome: 'review' }>,
  ctx: { text: string; fired: { id: string; label: string } | null; dismissed: readonly string[]; probabilities: Readonly<Record<string, number>>; releaseId: string; requestId: string; signal?: AbortSignal },
): Promise<{ candidate: Candidate; review: NonNullable<DecisionEvent['review']>; reviewMs: number }> {
  const t = now();
  const input: ReviewInput = { text: ctx.text, routes: reviewer.routes, evidence: { reason: c.reason, fired: ctx.fired, dismissed: [...ctx.dismissed], probabilities: { ...ctx.probabilities } } };
  const out = await callReviewAdapter(reviewer.options.adapter, input, reviewer.options.timeoutMs, ctx.signal);
  const reviewMs = now() - t;
  if ('answer' in out) {
    const forHuman = await selectedForHumanReview(ctx.releaseId, reviewer.salt, reviewer.options.humanSampleRate, ctx.requestId);
    const answer = { routeId: out.answer.routeId, destination: out.answer.routeId === null ? null : reviewer.destination.get(out.answer.routeId) ?? null, confidence: out.answer.confidence, adapterSha256: reviewer.adapterSha256 };
    return {
      candidate: { ...c, answer, forHuman },
      review: { answered: true, adapterSha256: reviewer.adapterSha256, routeId: answer.routeId, confidence: answer.confidence, errorCode: null, forHuman, reviewMs },
      reviewMs,
    };
  }
  return {
    candidate: { ...c, answerError: out.error, forHuman: true },
    review: { answered: false, adapterSha256: reviewer.adapterSha256, routeId: null, confidence: null, errorCode: out.error.code, forHuman: true, reviewMs },
    reviewMs,
  };
}

export function wrapCandidate(mode: RouterMode, releaseId: string, requestId: string, c: Candidate, embedded: boolean): RouteResult {
  const base = { releaseId, requestId, embedded };
  if (mode === 'shadow') return { ...base, mode: 'shadow', actionable: false, outcome: 'shadow', candidate: JSON.parse(JSON.stringify(c)) as Candidate };
  if (c.outcome === 'route') return { ...base, mode: 'enforce', actionable: true, outcome: 'route', routeId: c.routeId, destination: c.destination, mechanism: c.mechanism };
  if (c.outcome === 'review') {
    const extra = { ...(c.answer ? { answer: { ...c.answer } } : {}), ...(c.answerError ? { answerError: { ...c.answerError } } : {}), ...(c.forHuman !== undefined ? { forHuman: c.forHuman } : {}) };
    return { ...base, mode: 'enforce', actionable: false, outcome: 'review', reason: c.reason, ...extra };
  }
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
