/**
 * The review adapter: an injected, provider-neutral answerer for requests the router sends to review
 * (an LLM behind any provider - a direct API, Bedrock, Vertex - or a human queue with a synchronous
 * front). It mirrors the encoder contract: an identity, one call with a signal, a conformance checker.
 *
 * The answer is ADVISORY. It never changes the router's decision (a review stays a review, actionable
 * false), never reaches a threshold, rule or calibration, and the adapter never sees the request id or
 * the release digest: ReviewInput is the whole interface. A deterministic sample of reviews is marked
 * for a person (forHuman), whose labels estimate the adapter's accuracy per route.
 */
import { canonicalJson, isSha256Hex, sha256Hex, utf8 } from './bytes.ts';
import type { ReviewReason } from './decision.ts';

export const REVIEW_ADAPTER_SCHEMA = 'liquidau-review-adapter/1';

export interface ReviewAdapterIdentity {
  schema: typeof REVIEW_ADAPTER_SCHEMA;
  /** Who serves the model, e.g. 'anthropic', 'bedrock', 'vertex', 'self-hosted'. */
  provider: string;
  model: string;
  revision?: string;
  /** SHA-256 (lowercase hex) of the prompt template the adapter sends. */
  promptSha256: string;
  /** SHA-256 (lowercase hex) of the JSON schema the adapter constrains the answer to. */
  outputSchemaSha256: string;
}

export interface ReviewInput {
  text: string;
  /** The policy's routes: what the answer may name. */
  routes: ReadonlyArray<{ id: string; description?: string; examples?: readonly string[] }>;
  evidence: {
    reason: ReviewReason;
    fired: { id: string; label: string } | null;
    dismissed: readonly string[];
    /** Calibrated probabilities, when the classifier scored the request (empty when rules settled it). */
    probabilities: Readonly<Record<string, number>>;
  };
}

export interface ReviewAnswer {
  /** One of the routes' ids, or null: none fits. */
  routeId: string | null;
  confidence: number | null;
  /** Free text for the host; never recorded in telemetry. */
  rationale?: string;
}

export interface ReviewAdapter {
  readonly identity: ReviewAdapterIdentity;
  review(input: ReviewInput, options: { signal: AbortSignal }): Promise<ReviewAnswer>;
}

export type ReviewErrorCode = 'TIMEOUT' | 'ABORTED' | 'REFUSED' | 'FAILURE' | 'INVALID_ANSWER';

/** What a review result (or a shadow candidate that is a review) carries when an adapter is configured. */
export interface ReviewAnswerFields {
  /** The adapter's answer: advisory, never actionable by itself. */
  answer?: { routeId: string | null; destination: string | null; confidence: number | null; adapterSha256: string };
  /** The adapter failed, refused or timed out: the review stands without an answer. */
  answerError?: { code: ReviewErrorCode; retryable: boolean };
  /**
   * Send to a person: every review without an answer, and a deterministic sample of answered ones
   * (humanSampleRate) whose labels estimate the adapter's accuracy.
   */
  forHuman?: boolean;
}

/** The review option of loadRouter and loadRulesTier. */
export interface ReviewOptions {
  adapter: ReviewAdapter;
  /** Deadline for one adapter call, separate from the encoder's. */
  timeoutMs: number;
  /** Share of answered reviews also sent to a person, in [0, 1]. */
  humanSampleRate: number;
  /** Salt for that sample (default: the monitoring salt). */
  samplingSalt?: string;
}

/**
 * Adapters throw this for failures they can name: REFUSED (the provider's safety layer declined; not
 * retryable), FAILURE (transport or provider error; retryable by default). The router maps anything
 * else to FAILURE, its own deadline to TIMEOUT, the caller's abort to ABORTED, and an answer outside
 * the schema to INVALID_ANSWER.
 */
export class ReviewAdapterError extends Error {
  override readonly name = 'ReviewAdapterError';
  readonly code: ReviewErrorCode;
  readonly retryable: boolean;
  constructor(code: ReviewErrorCode, message: string = code, retryable = code === 'FAILURE' || code === 'TIMEOUT') {
    super(message);
    this.code = code;
    this.retryable = retryable;
  }
}

const IDENTITY_FIELDS = ['schema', 'provider', 'model', 'revision', 'promptSha256', 'outputSchemaSha256'] as const;

/** Problems with a review adapter identity (empty when valid). Unknown fields are refused. */
export function reviewAdapterIdentityProblems(identity: unknown): string[] {
  if (!identity || typeof identity !== 'object' || Array.isArray(identity)) return ['identity must be an object'];
  const id = identity as Record<string, unknown>;
  const p: string[] = [];
  for (const k of Object.keys(id)) if (!(IDENTITY_FIELDS as readonly string[]).includes(k)) p.push(`unknown field ${k}`);
  if (id.schema !== REVIEW_ADAPTER_SCHEMA) p.push(`schema must be '${REVIEW_ADAPTER_SCHEMA}'`);
  for (const k of ['provider', 'model'] as const) if (typeof id[k] !== 'string' || !id[k]) p.push(`${k} must be a non-empty string`);
  if (id.revision !== undefined && (typeof id.revision !== 'string' || !id.revision)) p.push('revision must be a non-empty string when present');
  for (const k of ['promptSha256', 'outputSchemaSha256'] as const) if (!isSha256Hex(id[k])) p.push(`${k} must be lowercase hex SHA-256`);
  return p;
}

/** SHA-256 of the identity's canonical JSON: what results and telemetry record as adapterSha256. */
export async function reviewAdapterIdentityDigest(identity: ReviewAdapterIdentity): Promise<string> {
  return sha256Hex(utf8(canonicalJson(Object.fromEntries(IDENTITY_FIELDS.filter((k) => identity[k] !== undefined).map((k) => [k, identity[k]])))));
}

/** Why an answer is outside the schema for these routes (null when it is valid). */
export function reviewAnswerProblem(answer: unknown, routes: ReviewInput['routes']): string | null {
  if (!answer || typeof answer !== 'object' || Array.isArray(answer)) return 'the answer is not an object';
  const a = answer as Record<string, unknown>;
  for (const k of Object.keys(a)) if (!['routeId', 'confidence', 'rationale'].includes(k)) return `unknown field ${k}`;
  if (a.routeId !== null && !(typeof a.routeId === 'string' && routes.some((r) => r.id === a.routeId))) return `routeId ${JSON.stringify(a.routeId)} is not one of the routes`;
  if (a.confidence !== null && !(typeof a.confidence === 'number' && a.confidence >= 0 && a.confidence <= 1)) return `confidence ${JSON.stringify(a.confidence)} must be in [0, 1] or null`;
  if (a.rationale !== undefined && typeof a.rationale !== 'string') return 'rationale must be a string';
  return null;
}

/**
 * Calls the adapter under its own deadline and the caller's signal. Never throws: a failure is
 * returned as an error code, so a review whose adapter fails is still a valid review.
 */
export async function callReviewAdapter(adapter: ReviewAdapter, input: ReviewInput, timeoutMs: number, outer?: AbortSignal): Promise<{ answer: ReviewAnswer } | { error: { code: ReviewErrorCode; retryable: boolean } }> {
  if (outer?.aborted) return { error: { code: 'ABORTED', retryable: false } };
  const controller = new AbortController();
  let why: 'TIMEOUT' | 'ABORTED' | null = null;
  let rejectStop!: (e: unknown) => void;
  const stopped = new Promise<never>((_, reject) => { rejectStop = reject; });
  stopped.catch(() => {});
  const stop = (code: 'TIMEOUT' | 'ABORTED') => { if (why) return; why = code; controller.abort(); rejectStop(new ReviewAdapterError(code)); };
  const timer = setTimeout(() => stop('TIMEOUT'), timeoutMs);
  const onAbort = () => stop('ABORTED');
  outer?.addEventListener('abort', onAbort, { once: true });
  try {
    const call = Promise.resolve().then(() => adapter.review(input, { signal: controller.signal }));
    call.catch(() => {});
    const answer = await Promise.race([call, stopped]);
    const problem = reviewAnswerProblem(answer, input.routes);
    if (problem) return { error: { code: 'INVALID_ANSWER', retryable: false } };
    const a = answer as ReviewAnswer;
    return { answer: { routeId: a.routeId, confidence: a.confidence, ...(a.rationale !== undefined ? { rationale: a.rationale } : {}) } };
  } catch (e) {
    if (why) return { error: { code: why, retryable: why === 'TIMEOUT' } };
    if (e instanceof ReviewAdapterError) return { error: { code: e.code, retryable: e.retryable } };
    return { error: { code: 'FAILURE', retryable: true } };
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener('abort', onAbort);
  }
}
