/**
 * The rules tier of a split deployment: the deterministic rules in their own small function (a Lambda
 * or an edge Worker), in front of a model tier that loads the same release with loadRouter.
 *
 * It loads only the manifest, rules, policy and evidence (no classifier, no encoder), so it starts in
 * milliseconds. Per request it runs exactly the single-process router's first steps - input checks, the
 * rules on the original text, conservative settlement - with the same shared code:
 *   - settled: it answers, emits the decision event, and when the request is selected for monitoring
 *     also returns a 'monitor' forward for the model tier to embed for telemetry only
 *   - not settled: it returns a 'decide' forward; the model tier's routeForwarded() decides it exactly
 *     as route() would in one process
 * Both tiers pin the same manifest digest; the model tier refuses a forward from any other release.
 */

import { createEdgeTier, resolveStaticEncoder, type EdgeTier, type StaticEncoderOption } from './edge-tier.ts';

import { deepFreeze, isTimerMs, MAX_TIMER_MS, now, platformProblems } from './bytes.ts';
import { createSettler, type RuleFindings, type Settler } from './decision.ts';
import { RETRYABLE, RouterLoadError, type RuntimeErrorCode } from './errors.ts';
import type { RouterEvent, TelemetryCounters } from './events.ts';
import { MANIFEST_SCHEMA_DOCUMENT } from './manifest.ts';
import { DEFAULT_LIMITS, readRulesRelease, rulesEnforcementProblems, type ReadLimits, type ReleaseSource } from './release.ts';
import type { ReviewOptions } from './review.ts';
import {
  answerReview, Background, decisionEvent, Drain, FORWARD_SCHEMA, makeReviewer, requestIdOf, requestProblem, reviewOptionProblems, selectedForMonitoring, unavailableEvent, unavailableResult, wrapCandidate,
  type EventContext, type ForwardedRequest, type Reviewer, type RouteRequest, type RouteResult, type RouterMode,
} from './serving.ts';

export interface LoadRulesTierOptions {
  source: ReleaseSource;
  manifestKey: string;
  /** The same digest the model tier pins. */
  expectedManifestSha256: string;
  /** Required, no default; use the model tier's mode. */
  mode: RouterMode;
  /** Deadline for each release read at load. */
  timeoutMs: number;
  /** Selection of rules-settled requests for monitoring: use the single-process router's settings. */
  monitoring: { sampleRate: number; samplingSalt: string };
  observer?: (event: RouterEvent) => void | Promise<void>;
  /** Budget for one asynchronous observer call before the event counts as dropped (default 250 ms). */
  observerTimeoutMs?: number;
  limits?: ReadLimits;
  /** Aborts loading. */
  signal?: AbortSignal;
  /**
   * The same review adapter as the model tier's: the rules alone can settle a request to review (every
   * head dismissed with onAbstain 'review'), and both tiers must answer such a review alike.
   */
  review?: ReviewOptions;
  /** Releases whose edge uses a static embedding: the same static encoder (or loader) the model tier is given. */
  staticEncoder?: StaticEncoderOption;
}

/**
 * The rules tier's answer, or the request to forward. `monitor` (when not null) goes to the model
 * tier's routeForwarded after the answer is returned; it never changes the answer.
 */
export type RulesTierOutcome =
  | { answered: true; result: RouteResult; monitor: ForwardedRequest | null }
  | { answered: false; forward: ForwardedRequest };

export interface RulesTier {
  readonly releaseId: string;
  readonly manifestSha256: string;
  readonly mode: RouterMode;
  handle(request: RouteRequest): Promise<RulesTierOutcome>;
  /** Resolves once observer deliveries started so far have finished. */
  flush(): Promise<void>;
  /** Refuses new requests (CLOSED) and resolves once in-flight requests and deliveries finish. Idempotent. */
  close(): Promise<void>;
  telemetry(): TelemetryCounters;
}

function optionProblems(o: LoadRulesTierOptions): string[] {
  const p: string[] = [];
  if (!o || typeof o !== 'object') return ['options must be an object'];
  if (!o.source || typeof o.source.read !== 'function') p.push('source must implement read(key, { signal })');
  if (typeof o.manifestKey !== 'string') p.push('manifestKey must be a string');
  if (o.mode !== 'enforce' && o.mode !== 'shadow') p.push("mode must be 'enforce' or 'shadow' (there is no default)");
  if (!isTimerMs(o.timeoutMs)) p.push(`timeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}] (longer delays overflow timers and fire at once)`);
  const m = o.monitoring;
  if (!m || typeof m !== 'object') p.push('monitoring must be { sampleRate, samplingSalt }');
  else {
    if (!(typeof m.sampleRate === 'number' && m.sampleRate >= 0 && m.sampleRate <= 1)) p.push('monitoring.sampleRate must be in [0, 1]');
    if (typeof m.samplingSalt !== 'string') p.push('monitoring.samplingSalt must be a string');
  }
  if (o.observer !== undefined && typeof o.observer !== 'function') p.push('observer must be a function');
  if (o.observerTimeoutMs !== undefined && !isTimerMs(o.observerTimeoutMs)) p.push(`observerTimeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}]`);
  for (const k of ['maxManifestBytes', 'maxFileBytes'] as const) {
    const v = o.limits?.[k];
    if (v !== undefined && !(Number.isInteger(v) && v > 0)) p.push(`limits.${k} must be a positive integer`);
  }
  p.push(...reviewOptionProblems(o.review));
  return p;
}

/** Validates the rules half of a release (throws RouterLoadError), then returns a ready rules tier. */
export async function loadRulesTier(options: LoadRulesTierOptions): Promise<RulesTier> {
  const platform = platformProblems();
  if (platform.length) throw new RouterLoadError('PLATFORM_UNSUPPORTED', 'the runtime lacks required Web platform primitives', platform);
  const problems = optionProblems(options);
  if (problems.length) throw new RouterLoadError('OPTIONS_INVALID', 'invalid loadRulesTier options', problems);
  const docs = await readRulesRelease(options.source, options.manifestKey, options.expectedManifestSha256, options.timeoutMs, { ...DEFAULT_LIMITS, ...options.limits }, options.signal);
  if (options.mode === 'enforce') {
    const why = rulesEnforcementProblems(docs);
    if (why.length) throw new RouterLoadError('GATES_FAILED', 'refusing to enforce this release (it may be served in shadow mode)', why);
  }
  deepFreeze(docs.evidence);
  const policy = deepFreeze(docs.policy);
  const reviewer = options.review ? await makeReviewer(options.review, options.monitoring.samplingSalt, policy) : null;
  const edgeTier = createEdgeTier(deepFreeze(docs.ruleSet), docs.edge ? deepFreeze(docs.edge) : null, policy.priority, await resolveStaticEncoder(docs.edge ?? null, options.staticEncoder));
  return new LoadedRulesTier(docs.manifestSha256, deepFreeze(docs.manifest), edgeTier, createSettler(policy), reviewer, options);
}

class LoadedRulesTier implements RulesTier {
  readonly releaseId: string;
  readonly manifestSha256: string;
  readonly mode: RouterMode;
  readonly #manifest: Parameters<typeof requestProblem>[2];
  readonly #edge: EdgeTier;
  readonly #settler: Settler;
  readonly #reviewer: Reviewer | null;
  readonly #sampleRate: number;
  readonly #salt: string;
  readonly #bg: Background;
  readonly #drain = new Drain();
  #closing: Promise<void> | null = null;

  constructor(manifestSha256: string, manifest: Parameters<typeof requestProblem>[2], edge: EdgeTier, settler: Settler, reviewer: Reviewer | null, o: LoadRulesTierOptions) {
    this.#reviewer = reviewer;
    this.releaseId = manifest.releaseId;
    this.manifestSha256 = manifestSha256;
    this.mode = o.mode;
    this.#manifest = manifest;
    this.#edge = edge;
    this.#settler = settler;
    this.#sampleRate = o.monitoring.sampleRate;
    this.#salt = o.monitoring.samplingSalt;
    this.#bg = new Background(o.observer, o.observerTimeoutMs ?? 250);
  }

  telemetry(): TelemetryCounters {
    return this.#bg.telemetry();
  }

  flush(): Promise<void> {
    return this.#bg.flush();
  }

  close(): Promise<void> {
    this.#closing ??= (async () => {
      await this.#drain.close();
      await this.#bg.flush();
    })();
    return this.#closing;
  }

  #base(requestId: string): EventContext {
    return { releaseId: this.releaseId, requestId, mode: this.mode, samplingProbability: this.#sampleRate };
  }

  #forward(text: string, context: string | undefined, requestId: string, purpose: ForwardedRequest['purpose']): ForwardedRequest {
    return { schema: FORWARD_SCHEMA, manifestSha256: this.manifestSha256, requestId, text, ...(context !== undefined ? { context } : {}), purpose, inclusionProbability: purpose === 'monitor' ? this.#sampleRate : 1 };
  }

  #unavailable(requestId: string, code: RuntimeErrorCode, totalMs: number): RulesTierOutcome {
    this.#bg.emit(unavailableEvent(this.#base(requestId), code, totalMs));
    return { answered: true, result: unavailableResult(this.mode, this.releaseId, requestId, code, false, RETRYABLE[code]), monitor: null };
  }

  async handle(request: RouteRequest): Promise<RulesTierOutcome> {
    const requestId = requestIdOf(request);
    if (this.#drain.closed) return this.#unavailable(requestId, 'CLOSED', 0);
    return this.#drain.track(async () => {
      const t0 = now();
      // The legacy character check needs the release's encoder identity, which the manifest records.
      const legacy = this.#manifest.schema === MANIFEST_SCHEMA_DOCUMENT ? null : this.#manifest.embedding;
      const problem = requestProblem(request, requestId, this.#manifest, legacy);
      if (problem) return this.#unavailable(requestId, problem, now() - t0);
      const text = request.text;
      const tr = now();
      let found: ReturnType<EdgeTier['evaluate']>;
      try {
        found = this.#edge.evaluate(text);
      } catch {
        return this.#unavailable(requestId, 'PREPROCESSING_FAILURE', now() - t0);
      }
      const rules: RuleFindings = found.findings;
      const settled = this.#settler.settle(rules);
      const rulesMs = now() - tr;
      if (!settled) return { answered: false, forward: this.#forward(text, request.context, requestId, 'decide') };
      const sampled = await selectedForMonitoring(this.releaseId, this.#salt, this.#sampleRate, requestId);
      let final = settled;
      let review: Awaited<ReturnType<typeof answerReview>> | null = null;
      if (settled.outcome === 'review' && this.#reviewer) {
        review = await answerReview(this.#reviewer, settled, { text, fired: rules.fired, dismissed: rules.dismissed, probabilities: {}, releaseId: this.releaseId, requestId, ...(request.signal ? { signal: request.signal } : {}) });
        final = review.candidate;
      }
      this.#bg.emit(decisionEvent(this.#base(requestId), final, {
        firedRule: rules.fired?.id ?? null, dismissed: rules.dismissed, ...(found.edge ? { edge: found.edge } : {}), rulesSettled: true, settledBy: 'rules', sampled, embedded: false, inclusionProbability: this.#sampleRate,
        errorCode: null, timings: { totalMs: now() - t0, rulesMs, embedMs: null, scoreMs: null, ...(review ? { reviewMs: review.reviewMs } : {}) }, ...(review ? { review: review.review } : {}),
      }));
      return { answered: true, result: wrapCandidate(this.mode, this.releaseId, requestId, final, false), monitor: sampled ? this.#forward(text, request.context, requestId, 'monitor') : null };
    });
  }
}
