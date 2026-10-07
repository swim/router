/**
 * loadRouter: validate a complete release once, then route requests against that immutable snapshot.
 *
 * Per request:
 *   1. validate the input (type, UTF-8 size, untruncated length) and cancellation
 *   2. run the validated rule matcher on the ORIGINAL text
 *   3. settle with rules alone when no score could change the outcome
 *   4. otherwise embed within the deadline, validate the vector, score, validate the probabilities
 *   5-7. the decision evaluator (priority, suppression, review band, onAbstain)
 *   8. wrap for enforce or shadow mode and emit one decision event
 * A deterministic sample of rules-settled requests is also embedded for monitoring; that never changes
 * their decision, and its failure is a diagnostic, not `unavailable`. Monitoring and observer delivery
 * run after the result is returned: `flush()` (and `close()`) wait for them.
 */
import { DocumentError, prepareScoring, scoreDocument, scoreEmbedding, type ClassifierArtifact, type DocumentAdapters, type DocumentClassifierArtifact, type Scores } from '@liquidau/embedding-classifier';
import { ruleSetMatcher } from '@liquidau/rule-miner';

import { canonicalJson, deepFreeze, isTimerMs, MAX_TIMER_MS, now, platformProblems } from './bytes.ts';
import { createDecisionEvaluator, ScoreError, type Candidate, type DecisionEvaluator, type RuleFindings } from './decision.ts';
import { EncoderError, RETRYABLE, RouterLoadError, type RuntimeErrorCode } from './errors.ts';
import type { DecisionEvent, DocumentTelemetry, RouterEvent, TelemetryCounters } from './events.ts';
import { encoderIdentityDiff, encoderIdentityProblems, vectorProblems, type Encoder, type EncoderIdentity } from './identity.ts';
import type { RouterManifest } from './manifest.ts';
import { DEFAULT_LIMITS, enforcementProblems, readRelease, type ReadLimits, type ReleaseSource } from './release.ts';
import {
  Background, decisionEvent, Drain, forwardProblem, requestIdOf, requestProblem, selectedForMonitoring, unavailableEvent, unavailableResult, wrapCandidate,
  type EventContext, type ForwardedRequest, type RouteRequest, type RouteResult, type RouterMode,
} from './serving.ts';

export type { ForwardedRequest, RouteRequest, RouteResult, RouterMode };

export interface LoadRouterOptions {
  source: ReleaseSource;
  manifestKey: string;
  /** From trusted deployment configuration: the release's identity. */
  expectedManifestSha256: string;
  /** Required, no default. */
  mode: RouterMode;
  encoder: Encoder;
  /** Deadline for each release read at load and for each request's embedding. */
  timeoutMs: number;
  monitoring: {
    /** Share of rules-settled requests embedded for monitoring, in [0, 1]. */
    sampleRate: number;
    samplingSalt: string;
    /** Deadline for a monitoring embedding (default timeoutMs). */
    timeoutMs?: number;
  };
  observer?: (event: RouterEvent) => void | Promise<void>;
  /** Budget for one asynchronous observer call before the event counts as dropped (default 250 ms). */
  observerTimeoutMs?: number;
  limits?: ReadLimits;
  /** Aborts loading. */
  signal?: AbortSignal;
  /**
   * Document releases ('liquidau-router/2') only: the exact tokenizer of the release pipeline. Legacy
   * releases don't need it. With a document release, `timeoutMs` is ONE deadline for the whole document
   * (planning, every embedding batch and scoring), not a fresh timeout per chunk.
   */
  tokenizer?: DocumentAdapters['tokenizer'];
  /** Semantic document releases ('adjacent-cosine/1'): the pipeline's boundary encoder and tokenizer. */
  boundaryEncoder?: Encoder;
  boundaryTokenizer?: DocumentAdapters['tokenizer'];
  /**
   * Document releases: chunks per encoder call, calls in flight, and how often planning yields to the event
   * loop. Operational only: never changes outputs. The deadline is checked at every planning step anyway.
   */
  document?: { batchSize?: number; concurrency?: number; yieldEvery?: number };
}

export interface Router {
  readonly releaseId: string;
  readonly manifestSha256: string;
  readonly mode: RouterMode;
  route(request: RouteRequest): Promise<RouteResult>;
  /**
   * The model tier of a split deployment (see loadRulesTier): a request its rules tier forwarded.
   * 'decide' is routed exactly as route() would; 'monitor' (a request the rules tier answered and
   * selected) is embedded and scored for telemetry only, in the background, and resolves to null. A
   * request forwarded by a tier serving another release is refused (RELEASE_MISMATCH).
   */
  routeForwarded(forward: ForwardedRequest, options?: { signal?: AbortSignal }): Promise<RouteResult | null>;
  /**
   * Resolves once background work started so far has finished: monitoring embeddings and observer
   * deliveries, which run after `route()` returns. Function hosts await it before the invocation ends
   * (or hand it to the platform, e.g. `ctx.waitUntil(router.flush())`), or that work may never run.
   */
  flush(): Promise<void>;
  /** Refuses new requests (CLOSED) and resolves once in-flight requests and background work finish. Idempotent. */
  close(): Promise<void>;
  /** Observer delivery counters, for host metrics. */
  telemetry(): TelemetryCounters;
}

class RuntimeFailure extends Error {
  readonly code: RuntimeErrorCode;
  readonly retryable: boolean;
  constructor(code: RuntimeErrorCode, retryable: boolean = RETRYABLE[code]) {
    super(code);
    this.code = code;
    this.retryable = retryable;
  }
}

interface Scored {
  scores: Scores;
  embedMs: number;
  scoreMs: number;
  document?: DocumentTelemetry;
}

function optionProblems(o: LoadRouterOptions): string[] {
  const p: string[] = [];
  if (!o || typeof o !== 'object') return ['options must be an object'];
  if (!o.source || typeof o.source.read !== 'function') p.push('source must implement read(key, { signal })');
  if (typeof o.manifestKey !== 'string') p.push('manifestKey must be a string');
  if (o.mode !== 'enforce' && o.mode !== 'shadow') p.push("mode must be 'enforce' or 'shadow' (there is no default)");
  if (!o.encoder || typeof o.encoder.embed !== 'function') p.push('encoder must implement embed(texts, { signal })');
  else p.push(...encoderIdentityProblems(o.encoder.identity).map((e) => `encoder identity ${e}`));
  if (!isTimerMs(o.timeoutMs)) p.push(`timeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}] (longer delays overflow timers and fire at once)`);
  const m = o.monitoring;
  if (!m || typeof m !== 'object') p.push('monitoring must be { sampleRate, samplingSalt }');
  else {
    if (!(typeof m.sampleRate === 'number' && m.sampleRate >= 0 && m.sampleRate <= 1)) p.push('monitoring.sampleRate must be in [0, 1]');
    if (typeof m.samplingSalt !== 'string') p.push('monitoring.samplingSalt must be a string');
    if (m.timeoutMs !== undefined && !isTimerMs(m.timeoutMs)) p.push(`monitoring.timeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}]`);
  }
  if (o.observer !== undefined && typeof o.observer !== 'function') p.push('observer must be a function');
  if (o.observerTimeoutMs !== undefined && !isTimerMs(o.observerTimeoutMs)) p.push(`observerTimeoutMs must be milliseconds in [1, ${MAX_TIMER_MS}]`);
  for (const k of ['tokenizer', 'boundaryTokenizer'] as const) if (o[k] !== undefined && (typeof o[k]?.countInput !== 'function' || typeof o[k]?.cutOffsets !== 'function')) p.push(`${k} must implement countInput and cutOffsets`);
  if (o.boundaryEncoder !== undefined) {
    if (typeof o.boundaryEncoder?.embed !== 'function') p.push('boundaryEncoder must implement embed(texts, { signal })');
    else p.push(...encoderIdentityProblems(o.boundaryEncoder.identity).map((e) => `boundaryEncoder identity ${e}`));
  }
  if (o.document?.batchSize !== undefined && !(Number.isInteger(o.document.batchSize) && o.document.batchSize >= 1 && o.document.batchSize <= 256)) p.push('document.batchSize must be an integer in [1, 256]');
  if (o.document?.yieldEvery !== undefined && !(Number.isInteger(o.document.yieldEvery) && o.document.yieldEvery >= 1)) p.push('document.yieldEvery must be a positive integer');
  if (o.document?.concurrency !== undefined && !(Number.isInteger(o.document.concurrency) && o.document.concurrency >= 1 && o.document.concurrency <= 16)) p.push('document.concurrency must be an integer in [1, 16]');
  for (const k of ['maxManifestBytes', 'maxFileBytes'] as const) {
    const v = o.limits?.[k];
    if (v !== undefined && !(Number.isInteger(v) && v > 0)) p.push(`limits.${k} must be a positive integer`);
  }
  return p;
}

/** Validates the complete release (throws RouterLoadError), then returns a ready router. No request runs before this resolves. */
export async function loadRouter(options: LoadRouterOptions): Promise<Router> {
  const platform = platformProblems();
  if (platform.length) throw new RouterLoadError('PLATFORM_UNSUPPORTED', 'the runtime lacks required Web platform primitives', platform);
  const problems = optionProblems(options);
  if (problems.length) throw new RouterLoadError('OPTIONS_INVALID', 'invalid loadRouter options', problems);
  const limits = { ...DEFAULT_LIMITS, ...options.limits };
  const docs = await readRelease(options.source, options.manifestKey, options.expectedManifestSha256, options.timeoutMs, limits, options.signal);

  // Step 7: the adapter must implement the release's exact identity.
  const identity = deepFreeze(JSON.parse(JSON.stringify(options.encoder.identity)) as EncoderIdentity);
  const diff = encoderIdentityDiff(docs.manifest.embedding, identity);
  if (diff.length) throw new RouterLoadError('ENCODER_IDENTITY_MISMATCH', 'the encoder does not implement the release encoder identity', diff);
  if (docs.document) {
    if (!options.tokenizer) throw new RouterLoadError('CAPABILITY_MISSING', 'this is a document release: supply the tokenizer adapter for its pipeline');
    if (canonicalJson(options.tokenizer.identity ?? null) !== canonicalJson(docs.document.pipeline.tokenizer)) {
      throw new RouterLoadError('TOKENIZER_MISMATCH', "the tokenizer does not implement the release pipeline's tokenizer identity", [`release: ${JSON.stringify(docs.document.pipeline.tokenizer)}`, `adapter: ${JSON.stringify(options.tokenizer.identity ?? null)}`]);
    }
    const g = docs.document.pipeline.grouping;
    if (g.algorithm === 'adjacent-cosine/1') {
      if (!options.boundaryEncoder || !options.boundaryTokenizer) throw new RouterLoadError('CAPABILITY_MISSING', 'this release groups semantically: supply boundaryEncoder and boundaryTokenizer for its pipeline');
      const bd = encoderIdentityDiff(g.boundaryEncoderIdentity as EncoderIdentity, options.boundaryEncoder.identity);
      if (bd.length) throw new RouterLoadError('ENCODER_IDENTITY_MISMATCH', "the boundary encoder does not implement the pipeline's boundary encoder identity", bd);
      if (canonicalJson(options.boundaryTokenizer.identity ?? null) !== canonicalJson(g.boundaryTokenizer)) throw new RouterLoadError('TOKENIZER_MISMATCH', "the boundary tokenizer does not implement the pipeline's boundary tokenizer identity");
    }
  }

  // Step 8: enforcement needs passed classifier and complete-release gates and sufficient evidence.
  if (options.mode === 'enforce') {
    const why = enforcementProblems(docs);
    if (why.length) throw new RouterLoadError('GATES_FAILED', 'refusing to enforce this release (it may be served in shadow mode)', why);
  }

  // Step 9: initialise reference state, then freeze the snapshot we parsed ourselves and compile rules.
  // Preparing first matters: a knn or stack head's reference projection reads the artifact's arrays,
  // which are several times slower to read once frozen (seconds of cold start on a large reference).
  prepareScoring(docs.artifact);
  const artifact = deepFreeze(docs.artifact);
  const ruleSet = deepFreeze(docs.ruleSet);
  const policy = deepFreeze(docs.policy);
  deepFreeze(docs.evidence);
  const manifest = deepFreeze(docs.manifest);
  const document = docs.document ? deepFreeze(docs.document) : null;
  const evaluator = createDecisionEvaluator(
    { heads: Object.fromEntries(Object.entries(artifact.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) },
    policy,
  );
  const snapshot: Snapshot = { manifest, manifestSha256: docs.manifestSha256, artifact, document, matcher: ruleSetMatcher(ruleSet), evaluator, identity };
  return new LoadedRouter(snapshot, options);
}

interface Snapshot {
  manifest: RouterManifest;
  manifestSha256: string;
  artifact: ClassifierArtifact;
  document: DocumentClassifierArtifact | null;
  matcher: ReturnType<typeof ruleSetMatcher>;
  evaluator: DecisionEvaluator;
  identity: Readonly<EncoderIdentity>;
}

class LoadedRouter implements Router {
  readonly releaseId: string;
  readonly manifestSha256: string;
  readonly mode: RouterMode;
  readonly #s: Snapshot;
  readonly #encoder: Encoder;
  readonly #timeoutMs: number;
  readonly #sampleRate: number;
  readonly #salt: string;
  readonly #monitorTimeoutMs: number;
  readonly #tokenizer: LoadRouterOptions['tokenizer'];
  readonly #boundary: { boundaryEncoder: Encoder; boundaryTokenizer: NonNullable<LoadRouterOptions['tokenizer']> } | null;
  readonly #documentOptions: { batchSize?: number; concurrency?: number; yieldEvery?: number };
  /** Observer deliveries and monitoring embeddings that run after a result is returned. */
  readonly #bg: Background;
  readonly #drain = new Drain();
  #closing: Promise<void> | null = null;

  constructor(snapshot: Snapshot, o: LoadRouterOptions) {
    this.#s = snapshot;
    this.releaseId = snapshot.manifest.releaseId;
    this.manifestSha256 = snapshot.manifestSha256;
    this.mode = o.mode;
    this.#encoder = o.encoder;
    this.#timeoutMs = o.timeoutMs;
    this.#sampleRate = o.monitoring.sampleRate;
    this.#salt = o.monitoring.samplingSalt;
    this.#monitorTimeoutMs = o.monitoring.timeoutMs ?? o.timeoutMs;
    this.#bg = new Background(o.observer, o.observerTimeoutMs ?? 250);
    this.#tokenizer = o.tokenizer;
    this.#boundary = o.boundaryEncoder && o.boundaryTokenizer ? { boundaryEncoder: o.boundaryEncoder, boundaryTokenizer: o.boundaryTokenizer } : null;
    this.#documentOptions = { ...o.document };
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

  #base(requestId: string, samplingProbability = this.#sampleRate): EventContext {
    return { releaseId: this.releaseId, requestId, mode: this.mode, samplingProbability };
  }

  /** Refusals are visible in telemetry too, e.g. while a slot drains this router during a swap. */
  #refuse(requestId: string, code: RuntimeErrorCode): RouteResult {
    this.#bg.emit(unavailableEvent(this.#base(requestId), code, 0));
    return unavailableResult(this.mode, this.releaseId, requestId, code, false);
  }

  async route(request: RouteRequest): Promise<RouteResult> {
    const requestId = requestIdOf(request);
    if (this.#drain.closed) return this.#refuse(requestId, 'CLOSED');
    return this.#drain.track(() => this.#route(request, requestId));
  }

  async routeForwarded(forward: ForwardedRequest, options: { signal?: AbortSignal } = {}): Promise<RouteResult | null> {
    const why = forwardProblem(forward);
    const requestId = typeof forward?.requestId === 'string' ? forward.requestId : '';
    if (why) return this.#refuse(requestId, 'INVALID_INPUT');
    if (forward.manifestSha256 !== this.manifestSha256) return this.#refuse(requestId, 'RELEASE_MISMATCH');
    if (forward.purpose === 'decide') return this.route({ text: forward.text, requestId, ...(options.signal ? { signal: options.signal } : {}) });
    if (this.#drain.closed) return this.#refuse(requestId, 'CLOSED');
    // Telemetry only: the rules tier has already answered this request.
    if (requestProblem({ text: forward.text, requestId }, requestId, this.#s.manifest, this.#s.document ? null : this.#s.identity) === null) {
      this.#bg.run(this.#monitor(forward.text, this.#base(requestId, forward.inclusionProbability), forward.inclusionProbability));
    }
    return null;
  }

  async #route(request: RouteRequest, requestId: string): Promise<RouteResult> {
    const t0 = now();
    const s = this.#s;
    const base = this.#base(requestId);
    const fail = (code: RuntimeErrorCode, extra: Partial<DecisionEvent> = {}, retryable = RETRYABLE[code]): RouteResult => {
      this.#bg.emit(unavailableEvent(base, code, now() - t0, extra));
      return unavailableResult(this.mode, this.releaseId, requestId, code, false, retryable);
    };

    // 1. Input and cancellation.
    const problem = requestProblem(request, requestId, s.manifest, s.document ? null : s.identity);
    if (problem) return fail(problem);
    const text = request.text;

    // 2-3. Rules on the original text; conservative settlement.
    const tr = now();
    const found = s.matcher.evaluate(text);
    const rules: RuleFindings = { fired: found.fired, dismissed: found.dismissed };
    const settled = s.evaluator.settle(rules);
    const rulesMs = now() - tr;
    const ruleFields = { firedRule: rules.fired?.id ?? null, dismissed: rules.dismissed };

    if (settled) {
      const sampled = await selectedForMonitoring(this.releaseId, this.#salt, this.#sampleRate, requestId);
      this.#bg.emit(decisionEvent(base, settled, { ...ruleFields, rulesSettled: true, sampled, embedded: false, inclusionProbability: this.#sampleRate, errorCode: null, timings: { totalMs: now() - t0, rulesMs, embedMs: null, scoreMs: null } }));
      // The decision doesn't need the embedding, so the response doesn't wait for it.
      if (sampled) this.#bg.run(this.#monitor(text, base, this.#sampleRate));
      return wrapCandidate(this.mode, this.releaseId, requestId, settled, false);
    }

    // 4. Embed and score within the deadline.
    let scored: Scored;
    try {
      scored = await this.#embedAndScore(text, request.signal, this.#timeoutMs);
    } catch (e) {
      const f = e instanceof RuntimeFailure ? e : new RuntimeFailure('ENCODER_FAILURE');
      return fail(f.code, { ...ruleFields, timings: { totalMs: now() - t0, rulesMs, embedMs: null, scoreMs: null } }, f.retryable);
    }
    // 5-7. Decision.
    let candidate: Candidate;
    try {
      candidate = s.evaluator.decide(rules, scored.scores);
    } catch (e) {
      if (!(e instanceof ScoreError)) throw e;
      return fail('INVALID_SCORE', { ...ruleFields, embedded: true, timings: { totalMs: now() - t0, rulesMs, embedMs: scored.embedMs, scoreMs: scored.scoreMs } });
    }
    this.#bg.emit(decisionEvent(base, candidate, {
      ...ruleFields, rulesSettled: false, sampled: false, embedded: true, inclusionProbability: 1, errorCode: null,
      timings: { totalMs: now() - t0, rulesMs, embedMs: scored.embedMs, scoreMs: scored.scoreMs }, calibratedProbabilities: Object.freeze({ ...scored.scores }) as Record<string, number>,
      ...(scored.document ? { document: scored.document } : {}),
    }));
    return wrapCandidate(this.mode, this.releaseId, requestId, candidate, true);
  }

  /**
   * Embeds a settled request for monitoring, after its result was returned, so the request's signal no
   * longer applies: monitoring.timeoutMs bounds it. Reports and swallows every failure.
   */
  async #monitor(text: string, base: EventContext, inclusionProbability: number): Promise<void> {
    const t = now();
    try {
      const { scores, embedMs, scoreMs, document } = await this.#embedAndScore(text, undefined, this.#monitorTimeoutMs);
      let classifierCandidate: { outcome: 'route' | 'review' | 'abstain'; routeId: string | null };
      try {
        const c = this.#s.evaluator.decide({ fired: null, dismissed: [] }, scores);
        classifierCandidate = { outcome: c.outcome, routeId: c.outcome === 'route' ? c.routeId : null };
      } catch (e) {
        if (!(e instanceof ScoreError)) throw e;
        throw new RuntimeFailure('INVALID_SCORE');
      }
      this.#bg.emit({ ...base, type: 'diagnostic', kind: 'monitoring_sample', inclusionProbability, classifierCandidate, calibratedProbabilities: Object.freeze({ ...scores }) as Record<string, number>, timings: { embedMs, scoreMs }, ...(document ? { document } : {}) });
    } catch (e) {
      const code = e instanceof RuntimeFailure ? e.code : 'ENCODER_FAILURE';
      this.#bg.emit({ ...base, type: 'monitoring_error', code, timings: { embedMs: now() - t } });
    }
  }

  /** Scores for the release's classifier: one vector (legacy) or the shared document scorer (document releases). */
  async #embedAndScore(text: string, signal: AbortSignal | undefined, timeoutMs: number): Promise<Scored> {
    if (this.#s.document) return this.#scoreDocument(this.#s.document, text, signal, timeoutMs);
    const te = now();
    const vector = await this.#embed(text, signal, timeoutMs);
    const embedMs = now() - te;
    const ts = now();
    let scores: Scores;
    try {
      scores = scoreEmbedding(this.#s.artifact, vector);
    } catch {
      throw new RuntimeFailure('INVALID_SCORE');
    }
    for (const v of Object.values(scores)) if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) throw new RuntimeFailure('INVALID_SCORE');
    return { scores, embedMs, scoreMs: now() - ts };
  }

  /**
   * The classifier's document scorer under ONE deadline covering planning, every embedding batch and
   * scoring. Cancellation is checked between planning steps; a timer can't preempt a synchronous
   * tokenizer call. Never yields scores from an incomplete set of chunk vectors.
   */
  async #scoreDocument(artifact: DocumentClassifierArtifact, text: string, outer: AbortSignal | undefined, timeoutMs: number): Promise<Scored> {
    if (outer?.aborted) throw new RuntimeFailure('ABORTED');
    const controller = new AbortController();
    let why: 'PROCESSING_TIMEOUT' | 'ABORTED' | null = null;
    let rejectStop!: (e: unknown) => void;
    const stopped = new Promise<never>((_, reject) => { rejectStop = reject; });
    stopped.catch(() => {});
    const stop = (code: 'PROCESSING_TIMEOUT' | 'ABORTED') => {
      if (why) return;
      why = code;
      controller.abort();
      rejectStop(new RuntimeFailure(code));
    };
    const deadline = now() + timeoutMs;
    const timer = setTimeout(() => stop('PROCESSING_TIMEOUT'), timeoutMs);
    const onAbort = () => stop('ABORTED');
    outer?.addEventListener('abort', onAbort, { once: true });
    try {
      const call = scoreDocument(artifact, text, { tokenizer: this.#tokenizer!, encoder: this.#encoder, ...(this.#boundary ?? {}) }, {
        signal: controller.signal, maxInputUtf8Bytes: this.#s.manifest.serving.maxInputUtf8Bytes, deadline, ...this.#documentOptions,
      });
      call.catch(() => {});
      const r = await Promise.race([call, stopped]);
      // A result that completes past the deadline (e.g. after a long synchronous step) is not served.
      if (why) throw new RuntimeFailure(why);
      if (now() > deadline) throw new RuntimeFailure('PROCESSING_TIMEOUT');
      const document: DocumentTelemetry = {
        chunkCount: r.chunkCount, inputTokens: r.inputTokens, embedCalls: r.embedCalls, featureIdentitySha256: r.featureIdentitySha256,
        planningMs: r.timings.planningMs, embeddingMs: r.timings.embeddingMs, scoringMs: r.timings.scoringMs,
        ...(r.boundary.calls ? { boundary: { calls: r.boundary.calls, inputTokens: r.boundary.inputTokens, ms: r.boundary.ms } } : {}),
      };
      return { scores: r.scores, embedMs: r.timings.planningMs + r.boundary.ms + r.timings.embeddingMs, scoreMs: r.timings.scoringMs, document };
    } catch (e) {
      if (e instanceof RuntimeFailure) throw e;
      if (e instanceof DocumentError) {
        if (e.code === 'ABORTED') throw new RuntimeFailure(why ?? 'ABORTED');
        if (e.code === 'DEADLINE_EXCEEDED') throw new RuntimeFailure(why ?? 'PROCESSING_TIMEOUT');
        const code: RuntimeErrorCode = e.code === 'IDENTITY_MISMATCH' ? 'PREPROCESSING_FAILURE' : e.code;
        throw new RuntimeFailure(code, code === 'ENCODER_FAILURE' ? (e.retryable ?? RETRYABLE.ENCODER_FAILURE) : RETRYABLE[code]);
      }
      throw new RuntimeFailure('ENCODER_FAILURE');
    } finally {
      clearTimeout(timer);
      outer?.removeEventListener('abort', onAbort);
    }
  }

  /** One validated vector, or a RuntimeFailure (ABORTED, ENCODER_TIMEOUT, ENCODER_FAILURE, INVALID_EMBEDDING). */
  async #embed(text: string, outer: AbortSignal | undefined, timeoutMs: number): Promise<readonly number[]> {
    if (outer?.aborted) throw new RuntimeFailure('ABORTED');
    const controller = new AbortController();
    let why: RuntimeErrorCode | null = null;
    let rejectStop!: (e: unknown) => void;
    const stopped = new Promise<never>((_, reject) => { rejectStop = reject; });
    const stop = (code: RuntimeErrorCode) => {
      if (why) return;
      why = code;
      controller.abort();
      rejectStop(new RuntimeFailure(code));
    };
    const timer = setTimeout(() => stop('ENCODER_TIMEOUT'), timeoutMs);
    const onAbort = () => stop('ABORTED');
    outer?.addEventListener('abort', onAbort, { once: true });
    let rows: unknown;
    try {
      const call = Promise.resolve().then(() => this.#encoder.embed([text], { signal: controller.signal }));
      call.catch(() => {});
      rows = await Promise.race([call, stopped]);
    } catch (e) {
      if (e instanceof RuntimeFailure) throw e;
      if (e instanceof EncoderError) throw new RuntimeFailure(e.code, e.retryable);
      throw new RuntimeFailure('ENCODER_FAILURE');
    } finally {
      clearTimeout(timer);
      outer?.removeEventListener('abort', onAbort);
    }
    if (!Array.isArray(rows) || rows.length !== 1) throw new RuntimeFailure('INVALID_EMBEDDING');
    if (vectorProblems(this.#s.identity, rows[0]) !== null) throw new RuntimeFailure('INVALID_EMBEDDING');
    return rows[0] as readonly number[];
  }
}
