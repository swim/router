/**
 * The review adapter (review.ts), end to end through loadRouter and loadRulesTier: answers are
 * advisory and never change a decision, failures keep the review, the human sample is deterministic,
 * and telemetry carries no text.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  loadRouter, loadRulesTier, memorySource, REVIEW_ADAPTER_SCHEMA, ReviewAdapterError, reviewAdapterContractProblems, reviewAdapterIdentityDigest, reviewAdapterIdentityProblems, RouterLoadError, type DecisionEvent, type ReviewAdapter, type ReviewAnswer, type ReviewInput, type RouterEvent,
} from '../src/index.ts';
import { EVAL, fakeEncoder, load, makeRelease, options, POLICY, vectorFor } from './helpers.ts';

const IDENTITY_R = { schema: REVIEW_ADAPTER_SCHEMA, provider: 'test', model: 'echo-1', promptSha256: 'a'.repeat(64), outputSchemaSha256: 'b'.repeat(64) } as const;

/** An adapter answering from a table (text -> route id), recording what it was shown. */
function tableAdapter(table: Record<string, string | null> = {}, behave: (input: ReviewInput, signal: AbortSignal) => Promise<ReviewAnswer> | null = () => null) {
  const seen: ReviewInput[] = [];
  const adapter: ReviewAdapter = {
    identity: IDENTITY_R,
    async review(input, { signal }) {
      seen.push(input);
      const special = behave(input, signal);
      if (special) return special;
      if (signal.aborted) throw new Error('aborted');
      return { routeId: table[input.text] ?? null, confidence: 0.8, rationale: 'because' };
    },
  };
  return { adapter, seen };
}

/** The review band: urgent at 0.5 is above its floor (0.3) and below its threshold (0.8). */
const REVIEW_TEXT = 'not sure if urgent';
const encoderWith = (extra: Record<string, Parameters<typeof vectorFor>[0]>) => fakeEncoder({ ...Object.fromEntries(EVAL.map((e) => [e.text, e.p])), ...extra }).encoder;

test('review adapter identity and conformance checks', async () => {
  assert.deepEqual(reviewAdapterIdentityProblems(IDENTITY_R), []);
  assert.deepEqual(reviewAdapterIdentityProblems({ ...IDENTITY_R, promptSha256: 'nope', extra: 1 }).sort(), ['promptSha256 must be lowercase hex SHA-256', 'unknown field extra']);
  assert.match((await reviewAdapterIdentityDigest(IDENTITY_R)), /^[0-9a-f]{64}$/);
  const routes = POLICY.routes;
  assert.deepEqual(await reviewAdapterContractProblems(tableAdapter().adapter, { routes }), []);
  const unknown = tableAdapter({}, () => Promise.resolve({ routeId: 'billing', confidence: 0.9 }));
  assert.match((await reviewAdapterContractProblems(unknown.adapter, { routes }))[0], /routeId "billing" is not one of the routes/);
  const ignoresAbort: ReviewAdapter = { identity: IDENTITY_R, review: async () => ({ routeId: null, confidence: null }) };
  assert.ok((await reviewAdapterContractProblems(ignoresAbort, { routes })).some((p) => /already-aborted signal answered/.test(p)));
  const slow: ReviewAdapter = { identity: IDENTITY_R, review: (_, { signal }) => new Promise((resolve, reject) => { const t = setTimeout(() => resolve({ routeId: null, confidence: null }), 200); signal.addEventListener('abort', () => { clearTimeout(t); reject(new Error('aborted')); }); }) };
  assert.ok((await reviewAdapterContractProblems(slow, { routes, timeoutMs: 20 })).some((p) => /did not finish within 20 ms/.test(p)));
});

test('loadRouter with a review adapter: a review carries the advisory answer; the decision never changes', async () => {
  const built = await makeRelease();
  const { adapter, seen } = tableAdapter({ [REVIEW_TEXT]: 'urgent' });
  const events: RouterEvent[] = [];
  const encoder = encoderWith({ [REVIEW_TEXT]: { urgent: 0.5 } });
  const plain = await loadRouter(options(built, { encoder }));
  const reviewed = await loadRouter(options(built, { encoder, observer: (e) => { events.push(e); }, review: { adapter, timeoutMs: 1000, humanSampleRate: 0 } }));
  const r = await reviewed.route({ text: REVIEW_TEXT, requestId: 'q1' });
  assert.equal(r.outcome, 'review');
  assert.equal(r.actionable, false);
  if (r.outcome !== 'review') return;
  assert.equal(r.reason, 'near_threshold');
  assert.deepEqual(r.answer, { routeId: 'urgent', destination: 'queue:urgent', confidence: 0.8, adapterSha256: await reviewAdapterIdentityDigest(IDENTITY_R) });
  assert.equal(r.forHuman, false);
  // The adapter saw the routes and the evidence, never the request id or the release digest.
  assert.deepEqual(seen[0].routes.map((x) => x.id), ['urgent', 'refund', 'other']);
  assert.equal(seen[0].evidence.reason, 'near_threshold');
  assert.ok(Math.abs(seen[0].evidence.probabilities.urgent - 0.5) < 1e-9);
  assert.ok(!JSON.stringify(seen[0]).includes('q1') && !JSON.stringify(seen[0]).includes(built.manifestSha256));
  await reviewed.flush();
  const ev = events.find((e): e is DecisionEvent => e.type === 'decision')!;
  assert.deepEqual({ ...ev.review, reviewMs: 0 }, { answered: true, adapterSha256: r.answer!.adapterSha256, routeId: 'urgent', confidence: 0.8, errorCode: null, forHuman: false, reviewMs: 0 });
  assert.ok(typeof ev.timings.reviewMs === 'number');
  assert.ok(!JSON.stringify(events).includes('because') && !JSON.stringify(events).includes(REVIEW_TEXT), 'no rationale or text in telemetry');
  // Every evaluation text decides exactly as without the adapter, apart from the answer fields.
  for (const t of [...EVAL.map((e) => e.text), REVIEW_TEXT]) {
    const a = await plain.route({ text: t, requestId: t }), b = await reviewed.route({ text: t, requestId: t });
    const strip = (x: typeof a) => { const { answer, answerError, forHuman, ...rest } = x as typeof x & { answer?: unknown; answerError?: unknown; forHuman?: unknown }; void answer; void answerError; void forHuman; return rest; };
    assert.deepEqual(strip(b), strip(a), t);
  }
});

test('adapter failures keep the review: refused, failed, timed out and invalid answers', async () => {
  const built = await makeRelease();
  const encoder = encoderWith({ [REVIEW_TEXT]: { urgent: 0.5 } });
  const cases: Array<[string, ReviewAdapter, string, boolean]> = [
    ['refused', tableAdapter({}, () => Promise.reject(new ReviewAdapterError('REFUSED', 'declined'))).adapter, 'REFUSED', false],
    ['throws', tableAdapter({}, () => Promise.reject(new Error('socket hang up'))).adapter, 'FAILURE', true],
    ['slow', tableAdapter({}, (_, signal) => new Promise((resolve, reject) => { const t = setTimeout(() => resolve({ routeId: null, confidence: null }), 500); signal.addEventListener('abort', () => { clearTimeout(t); reject(new Error('aborted')); }); })).adapter, 'TIMEOUT', true],
    ['invalid', tableAdapter({}, () => Promise.resolve({ routeId: 'billing', confidence: 2 })).adapter, 'INVALID_ANSWER', false],
  ];
  for (const [name, adapter, code, retryable] of cases) {
    const router = await loadRouter(options(built, { encoder, review: { adapter, timeoutMs: 30, humanSampleRate: 0 } }));
    const r = await router.route({ text: REVIEW_TEXT, requestId: name });
    assert.equal(r.outcome, 'review', name);
    if (r.outcome !== 'review') continue;
    assert.deepEqual(r.answerError, { code, retryable }, name);
    assert.equal(r.answer, undefined);
    assert.equal(r.forHuman, true, `${name}: an unanswered review goes to a person`);
  }
});

test('the human sample is deterministic and independent of the monitoring sample', async () => {
  const built = await makeRelease();
  const encoder = encoderWith({ [REVIEW_TEXT]: { urgent: 0.5 } });
  const make = () => loadRouter(options(built, { encoder, review: { adapter: tableAdapter().adapter, timeoutMs: 1000, humanSampleRate: 0.5 } }));
  const [a, b] = await Promise.all([make(), make()]);
  let human = 0;
  for (let i = 0; i < 200; i++) {
    const ra = await a.route({ text: REVIEW_TEXT, requestId: `id-${i}` }), rb = await b.route({ text: REVIEW_TEXT, requestId: `id-${i}` });
    assert.equal((ra as { forHuman?: boolean }).forHuman, (rb as { forHuman?: boolean }).forHuman);
    if ((ra as { forHuman?: boolean }).forHuman) human++;
  }
  assert.ok(human > 70 && human < 130, `${human} of 200 sampled for a person at rate 0.5`);
});

test('shadow mode: the candidate carries the answer; review options are validated', async () => {
  const built = await makeRelease();
  const encoder = encoderWith({ [REVIEW_TEXT]: { urgent: 0.5 } });
  const shadow = await loadRouter(options(built, { mode: 'shadow', encoder, review: { adapter: tableAdapter({ [REVIEW_TEXT]: 'refund' }).adapter, timeoutMs: 1000, humanSampleRate: 0 } }));
  const r = await shadow.route({ text: REVIEW_TEXT, requestId: 's' });
  assert.equal(r.outcome, 'shadow');
  if (r.outcome === 'shadow') assert.deepEqual(r.candidate.outcome === 'review' && r.candidate.answer?.routeId, 'refund');
  await assert.rejects(load({}, { review: { adapter: { identity: { ...IDENTITY_R, schema: 'x' } } as never, timeoutMs: 0, humanSampleRate: 2 } }), (e: unknown) => e instanceof RouterLoadError && e.code === 'OPTIONS_INVALID' && e.details.length === 3);
});

test('a review the rules settle is answered alike by one router and by a split rules tier', async () => {
  // onAbstain 'review': 'hello' dismisses every head, so the rules alone settle it to review.
  const built = await makeRelease({ policy: (p) => { p.onAbstain = 'review'; } });
  const review = { adapter: tableAdapter({ hello: null }).adapter, timeoutMs: 1000, humanSampleRate: 0.5 };
  const router = await loadRouter(options(built, { review }));
  const tier = await loadRulesTier({ source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000, monitoring: { sampleRate: 0, samplingSalt: 'salt' }, review });
  for (const requestId of ['h1', 'h2', 'h3', 'h4']) {
    const one = await router.route({ text: 'hello', requestId });
    const split = await tier.handle({ text: 'hello', requestId });
    assert.equal(split.answered, true);
    if (!split.answered) continue;
    assert.deepEqual(split.result, one);
    assert.equal(one.outcome === 'review' && one.reason, 'abstention_policy');
    assert.equal(one.outcome === 'review' && one.answer?.routeId, null);
  }
});
