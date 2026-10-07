import assert from 'node:assert/strict';
import { test } from 'node:test';

import { EncoderError, functionEncoder, loadRouter, RouterSlot, type Encoder, type RouteResult, type RouterEvent } from '../src/index.ts';
import { fakeEncoder, IDENTITY, load, makeRelease, options, vectorFor } from './helpers.ts';

/** The decision fields that must not depend on monitoring, mode wrappers or hosting. */
const semantic = (r: RouteResult) => {
  if (r.outcome === 'shadow') return r.candidate;
  if (r.outcome === 'route') return { outcome: r.outcome, routeId: r.routeId, destination: r.destination, mechanism: r.mechanism };
  if (r.outcome === 'unavailable') return { outcome: r.outcome, code: r.code };
  return { outcome: r.outcome, reason: r.reason };
};

test('enforce: rules settle without embedding; otherwise the classifier decides', async () => {
  const { encoder, calls } = fakeEncoder({ 'refund please': { refund: 0.2 }, 'money back': { refund: 0.9 }, borderline: { refund: 0.5 }, nothing: {}, 'emergency hello': {} });
  const router = await load({}, { encoder });
  const r0 = await router.route({ text: 'EMERGENCY!', requestId: '0' });
  assert.deepEqual(r0, { releaseId: 'release-1', requestId: '0', embedded: false, mode: 'enforce', actionable: true, outcome: 'route', routeId: 'urgent', destination: 'queue:urgent', mechanism: 'rule' });
  assert.deepEqual(calls, [], 'a top-priority, unsuppressible rule settles without embedding');
  // A refund rule can't settle while the higher-priority urgent head is unresolved: embed, then the rule still decides.
  const r1 = await router.route({ text: 'Refund, please!', requestId: '1' });
  assert.deepEqual(semantic(r1), { outcome: 'route', routeId: 'refund', destination: 'queue:refunds', mechanism: 'rule' });
  assert.equal(r1.embedded, true);
  const r2 = await router.route({ text: 'money back', requestId: '2' });
  assert.equal(r2.outcome === 'route' && r2.mechanism, 'classifier');
  assert.equal(r2.embedded, true);
  assert.deepEqual(semantic(await router.route({ text: 'borderline', requestId: '3' })), { outcome: 'review', reason: 'near_threshold' });
  assert.deepEqual(semantic(await router.route({ text: 'nothing', requestId: '4' })), { outcome: 'abstain', reason: 'no_match' });
  assert.deepEqual(semantic(await router.route({ text: 'hello', requestId: '5' })), { outcome: 'abstain', reason: 'all_dismissed' });
  assert.deepEqual(semantic(await router.route({ text: 'emergency hello', requestId: '6' })), { outcome: 'route', routeId: 'urgent', destination: 'queue:urgent', mechanism: 'rule' }, 'firing beats dismissal');
});

test('onAbstain review turns abstentions into reviews', async () => {
  const router = await load({ policy: (p) => { p.onAbstain = 'review'; } });
  assert.deepEqual(semantic(await router.route({ text: 'hello', requestId: '1' })), { outcome: 'review', reason: 'abstention_policy' });
});

test('shadow results are never actionable and carry no top-level destination', async () => {
  const router = await load({}, { mode: 'shadow' });
  const r = await router.route({ text: 'refund please', requestId: '1' });
  assert.equal(r.mode, 'shadow');
  assert.equal(r.actionable, false);
  assert.equal('destination' in r, false);
  assert.deepEqual(r.outcome === 'shadow' && r.candidate, { outcome: 'route', routeId: 'refund', destination: 'queue:refunds', mechanism: 'rule' });
});

test('unavailable: invalid input, encoder timeout, failure, cancellation and invalid embeddings', async () => {
  const built = await makeRelease();
  const route = async (encoder: Encoder, text = 'money back', extra: { signal?: AbortSignal; timeoutMs?: number } = {}) => {
    const router = await loadRouter(options(built, { encoder, timeoutMs: extra.timeoutMs ?? 1000 }));
    return router.route({ text, requestId: 'r', signal: extra.signal });
  };
  const ok = fakeEncoder().encoder;
  const expectCode = (r: RouteResult, code: string, retryable?: boolean) => {
    assert.equal(r.outcome, 'unavailable');
    assert.equal(r.actionable, false);
    if (r.outcome === 'unavailable') { assert.equal(r.code, code); if (retryable !== undefined) assert.equal(r.retryable, retryable); }
  };
  expectCode(await route(ok, 'x'.repeat(4097)), 'INVALID_INPUT', false);
  expectCode(await route(ok, 'é'.repeat(2049)), 'INVALID_INPUT');
  expectCode(await (await loadRouter(options(built))).route({ text: 42 as never, requestId: 'r' }), 'INVALID_INPUT');
  expectCode(await (await loadRouter(options(built))).route({ text: 'x', requestId: '' }), 'INVALID_INPUT');
  const hang: Encoder = { identity: IDENTITY, embed: () => new Promise(() => {}) };
  expectCode(await route(hang, 'money back', { timeoutMs: 20 }), 'ENCODER_TIMEOUT', true);
  expectCode(await route({ identity: IDENTITY, embed: async () => { throw new Error('503'); } }), 'ENCODER_FAILURE', true);
  expectCode(await route({ identity: IDENTITY, embed: async () => { throw new EncoderError('ENCODER_FAILURE', 'quota', { retryable: false }); } }), 'ENCODER_FAILURE', false);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10);
  expectCode(await route(hang, 'money back', { signal: controller.signal }), 'ABORTED', false);
  for (const rows of [[], [[0, 0, 0]], [[0, 0, 0, Number.NaN]], [vectorFor({}), vectorFor({})], ['x']]) {
    expectCode(await route({ identity: IDENTITY, embed: async () => rows as never }), 'INVALID_EMBEDDING', false);
  }
});

test('normalising releases refuse vectors that break the declared normalisation, through route() (review S5)', async () => {
  const outcome = async (r: Promise<RouteResult>) => { const x = await r; return x.outcome === 'unavailable' ? x.code : x.outcome; };
  // 'unit': the whole vector has length 1 (classifier: normalize true, no layers).
  const unit = { ...IDENTITY, normalization: 'unit' as const };
  const unitRelease = await makeRelease({ identity: unit, artifact: (a) => { a.embedding.normalize = true; } });
  const route = async (built: typeof unitRelease, identity: typeof IDENTITY, vector: number[]) =>
    outcome((await loadRouter(options(built, { encoder: functionEncoder(identity, () => vector) }))).route({ text: 'nothing', requestId: '1' }));
  assert.equal(await route(unitRelease, unit, [3, 0, 0, 0]), 'INVALID_EMBEDDING');
  assert.notEqual(await route(unitRelease, unit, [0.6, 0.8, 0, 0]), 'INVALID_EMBEDDING');

  // 'per-layer-unit': each layer block has length 1; the concatenation does not.
  const layered = { ...IDENTITY, layers: [1, 2], normalization: 'per-layer-unit' as const };
  const layeredRelease = await makeRelease({ identity: layered, artifact: (a) => { Object.assign(a.embedding, { normalize: false, layers: [1, 2], pooling: 'mean', layer_normalize: true }); } });
  assert.equal(await route(layeredRelease, layered, [0.6, 0.8, 0, 0]), 'INVALID_EMBEDDING', 'a globally unit vector is not per-layer unit');
  assert.notEqual(await route(layeredRelease, layered, [1, 0, 0.6, 0.8]), 'INVALID_EMBEDDING');
  assert.equal(await route(layeredRelease, layered, [1, 0, 3, 4]), 'INVALID_EMBEDDING');
});

test('a firing rule on a disabled head decides identically at monitoring shares 0 and 1', async () => {
  const disabled = { artifact: (a: { heads: Record<string, { threshold: number; review_floor: number } | undefined> }) => { a.heads.urgent!.threshold = 1.0000000000000002; a.heads.urgent!.review_floor = 0.9; } };
  const texts = ['emergency now', 'refund please', 'money back', 'hello', 'nothing', 'emergency hello'];
  const outcomes: unknown[][] = [];
  for (const sampleRate of [0, 1]) {
    const router = await load(disabled as never, { mode: 'shadow', encoder: fakeEncoder({ 'money back': { refund: 0.9 } }).encoder, monitoring: { sampleRate, samplingSalt: 's' } });
    outcomes.push(await Promise.all(texts.map(async (text, i) => semantic(await router.route({ text, requestId: String(i) })))));
  }
  assert.deepEqual(outcomes[0], outcomes[1]);
  assert.deepEqual(outcomes[0][0], { outcome: 'route', routeId: 'urgent', destination: 'queue:urgent', mechanism: 'rule' });
});

test('monitoring: deterministic sampling by request id, failures never change settled decisions', async () => {
  const events: RouterEvent[] = [];
  const built = await makeRelease();
  const { encoder, calls } = fakeEncoder();
  const half = await loadRouter(options(built, { encoder, monitoring: { sampleRate: 0.5, samplingSalt: 'salt' }, observer: (e) => { events.push(e); } }));
  const picks = async () => { const out: boolean[] = []; for (let i = 0; i < 200; i++) { const before = calls.length; await half.route({ text: 'emergency', requestId: `id-${i}` }); await half.flush(); out.push(calls.length > before); } return out; };
  const first = await picks();
  assert.deepEqual(await picks(), first, 'same release, salt and id: same selection');
  const share = first.filter(Boolean).length / first.length;
  assert.ok(share > 0.35 && share < 0.65, `share ${share}`);
  const diag = events.filter((e) => e.type === 'diagnostic');
  assert.ok(diag.length > 0);
  for (const e of events) {
    if (e.type === 'decision' && e.rulesSettled) assert.equal(e.inclusionProbability, 0.5);
    assert.equal(JSON.stringify(e).includes('emergency'), false, 'no request text in telemetry');
  }

  events.length = 0;
  const failing = await loadRouter(options(built, { encoder: { identity: IDENTITY, embed: async () => { throw new Error('down'); } }, monitoring: { sampleRate: 1, samplingSalt: 's' }, observer: (e) => { events.push(e); } }));
  const r = await failing.route({ text: 'emergency', requestId: 'x' });
  assert.deepEqual(semantic(r), { outcome: 'route', routeId: 'urgent', destination: 'queue:urgent', mechanism: 'rule' });
  assert.equal(r.embedded, false);
  await failing.flush();
  assert.deepEqual(events.map((e) => e.type), ['decision', 'monitoring_error'], 'the decision is reported first; monitoring runs after it');
});

test('observer failures and slow observers cannot change decisions; drops are counted', async () => {
  const built = await makeRelease();
  for (const observer of [() => { throw new Error('boom'); }, async () => { throw new Error('boom'); }, () => new Promise<void>(() => {})]) {
    const router = await loadRouter(options(built, { observer, observerTimeoutMs: 10 }));
    const r = await router.route({ text: 'I want my money back', requestId: '1' });
    assert.equal(r.outcome, 'route');
    await router.flush();
    assert.deepEqual(router.telemetry(), { delivered: 0, dropped: 1 });
  }
  const events: RouterEvent[] = [];
  const router = await loadRouter(options(built, { observer: async (e) => { events.push(e); } }));
  await router.route({ text: 'I want my money back', requestId: '1' });
  const [e] = events;
  assert.equal(e.type, 'decision');
  if (e.type === 'decision') {
    assert.equal(e.releaseId, 'release-1');
    assert.equal(e.mechanism, 'classifier');
    assert.equal(e.embedded, true);
    assert.equal(e.inclusionProbability, 1);
    assert.ok(e.calibratedProbabilities && e.calibratedProbabilities.refund > 0.8);
    assert.ok(e.timings.embedMs !== null);
  }
  await router.flush();
  assert.deepEqual(router.telemetry(), { delivered: 1, dropped: 0 });
});

test('close drains in-flight requests, then refuses new ones', async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const encoder: Encoder = { identity: IDENTITY, embed: async () => { await gate; return [vectorFor({ refund: 0.9 })]; } };
  const router = await load({}, { encoder });
  const pending = router.route({ text: 'I want my money back', requestId: 'in-flight' });
  let closed = false;
  const closing = router.close().then(() => { closed = true; });
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(closed, false);
  const late = await router.route({ text: 'I want my money back', requestId: 'late' });
  assert.equal(late.outcome === 'unavailable' && late.code, 'CLOSED');
  release();
  assert.equal((await pending).outcome, 'route');
  await closing;
  assert.equal(closed, true);
});

test('reload: a failed reload keeps the previous release; a good one swaps after draining', async () => {
  const first = await load();
  const slot = new RouterSlot(first);
  const failed = await slot.reload(async () => { throw new Error('bad release'); });
  assert.equal(failed.ok, false);
  assert.equal(slot.current, first);
  assert.equal((await slot.route({ text: 'refund please', requestId: '1' })).releaseId, 'release-1');

  const built = await makeRelease();
  const swapped = await slot.reload(() => loadRouter(options(built)));
  assert.equal(swapped.ok, true);
  assert.notEqual(slot.current, first);
  assert.equal((await first.route({ text: 'x', requestId: '2' })).outcome, 'unavailable', 'the old router was closed');
});
