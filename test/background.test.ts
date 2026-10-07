import assert from 'node:assert/strict';
import { test } from 'node:test';

import { loadRouter, type Encoder, type RouterEvent } from '../src/index.ts';
import { fakeEncoder, makeRelease, options } from './helpers.ts';

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const timed = async <T>(p: Promise<T>) => { const t0 = performance.now(); const value = await p; return { value, ms: performance.now() - t0 }; };
/** The fixture encoder, `ms` slower per call. */
const slowEncoder = (ms: number): Encoder => {
  const { encoder } = fakeEncoder();
  return { identity: encoder.identity, embed: async (texts, o) => { await sleep(ms); return encoder.embed(texts, o); } };
};
const SLOW = 300;

test('an asynchronous observer never delays a result; flush waits for delivery', async () => {
  const built = await makeRelease();
  const router = await loadRouter(options(built, { observer: async () => { await sleep(SLOW); }, observerTimeoutMs: 2 * SLOW }));
  const { value: r, ms } = await timed(router.route({ text: 'refund please', requestId: '1' }));
  assert.equal(r.outcome, 'route');
  assert.ok(ms < SLOW / 2, `route took ${ms} ms`);
  assert.deepEqual(router.telemetry(), { delivered: 0, dropped: 0 }, 'still delivering');
  await router.flush();
  assert.deepEqual(router.telemetry(), { delivered: 1, dropped: 0 });
});

test('a monitoring sample runs after the result: the rules tier answers without waiting for the embedding', async () => {
  const built = await makeRelease();
  const events: RouterEvent[] = [];
  const router = await loadRouter(options(built, { encoder: slowEncoder(SLOW), monitoring: { sampleRate: 1, samplingSalt: 's' }, observer: (e) => { events.push(e); } }));
  const controller = new AbortController();
  // 'emergency' fires the top-priority rule, so no score can change the outcome: rules-settled.
  const { value: r, ms } = await timed(router.route({ text: 'emergency', requestId: '1', signal: controller.signal }));
  assert.ok(ms < SLOW / 2, `route took ${ms} ms`);
  assert.equal(r.outcome === 'route' && r.mechanism, 'rule');
  assert.equal(r.embedded, false, 'the decision did not embed');
  assert.deepEqual(events.map((e) => e.type === 'decision' && [e.rulesSettled, e.sampled, e.embedded]), [[true, true, false]]);
  // The request is answered, so its signal no longer applies to the sample.
  controller.abort();
  await router.flush();
  assert.deepEqual(events.map((e) => e.type), ['decision', 'diagnostic']);
});

test('close waits for background monitoring and delivery to finish', async () => {
  const built = await makeRelease();
  const events: RouterEvent[] = [];
  const router = await loadRouter(options(built, { encoder: slowEncoder(SLOW / 3), monitoring: { sampleRate: 1, samplingSalt: 's' }, observer: async (e) => { await sleep(10); events.push(e); } }));
  await router.route({ text: 'emergency', requestId: '1' });
  await router.close();
  assert.deepEqual(events.map((e) => e.type), ['decision', 'diagnostic']);
  assert.deepEqual(router.telemetry(), { delivered: 2, dropped: 0 });
});
