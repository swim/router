import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createDecisionEvaluator, createSettler, loadRouter, loadRulesTier, memorySource, RouterLoadError, type ForwardedRequest, type LoadRulesTierOptions, type RouteRequest, type RouterEvent, type RouteResult } from '../src/index.ts';
import { EVAL, fakeEncoder, HEADS, makeRelease, options, POLICY } from './helpers.ts';

const code = (c: string) => (e: unknown) => e instanceof RouterLoadError && e.code === c;
const rulesOptions = (built: Awaited<ReturnType<typeof makeRelease>>, over: Partial<LoadRulesTierOptions> = {}): LoadRulesTierOptions => ({
  source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000,
  monitoring: { sampleRate: 0, samplingSalt: 'salt' }, ...over,
});
/** Events without timings (which differ by construction), as comparable strings. */
const comparable = (events: RouterEvent[]) => events.map((e) => { const { timings: _t, ...rest } = e as RouterEvent & { timings?: unknown }; return JSON.stringify(rest); }).sort();

// Texts the fixture rules settle, dismiss or leave to the classifier, plus some the encoder table doesn't know.
const TEXTS = [...EVAL.map((e) => e.text), 'emergency refund please', 'hello there', 'refund please and hello', 'nothing to see', '', 'x'.repeat(5000)];
const table = { ...Object.fromEntries(EVAL.map((e) => [e.text, e.p])), 'nothing to see': { urgent: 0.5, refund: 0.5 }, 'hello there': { other: 0.99 } };

for (const mode of ['enforce', 'shadow'] as const) {
  test(`split deployment (${mode}): rules tier + model tier answer exactly as one router, with the same telemetry`, async () => {
    const built = await makeRelease();
    const monitoring = { sampleRate: 0.5, samplingSalt: 'salt' };
    const single: RouterEvent[] = [], rulesEvents: RouterEvent[] = [], modelEvents: RouterEvent[] = [];
    const router = await loadRouter(options(built, { mode, monitoring, encoder: fakeEncoder(table).encoder, observer: (e) => { single.push(e); } }));
    const model = await loadRouter(options(built, { mode, monitoring, encoder: fakeEncoder(table).encoder, observer: (e) => { modelEvents.push(e); } }));
    const rules = await loadRulesTier(rulesOptions(built, { mode, monitoring, observer: (e) => { rulesEvents.push(e); } }));
    let forwarded = 0, answered = 0, monitored = 0;
    for (let round = 0; round < 3; round++) {
      for (const [i, text] of TEXTS.entries()) {
        const request: RouteRequest = { text, requestId: `r${round}-${i}` };
        const expected = await router.route(request);
        const outcome = await rules.handle(request);
        let actual: RouteResult | null;
        if (outcome.answered) {
          answered++;
          actual = outcome.result;
          // The forward is JSON: the tiers can be separate functions.
          if (outcome.monitor) { monitored++; assert.equal(await model.routeForwarded(JSON.parse(JSON.stringify(outcome.monitor))), null); }
        } else {
          forwarded++;
          actual = await model.routeForwarded(JSON.parse(JSON.stringify(outcome.forward)));
        }
        assert.deepEqual(actual, expected, text);
      }
    }
    await Promise.all([router.flush(), model.flush(), rules.flush()]);
    assert.ok(answered > 0 && forwarded > 0 && monitored > 0, `answered ${answered}, forwarded ${forwarded}, monitored ${monitored}`);
    assert.deepEqual(comparable([...rulesEvents, ...modelEvents]), comparable(single), 'the two tiers together report exactly what one router reports');
  });
}

test('the rules tier never reads the classifier and validates what it loads like loadRouter', async () => {
  const built = await makeRelease();
  const reads: string[] = [];
  const inner = memorySource(built.files);
  const rules = await loadRulesTier(rulesOptions(built, { source: { read: (key, o) => { reads.push(key); return inner.read(key, o); } } }));
  assert.deepEqual(reads.sort(), ['evidence.json', 'manifest.json', 'policy.json', 'rules.json']);
  assert.equal(rules.manifestSha256, built.manifestSha256);
  // A changed rules file, another manifest digest, failed gates in enforce mode.
  const files = new Map(built.files);
  files.set('rules.json', new TextEncoder().encode(new TextDecoder().decode(files.get('rules.json')).replace('"emergency"', '"emergencies"')));
  await assert.rejects(loadRulesTier(rulesOptions(built, { source: memorySource(files) })), code('FILE_DIGEST_MISMATCH'));
  await assert.rejects(loadRulesTier(rulesOptions(built, { expectedManifestSha256: 'e'.repeat(64) })), code('MANIFEST_DIGEST_MISMATCH'));
  const failing = await makeRelease({ evaluationPasses: false });
  await assert.rejects(loadRulesTier(rulesOptions(failing)), code('GATES_FAILED'));
  await loadRulesTier(rulesOptions(failing, { mode: 'shadow' }));
  await assert.rejects(loadRulesTier(rulesOptions(built, { mode: 'live' as never })), code('OPTIONS_INVALID'));
});

test('the model tier refuses forwards from another release or malformed ones; a monitor forward never answers', async () => {
  const built = await makeRelease();
  const events: RouterEvent[] = [];
  const model = await loadRouter(options(built, { encoder: fakeEncoder(table).encoder, observer: (e) => { events.push(e); } }));
  const forward: ForwardedRequest = { schema: 'liquidau-router-forward/1', manifestSha256: built.manifestSha256, requestId: 'f1', text: 'I want my money back', purpose: 'decide', inclusionProbability: 1 };
  assert.equal((await model.routeForwarded(forward))?.outcome, 'route');
  const other = await model.routeForwarded({ ...forward, manifestSha256: 'a'.repeat(64) });
  assert.deepEqual(other && other.outcome === 'unavailable' && [other.code, other.retryable], ['RELEASE_MISMATCH', true]);
  for (const bad of [{ ...forward, purpose: 'other' }, { ...forward, extra: 1 }, { ...forward, inclusionProbability: 0 }, null]) {
    const r = await model.routeForwarded(bad as never);
    assert.equal(r && r.outcome === 'unavailable' && r.code, 'INVALID_INPUT');
  }
  // 'monitor': telemetry only, at the rules tier's inclusion probability.
  events.length = 0;
  assert.equal(await model.routeForwarded({ ...forward, requestId: 'm1', text: 'emergency', purpose: 'monitor', inclusionProbability: 0.05 }), null);
  await model.flush();
  assert.deepEqual(events.map((e) => [e.type, e.requestId, e.type === 'diagnostic' && e.inclusionProbability]), [['diagnostic', 'm1', 0.05]]);
  await model.close();
  const closed = await model.routeForwarded(forward);
  assert.equal(closed && closed.outcome === 'unavailable' && closed.code, 'CLOSED');
});

test('createSettler settles exactly as the evaluator, on random policies and findings', () => {
  let seed = 11;
  const rand = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 2 ** 32; };
  const model = { heads: Object.fromEntries(HEADS.map((h) => [h, { threshold: 0.6, reviewFloor: 0.3 }])) };
  let compared = 0;
  for (let n = 0; n < 2000; n++) {
    const priority = [...HEADS].sort(() => rand() - 0.5);
    const suppress = rand() < 0.5 ? [] : [{ when: [priority[2]], heads: [priority[Math.floor(rand() * 2)]] }];
    const policy = { ...POLICY, priority, suppress, onAbstain: rand() < 0.5 ? 'none' as const : 'review' as const };
    const fired = rand() < 0.5 ? { id: 'r', label: HEADS[Math.floor(rand() * 3)] } : null;
    const dismissed = HEADS.filter((h) => h !== fired?.label && rand() < 0.4);
    assert.deepEqual(createSettler(policy).settle({ fired, dismissed }), createDecisionEvaluator(model, policy).settle({ fired, dismissed }));
    compared++;
  }
  assert.equal(compared, 2000);
});
