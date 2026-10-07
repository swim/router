/** Regression tests for earlier review findings R1-R6 and S1-S5 (R4 is in decision.test.ts, S5 in routing.test.ts). */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildRelease, createDecisionEvaluator, encoderContractProblems, loadRouter, memorySource, releaseSourceContractProblems, RouterLoadError, RouterSlot, type RouterEvent, type Router } from '../src/index.ts';
import { buildEvidence, checkAcceptance, evaluateFinal, type AcceptanceCriteria, type FinalReport } from '../src/evaluation/index.ts';
import { fakeEncoder, load, makeRelease, options, POLICY, type Tweaks } from './helpers.ts';

const code = (c: string) => (e: unknown) => e instanceof RouterLoadError && e.code === c;

test('R1: an empty or inconsistent final report can never become enforceable', async () => {
  const structural: Tweaks['evidence'][] = [
    (e) => { e.final = {}; e.claims = []; delete e.perHead; },
    (e) => { delete (e.final as Record<string, unknown>).outcomes; },
    (e) => { (e.final as { outcomes: { route: number } }).outcomes.route += 1; },
    (e) => { (e.final as { items: number }).items = 2.5; },
    (e) => { e.datasets[0].items += 1; },
  ];
  for (const evidence of structural) await assert.rejects(makeRelease({ evidence }), code('EVIDENCE_INVALID'));

  // Structurally valid but unusable: shadow-only, even with passing gates.
  const insufficient: Tweaks['evidence'][] = [
    (e) => { (e.final as { pathMismatches: string[] }).pathMismatches = ['3']; },
    (e) => {
      // A well-formed evaluation of zero items.
      const evaluator = createDecisionEvaluator({ heads: { urgent: { threshold: 0.8, reviewFloor: 0.3 }, refund: { threshold: 0.7, reviewFloor: 0.4 }, other: { threshold: 0.6, reviewFloor: 0.5 } } }, POLICY);
      e.final = JSON.parse(JSON.stringify(evaluateFinal(evaluator, []))); e.datasets[0].items = 0;
    },
  ];
  for (const evidence of insufficient) {
    const built = await makeRelease({ evidence });
    assert.ok(built.enforcement.length, 'reported as not enforceable');
    await assert.rejects(loadRouter(options(built)), code('GATES_FAILED'));
    await loadRouter(options(built, { mode: 'shadow' }));
  }

  const failing = await makeRelease({ evaluationPasses: false });
  await loadRouter(options(failing, { mode: 'shadow' })); // a valid failed evaluation still loads for shadow
});

test('R2: closing the slot is terminal, including during and after reloads', async () => {
  const deferred = () => { let resolve!: (r: Router) => void; const promise = new Promise<Router>((r) => { resolve = r; }); return { promise, resolve }; };

  // Close while a replacement is still loading: the replacement is closed, never installed.
  const slot = new RouterSlot(await load());
  const pending = deferred();
  const reloading = slot.reload(() => pending.promise);
  const closing = slot.close();
  const replacement = await load();
  pending.resolve(replacement);
  assert.equal((await reloading).ok, false);
  await closing;
  for (const r of [await slot.route({ text: 'emergency', requestId: '1' }), await replacement.route({ text: 'emergency', requestId: '2' })]) {
    assert.equal(r.outcome === 'unavailable' && r.code, 'CLOSED');
  }

  // Reloads after close, and reloads queued behind one in progress, are refused.
  assert.equal((await slot.reload(() => load())).ok, false);
  const slot2 = new RouterSlot(await load());
  const first = deferred();
  const r1 = slot2.reload(() => first.promise);
  const r2 = slot2.reload(() => load());
  const closing2 = slot2.close();
  first.resolve(await load());
  assert.equal((await r1).ok, false);
  assert.equal((await r2).ok, false);
  await closing2;
  assert.equal((await slot2.route({ text: 'emergency', requestId: '3' })).actionable, false);
  assert.equal(slot2.close(), closing2, 'idempotent');

  // Close while a retired router is still draining: close waits for it.
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const slow = await load({}, { encoder: { identity: fakeEncoder().encoder.identity, embed: async () => { await gate; return [[0, 0, 0, 0]]; } } });
  const slot3 = new RouterSlot(slow);
  const inflight = slot3.route({ text: 'nothing', requestId: '4' });
  const swapping = slot3.reload(() => load());
  await new Promise((r) => setTimeout(r, 5));
  let closed = false;
  const closing3 = slot3.close().then(() => { closed = true; });
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(closed, false, 'still draining the retired router');
  release();
  await inflight;
  await Promise.all([swapping, closing3]);
  assert.equal(closed, true);
});

test('R3: the classifier-only diagnostic ignores dismissals; the settled decision is unchanged', async () => {
  const events: RouterEvent[] = [];
  const router = await load({}, { encoder: fakeEncoder({ hello: { urgent: 0.99 } }).encoder, monitoring: { sampleRate: 1, samplingSalt: 's' }, observer: (e) => { events.push(e); } });
  const r = await router.route({ text: 'hello', requestId: '1' });
  assert.deepEqual([r.outcome, r.outcome === 'abstain' && r.reason], ['abstain', 'all_dismissed']);
  await router.flush();
  const diag = events.find((e) => e.type === 'diagnostic');
  assert.deepEqual(diag?.type === 'diagnostic' && diag.classifierCandidate, { outcome: 'route', routeId: 'urgent' });
});

test('R5: no document may take the manifest key; custom keys round-trip', async () => {
  const built = await makeRelease();
  const documents = { classifier: built.files.get('classifier.json')!, rules: built.files.get('rules.json')!, policy: built.files.get('policy.json')!, evidence: built.files.get('evidence.json')! };
  const input = { releaseId: 'r', documents, embedding: built.manifest.embedding, serving: built.manifest.serving, packages: built.manifest.packages };
  for (const role of ['classifier', 'rules', 'policy', 'evidence'] as const) {
    await assert.rejects(buildRelease({ ...input, keys: { [role]: 'manifest.json' } }), code('MANIFEST_INVALID'));
  }
  const custom = await buildRelease({ ...input, keys: { classifier: 'model/classifier.json', rules: 'model/rules.json' } });
  documents.classifier.fill(0); // the result owns its bytes
  const router = await loadRouter(options(custom));
  assert.equal((await router.route({ text: 'emergency', requestId: '1' })).outcome, 'route');
});

test('R6: acceptance criteria are validated and an empty evaluation never passes', () => {
  const ev = createDecisionEvaluator({ heads: { a: { threshold: 0.5, reviewFloor: 0.3 } } }, {
    schema: 'liquidau-router-policy/1', routes: [{ id: 'a', destination: 'A' }], priority: ['a'], suppress: [], onAbstain: 'none',
  });
  const empty = evaluateFinal(ev, []);
  const one = evaluateFinal(ev, [{ id: 'x', rules: { fired: null, dismissed: [] }, scores: { a: 0.1 }, acceptable: ['a'] }]);
  for (const minItems of [0, -1, 0.5, Number.NaN]) assert.equal(checkAcceptance(empty, null, { minItems }).passed, false, `minItems ${minItems}`);
  const bad: AcceptanceCriteria[] = [
    { minCoverage: Number.NaN }, { maxReviewRate: Infinity }, { minRoutedAccuracy: 1.5 }, { routes: { a: { minRecall: Number.NaN } } },
    { heads: { a: { maxFalseAlarmRate: -0.1 } } }, { compare: 'lenient' as never },
  ];
  for (const c of bad) assert.match(checkAcceptance(one, null, c).failures.join(), /invalid acceptance criteria/, JSON.stringify(c));
  assert.equal(checkAcceptance(one, null, { minCoverage: 0 }).passed, true);
});

test('S1: recorded gates must follow from the recorded criteria and reports; no criteria is shadow-only', async () => {
  // A consistent report that fails its recorded criterion, with gates hand-set to passing.
  const handSetGates = (e: { criteria?: Record<string, unknown>; gates: unknown }) => {
    e.criteria = { ...e.criteria, minCoverage: 0.99 };
    e.gates = { passed: true, failures: [], warnings: ['acceptance compares empirical point estimates, not confidence bounds'] };
  };
  const cases: Array<[string, Tweaks['evidence'], RegExp]> = [
    ['a failing report with hand-set passing gates', handSetGates as Tweaks['evidence'], /recorded gates differ/],
    ['no recorded criteria', (e) => { delete e.criteria; }, /no acceptance criteria/],
    ['criteria with no performance target', (e) => { e.criteria = { minItems: 1 }; e.gates = checkAcceptance(e.final as unknown as FinalReport, null, { minItems: 1 }); e.gates = { passed: true, failures: [], warnings: e.gates.warnings }; }, /no performance target/],
  ];
  for (const [name, evidence, pattern] of cases) {
    const built = await makeRelease({ evidence });
    assert.match(built.enforcement.join('; '), pattern, name);
    await assert.rejects(loadRouter(options(built)), code('GATES_FAILED'), name);
    await loadRouter(options(built, { mode: 'shadow' }));
  }
  // The healthy fixture still enforces, and invalid recorded criteria are a structural failure.
  assert.deepEqual((await makeRelease()).enforcement, []);
  await assert.rejects(makeRelease({ evidence: (e) => { e.criteria = { minCoverage: 'high' }; } }), code('EVIDENCE_INVALID'));
  // A report missing a measurement the gates read is malformed, not merely failing.
  await assert.rejects(makeRelease({ evidence: (e) => { delete (e.final as Record<string, unknown>).coverage; } }), code('EVIDENCE_INVALID'));

  // checkAcceptance alone: no criteria (or minItems only) never passes, even on a good evaluation.
  const ev = createDecisionEvaluator({ heads: { a: { threshold: 0.5, reviewFloor: 0.3 } } }, {
    schema: 'liquidau-router-policy/1', routes: [{ id: 'a', destination: 'A' }], priority: ['a'], suppress: [], onAbstain: 'none',
  });
  const bad = evaluateFinal(ev, [{ id: 'x', rules: { fired: null, dismissed: [] }, scores: { a: 0.9 }, acceptable: [] }]);
  for (const c of [{}, { minItems: 1 }, { routes: {} }, { heads: { a: {} } }]) assert.match(checkAcceptance(bad, null, c).failures.join(), /no performance criterion/, JSON.stringify(c));
});

test('S2: timeouts beyond the 32-bit timer range are refused, not clamped to 1 ms', async () => {
  const built = await makeRelease();
  for (const o of [{ timeoutMs: 2 ** 31 }, { timeoutMs: 0.5 }, { monitoring: { sampleRate: 0, samplingSalt: 's', timeoutMs: 2 ** 31 } }, { observerTimeoutMs: 2 ** 31 }]) {
    await assert.rejects(loadRouter(options(built, o)), (e: unknown) => e instanceof RouterLoadError && e.code === 'OPTIONS_INVALID' && /2147483647/.test(e.message), JSON.stringify(o));
  }
  const ok = await loadRouter(options(built, { timeoutMs: 2_147_483_647 }));
  assert.equal((await ok.route({ text: 'I want my money back', requestId: '1' })).outcome, 'route');
  assert.match((await encoderContractProblems(fakeEncoder().encoder, { timeoutMs: 2 ** 31 })).join(), /2147483647/);
  assert.match((await releaseSourceContractProblems(memorySource(built.files), { present: {}, timeoutMs: 2 ** 31 })).join(), /2147483647/);
});

test('S3: a claim measured on a second dataset is recorded with its own item count', async () => {
  const built = await makeRelease({ evidence: (e) => {
    e.datasets.push({ id: 'fixture-calibration', split: 'calibration', items: 100 });
    e.claims.push({ id: 'urgent.recall', scope: 'per-head', subject: 'urgent', kind: 'approximate', method: 'conformal-pac (calibration split)', statement: 'recall >= 0.9 with 95% confidence', dataset: 'fixture-calibration' });
  } });
  assert.deepEqual(built.enforcement, []);
  await loadRouter(options(built));
  // final.items is still checked against its own dataset.
  await assert.rejects(makeRelease({ evidence: (e) => { e.datasets[0].items += 1; } }), code('EVIDENCE_INVALID'));
  await assert.rejects(makeRelease({ evidence: (e) => { e.finalDataset = 'nope'; } }), code('EVIDENCE_INVALID'));
  await assert.rejects(makeRelease({ evidence: (e) => { delete e.finalDataset; } }), code('EVIDENCE_INVALID'));
  // buildEvidence accepts the extra datasets directly.
  const ev = createDecisionEvaluator({ heads: { a: { threshold: 0.5, reviewFloor: 0.3 } } }, { schema: 'liquidau-router-policy/1', routes: [{ id: 'a', destination: 'A' }], priority: ['a'], suppress: [], onAbstain: 'none' });
  const final = evaluateFinal(ev, [{ id: 'x', rules: { fired: null, dismissed: [] }, scores: { a: 0.9 }, acceptable: ['a'] }]);
  const e = buildEvidence({ inputs: { classifierSha256: '1'.repeat(64), rulesSha256: '2'.repeat(64), policySha256: '3'.repeat(64), rulesSemanticHash: '4'.repeat(64) }, dataset: { id: 'test', split: 'test' }, otherDatasets: [{ id: 'cal', split: 'calibration', items: 50 }], final, criteria: { minCoverage: 0.5 } });
  assert.deepEqual(e.datasets.map((d) => [d.id, d.items]), [['test', 1], ['cal', 50]]);
});

test('S4: requests refused after close are visible in telemetry', async () => {
  const events: RouterEvent[] = [];
  const router = await load({}, { observer: (e) => { events.push(e); } });
  await router.close();
  const r = await router.route({ text: 'emergency', requestId: 'late' });
  assert.equal(r.outcome === 'unavailable' && r.code, 'CLOSED');
  assert.deepEqual(events.map((e) => e.type === 'decision' && [e.requestId, e.outcome, e.errorCode]), [['late', 'unavailable', 'CLOSED']]);
  await router.flush();
  assert.deepEqual(router.telemetry(), { delivered: 1, dropped: 0 });
});
