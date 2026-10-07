import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createDecisionEvaluator, type RouterPolicy } from '../src/index.ts';
import { acceptanceExitCode, buildEvidence, checkAcceptance, evaluateFinal, evaluatePerHead, rate, type FinalItem } from '../src/evaluation/index.ts';

const policy: RouterPolicy = {
  schema: 'liquidau-router-policy/1', routes: [{ id: 'a', destination: 'A' }, { id: 'b', destination: 'B' }], priority: ['a', 'b'], suppress: [], onAbstain: 'none',
};
const model = { heads: { a: { threshold: 0.5, reviewFloor: 0.3 }, b: { threshold: 0.5, reviewFloor: 0.3 } } };
const none = { fired: null, dismissed: [] as string[] };
const items: FinalItem[] = [
  { id: '1', rules: none, scores: { a: 0.9, b: 0.9 }, acceptable: ['a', 'b'], slice: 'short' },
  { id: '2', rules: none, scores: { a: 0.1, b: 0.9 }, acceptable: ['b'], slice: 'short' },
  { id: '3', rules: none, scores: { a: 0.4, b: 0.1 }, acceptable: ['a'], slice: 'long' },
  { id: '4', rules: { fired: { id: 'r', label: 'a' }, dismissed: [] }, scores: { a: 0.1, b: 0.1 }, acceptable: [], slice: 'long' },
  { id: '5', rules: { fired: null, dismissed: ['a', 'b'] }, scores: { a: 0.9, b: 0.9 }, acceptable: [] },
];

test('final routing and per-head behaviour are reported separately', () => {
  const ev = createDecisionEvaluator(model, policy);
  const final = evaluateFinal(ev, items);
  assert.deepEqual(final.outcomes, { route: 3, review: 1, abstain: 1 });
  assert.deepEqual(final.mechanism, { rule: 1, classifier: 2 });
  assert.equal(final.routedAccuracy.count, 2);
  assert.equal(final.falseRouteRate.rate, 0.5);
  assert.deepEqual(final.pathMismatches, []);
  assert.equal(final.rulesSettlementRate.count, 2);
  // Item 1 accepts a or b and went to a: correct, but b's recall still counts it as a miss.
  assert.deepEqual([final.perRoute.b.recall.count, final.perRoute.b.recall.of], [1, 2]);
  assert.equal(final.confusion['a+b']['route:a'], 1);
  assert.equal(final.slices.short.items, 2);

  const perHead = evaluatePerHead(model, items.map((it) => ({ rules: it.rules, scores: it.scores, labels: { a: it.acceptable.includes('a') ? 1 : 0, b: it.acceptable.includes('b') ? 1 : 0 } })));
  assert.equal(perHead.kind, 'per-head');
  assert.deepEqual([perHead.heads.b.recall.count, perHead.heads.b.recall.of], [2, 2], 'per head, b fired on both of its positives');
  assert.equal(perHead.heads.a.ruleOnlyPositives, 1);
  assert.equal(perHead.heads.a.dismissed, 1);
});

test('acceptance: a known failure exits non-zero; no evidence never passes; diagnostics-only is explicit', () => {
  const final = evaluateFinal(createDecisionEvaluator(model, policy), items);
  const failing = checkAcceptance(final, null, { minCoverage: 0.9 });
  assert.equal(failing.passed, false);
  assert.equal(acceptanceExitCode(failing), 1);
  assert.equal(acceptanceExitCode(failing, { diagnosticsOnly: true }), 0);
  assert.equal(acceptanceExitCode(checkAcceptance(final, null, { minCoverage: 0.5 })), 0);
  assert.equal(checkAcceptance(null, null, { minCoverage: 0 }).passed, false);
  assert.match(checkAcceptance(evaluateFinal(createDecisionEvaluator(model, policy), []), null, { minCoverage: 0 }).failures.join(), /fewer than 1/);
  assert.match(checkAcceptance(final, null, { heads: { a: { minRecall: 0.5 } } }).failures.join(), /no per-head evaluation/);
  assert.match(checkAcceptance(final, null, { routes: { a: { minPrecision: 0.4 } }, compare: 'wilson-lower' }).failures.join(), /route a precision/);
});

test('evidence records inputs, gates and claim kinds; diagnostics-only evidence is marked', () => {
  const final = evaluateFinal(createDecisionEvaluator(model, policy), items);
  const inputs = { classifierSha256: '1'.repeat(64), rulesSha256: '2'.repeat(64), policySha256: '3'.repeat(64), rulesSemanticHash: '4'.repeat(64) };
  const criteria = { minCoverage: 0.5 };
  const e = buildEvidence({ inputs, dataset: { id: 'held-out', split: 'test' }, final, criteria, diagnosticsOnly: true });
  assert.deepEqual(e.criteria, criteria);
  assert.deepEqual(e.gates, checkAcceptance(final, null, criteria));
  assert.equal(e.finalDataset, 'held-out');
  assert.equal(e.datasets[0].items, 5);
  assert.ok(e.claims.every((c) => c.kind === 'empirical' && c.scope === 'final-routing'));
  assert.equal(e.diagnosticsOnly, true);
});

test('Wilson intervals', () => {
  assert.deepEqual(rate(0, 0), { count: 0, of: 0, rate: null, wilson95: null });
  const r = rate(9, 10);
  assert.ok(r.wilson95![0] > 0.59 && r.wilson95![0] < 0.6 && r.wilson95![1] > 0.98);
});
