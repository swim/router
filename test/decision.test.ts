import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createDecisionEvaluator, ScoreError, type Candidate, type DecisionModel, type RouterPolicy, type RuleFindings } from '../src/index.ts';

type H = 'a' | 'b' | 'c';
const HEADS: H[] = ['a', 'b', 'c'];

/**
 * An oracle written from the specification's decision semantics (section 9), structured independently
 * of the classifier's decide/settleWithRules, so one defect can't be copied into both.
 */
function oracle(model: DecisionModel, policy: RouterPolicy, rules: RuleFindings, scores: Record<string, number>): Candidate {
  const off = new Set(rules.dismissed);
  const fired = rules.fired?.label ?? null; // the matcher never reports a fired head as dismissed
  const reaches = (h: string, level: 'threshold' | 'reviewFloor') => (h === fired ? true : off.has(h) ? false : scores[h] >= model.heads[h][level]);
  const blocked = new Set<string>();
  for (const s of policy.suppress) if (s.when.some((w) => reaches(w, 'reviewFloor'))) for (const h of s.heads) blocked.add(h);
  for (const h of policy.priority) {
    if (blocked.has(h) || !reaches(h, 'threshold')) continue;
    return { outcome: 'route', routeId: h, destination: `dest:${h}`, mechanism: h === fired ? 'rule' : 'classifier' };
  }
  if (policy.priority.some((h) => reaches(h, 'reviewFloor'))) return { outcome: 'review', reason: 'near_threshold' };
  if (policy.onAbstain === 'review') return { outcome: 'review', reason: 'abstention_policy' };
  return { outcome: 'abstain', reason: policy.priority.every((h) => off.has(h)) ? 'all_dismissed' : 'no_match' };
}

const policy = (priority: H[], suppress: RouterPolicy['suppress'], onAbstain: RouterPolicy['onAbstain'] = 'none'): RouterPolicy => ({
  schema: 'liquidau-router-policy/1', routes: HEADS.map((h) => ({ id: h, destination: `dest:${h}` })), priority, suppress, onAbstain,
});

const MODELS: Record<string, DecisionModel> = {
  ordinary: { heads: { a: { threshold: 0.8, reviewFloor: 0.3 }, b: { threshold: 0.7, reviewFloor: 0.4 }, c: { threshold: 0.6, reviewFloor: 0.5 } } },
  // b disabled (threshold just above 1), c without a review band, a with floor 0.
  edges: { heads: { a: { threshold: 0.5, reviewFloor: 0 }, b: { threshold: 1.0000000000000002, reviewFloor: 0.9 }, c: { threshold: 0.5, reviewFloor: 0.5 } } },
};
const POLICIES: Record<string, RouterPolicy> = {
  plain: policy(['a', 'b', 'c'], []),
  guarded: policy(['a', 'b', 'c'], [{ when: ['a'], heads: ['c'] }]),
  twoGuards: policy(['c', 'b', 'a'], [{ when: ['a', 'b'], heads: ['c'] }], 'review'),
  guardedFirst: policy(['b', 'a', 'c'], [{ when: ['c'], heads: ['a'] }, { when: ['c'], heads: ['b'] }]),
};

/** Scores below, at and above each boundary, plus 0 and 1. */
function grid(m: DecisionModel['heads'][string]): number[] {
  const v = [0, 1, m.reviewFloor, m.threshold, (m.reviewFloor + m.threshold) / 2, m.reviewFloor - 1e-9, m.threshold - 1e-9, m.threshold + 1e-9, m.reviewFloor + 1e-9];
  return [...new Set(v.filter((x) => x >= 0 && x <= 1))];
}

function* findings(): Generator<RuleFindings> {
  for (const fired of [null, ...HEADS]) {
    for (let mask = 0; mask < 8; mask++) {
      if (fired && mask & (1 << HEADS.indexOf(fired))) continue; // conflicting: rejected, tested below
      yield { fired: fired ? { id: `rule-${fired}`, label: fired } : null, dismissed: HEADS.filter((_, i) => mask & (1 << i)) };
    }
  }
}

test('reference decisions match the oracle and settled decisions match the reference for every score', () => {
  for (const [mn, model] of Object.entries(MODELS)) {
    for (const [pn, pol] of Object.entries(POLICIES)) {
      const ev = createDecisionEvaluator(model, pol);
      let settled = 0, cases = 0;
      for (const rules of findings()) {
        const s = ev.settle(rules);
        if (s) settled++;
        for (const sa of grid(model.heads.a)) for (const sb of grid(model.heads.b)) for (const sc of grid(model.heads.c)) {
          const scores = { a: sa, b: sb, c: sc };
          const expected = oracle(model, pol, rules, scores);
          const where = `${mn}/${pn} rules=${JSON.stringify(rules)} scores=${JSON.stringify(scores)}`;
          assert.deepEqual(ev.decide(rules, scores), expected, where);
          if (s) assert.deepEqual(s, expected, `settled ${where}`);
          cases++;
        }
      }
      assert.ok(settled > 0, `${mn}/${pn}: the rules tier settles something`);
      assert.ok(cases > 1000);
    }
  }
});

test('settlement: a top-priority unsuppressible rule and full dismissal settle; a suppressible rule does not', () => {
  const ev = createDecisionEvaluator(MODELS.ordinary, POLICIES.guarded);
  assert.deepEqual(ev.settle({ fired: { id: 'r', label: 'a' }, dismissed: [] }), { outcome: 'route', routeId: 'a', destination: 'dest:a', mechanism: 'rule' });
  assert.deepEqual(ev.settle({ fired: null, dismissed: ['a', 'b', 'c'] }), { outcome: 'abstain', reason: 'all_dismissed' });
  assert.equal(ev.settle({ fired: { id: 'r', label: 'c' }, dismissed: [] }), null, 'a can suppress c');
  assert.deepEqual(ev.settle({ fired: { id: 'r', label: 'c' }, dismissed: ['a', 'b'] }), { outcome: 'route', routeId: 'c', destination: 'dest:c', mechanism: 'rule' });
  assert.equal(ev.settle({ fired: null, dismissed: [] }), null);
});

test('a firing rule acts on a disabled head; its probability is never replaced by the threshold', () => {
  const ev = createDecisionEvaluator(MODELS.edges, POLICIES.plain);
  const rules = { fired: { id: 'r', label: 'b' }, dismissed: [] };
  assert.deepEqual(ev.decide(rules, { a: 0.1, b: 0.2, c: 0.1 }), { outcome: 'route', routeId: 'b', destination: 'dest:b', mechanism: 'rule' });
  assert.deepEqual(ev.decide({ fired: null, dismissed: [] }, { a: 0.1, b: 1, c: 0.1 }), { outcome: 'review', reason: 'near_threshold' }, 'a disabled head never fires on its score');
});

test('scores: missing or invalid probabilities for unresolved heads are errors; dismissed and fired heads need none', () => {
  const ev = createDecisionEvaluator(MODELS.ordinary, POLICIES.plain);
  const none = { fired: null, dismissed: [] };
  for (const bad of [undefined, Number.NaN, -0.01, 1.01, Infinity]) {
    assert.throws(() => ev.decide(none, { a: 0.1, b: bad as number, c: 0.1 }), ScoreError);
  }
  assert.deepEqual(ev.requiredScores({ fired: { id: 'r', label: 'a' }, dismissed: ['b'] }), ['c']);
  assert.deepEqual(ev.decide({ fired: { id: 'r', label: 'b' }, dismissed: ['a'] }, { c: 0.1 }), { outcome: 'route', routeId: 'b', destination: 'dest:b', mechanism: 'rule' });
});

test('findings that fire and dismiss the same head are rejected on every path (review R4)', async () => {
  const ev = createDecisionEvaluator(MODELS.ordinary, POLICIES.plain);
  const conflict = { fired: { id: 'r', label: 'a' }, dismissed: ['a'] };
  assert.throws(() => ev.settle(conflict), /conflicting findings/);
  assert.throws(() => ev.decide(conflict, { a: 0.9, b: 0.1, c: 0.1 }), /conflicting findings/);
  assert.throws(() => ev.requiredScores(conflict), /conflicting findings/);
  const { evaluatePerHead } = await import('../src/evaluation/index.ts');
  assert.throws(() => evaluatePerHead(MODELS.ordinary, [{ rules: conflict, scores: { a: 0.9, b: 0.1, c: 0.1 }, labels: {} }]), /conflicting findings/);
});

test('labels outside the policy are programmer errors; the evaluator copies its inputs', () => {
  const pol = policy(['a', 'b', 'c'], []);
  const model: DecisionModel = JSON.parse(JSON.stringify(MODELS.ordinary));
  const ev = createDecisionEvaluator(model, pol);
  assert.throws(() => ev.decide({ fired: { id: 'r', label: 'zzz' }, dismissed: [] }, {}), /not a route/);
  pol.priority.reverse();
  (model.heads.a as { threshold: number }).threshold = 0;
  assert.deepEqual(ev.priority, ['a', 'b', 'c']);
  assert.equal(ev.decide({ fired: null, dismissed: [] }, { a: 0.1, b: 0.1, c: 0.1 }).outcome, 'abstain');
  assert.throws(() => createDecisionEvaluator({ heads: { ...MODELS.ordinary.heads, a: { threshold: 0.2, reviewFloor: 0.3 } } }, pol), /reviewFloor/);
});
