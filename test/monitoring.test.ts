/**
 * Drift checks on telemetry windows: quiet on unchanged traffic, the edge's own rates catch a surge the
 * scores never see, an encoder outage asks for investigation rather than retraining, and thin score
 * strata are skipped and listed.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { checkWindow, windowOf, type DecisionEvent, type RouterEvent } from '../src/index.ts';

const HEADS = { urgent: { threshold: 0.5, review_floor: 0.3 }, refund: { threshold: 0.5, review_floor: 0.3 } };
const NAMES = Object.keys(HEADS);
const labelOf = (id: string) => ({ 'rule-urgent': 'urgent', 'rule-refund': 'refund' } as Record<string, string>)[id];

/** A deterministic score in [0, 1) for request i. */
const score = (i: number, salt: number) => ((i * 7919 + salt * 104729) % 1000) / 1000;

function decision(i: number, over: Partial<DecisionEvent> = {}): DecisionEvent {
  return {
    type: 'decision', releaseId: 'r1', requestId: `q${i}`, mode: 'enforce', samplingProbability: 0.05,
    outcome: 'abstain', routeId: null, mechanism: null, reason: null, rulesSettled: false, sampled: false, embedded: true,
    inclusionProbability: 1, firedRule: null, dismissed: [], errorCode: null,
    timings: { totalMs: 1, rulesMs: 0, embedMs: 1, scoreMs: 0 },
    calibratedProbabilities: { urgent: score(i, 1), refund: score(i, 2) },
    ...over,
  };
}

/**
 * A window of n requests: every tenth is settled by the edge firing `urgent` (or, with `surge`, every
 * third), every fifth is dismissed by the edge, and every seventh fails when `failing`.
 */
function traffic(n: number, opts: { surge?: boolean; failing?: boolean; offset?: number } = {}): RouterEvent[] {
  const out: RouterEvent[] = [];
  for (let k = 0; k < n; k++) {
    const i = k + (opts.offset ?? 0);
    if (opts.failing && k % 7 === 0) { out.push(decision(i, { outcome: 'unavailable', errorCode: 'ENCODER_FAILURE', calibratedProbabilities: undefined, embedded: false })); continue; }
    if (k % (opts.surge ? 3 : 10) === 0) {
      out.push(decision(i, { outcome: 'route', routeId: 'urgent', mechanism: 'edge', rulesSettled: true, embedded: false, firedRule: 'edge:urgent', calibratedProbabilities: undefined, edge: { fired: ['urgent'], dismissed: [] } }));
      continue;
    }
    out.push(decision(i, { edge: { fired: [], dismissed: k % 5 === 0 ? ['refund'] : [] } }));
  }
  return out;
}

test('windowOf folds decisions, edge findings, routes and unavailability', () => {
  const w = windowOf([...traffic(100, { failing: true }), { type: 'diagnostic', kind: 'monitoring_sample', releaseId: 'r1', requestId: 'd', mode: 'enforce', samplingProbability: 0.05, inclusionProbability: 0.05, classifierCandidate: { outcome: 'abstain', routeId: null }, calibratedProbabilities: { urgent: 0.9, refund: 0.1 }, timings: { embedMs: 1, scoreMs: 0 } }], NAMES, labelOf);
  assert.equal(w.requests, 100);
  assert.equal(w.unavailable, 15);
  assert.ok(w.hasEdge);
  assert.equal(w.edgeFired.urgent, w.routed.urgent);
  assert.equal(w.settled, w.routed.urgent);
  assert.equal(w.fired.urgent, 0, 'edge fires are not rule firings');
  assert.deepEqual(w.sampled.urgent, [0.9]);
  assert.equal(w.scored.urgent.length, 100 - 15 - w.settled);
});

test('unchanged traffic raises no alert', () => {
  const r = checkWindow(windowOf(traffic(4000), NAMES, labelOf), windowOf(traffic(1000, { offset: 4000 }), NAMES, labelOf), HEADS);
  assert.equal(r.alert, false, r.checks.filter((c) => c.alert).map((c) => c.name).join(', '));
  assert.equal(r.retrain.recommended, false);
});

test('a surge the edge settles is caught by the edge rates, and asks for retraining', () => {
  const r = checkWindow(windowOf(traffic(4000), NAMES, labelOf), windowOf(traffic(1000, { surge: true, offset: 4000 }), NAMES, labelOf), HEADS);
  const names = r.checks.filter((c) => c.alert).map((c) => c.name);
  assert.ok(names.includes('urgent edge firing-up'), names.join(', '));
  assert.ok(r.retrain.recommended);
  assert.deepEqual(r.investigate, []);
});

test('an encoder outage asks for investigation, not retraining', () => {
  const r = checkWindow(windowOf(traffic(4000), NAMES, labelOf), windowOf(traffic(1000, { failing: true, offset: 4000 }), NAMES, labelOf), HEADS);
  assert.ok(r.investigate.some((x) => x.startsWith('unavailable-up')), r.investigate.join('; '));
  assert.ok(!r.checks.some((c) => c.alert && c.kind === 'drift'), r.retrain.reasons.join('; '));
});

test('thin score strata are skipped and listed; alpha is validated', () => {
  const r = checkWindow(windowOf(traffic(4000), NAMES, labelOf), windowOf(traffic(1000, { offset: 4000 }), NAMES, labelOf), HEADS);
  assert.ok(r.skipped.some((s) => s.startsWith('urgent (sampled): 0 live')));
  assert.throws(() => checkWindow(windowOf([], NAMES, labelOf), windowOf([], NAMES, labelOf), HEADS, { alpha: 0 }), RangeError);
});
