/**
 * Drift checks on router telemetry, and a retraining trigger. The host stores the events a router (or
 * both tiers of a split deployment) emits and runs a check per window (an hour, a day, N requests):
 * `windowOf` folds a window's events, and `checkWindow` compares it with a reference window - the
 * release's own traffic, routed by the same release.
 *
 * Scores come in two strata, kept apart so no inverse weighting is needed: requests the router always
 * scores (decision events with calibrated probabilities) and the monitoring sample of rules-settled
 * requests (diagnostic events). Each stratum is compared per head with embedding-classifier's
 * `monitorWindow`. What the rules tier settles never reaches the scores, so its decision rates are
 * compared too: the share settled without the model, each label's rule firing, the edge tier's fires and
 * dismissals per label, and unavailable results. Without the edge's own rates, a surge in one label's
 * traffic that the edge settles raises no alert at all (ABCD, lab). All checks share one family-wise
 * false-alarm budget (Bonferroni).
 */
import { monitorWindow } from '@liquidau/embedding-classifier';
import { binomialCdf, clopperPearsonLower, clopperPearsonUpper, exceedanceTest } from '@liquidau/solvers';

import type { RouterEvent } from './events.ts';

/** One window of telemetry, folded by `windowOf`. */
export interface TelemetryWindow {
  requests: number;
  /** Calibrated probabilities per head: requests the router scored to decide them. */
  scored: Record<string, number[]>;
  /** Calibrated probabilities per head: the monitoring sample of rules-settled requests. */
  sampled: Record<string, number[]>;
  /** Requests settled by the rules tier (rules and edge) without the main encoder. */
  settled: number;
  /** Requests whose firing rule names each label. */
  fired: Record<string, number>;
  /** Requests the edge tier fired, and dismissed, for each label. */
  edgeFired: Record<string, number>;
  edgeDismissed: Record<string, number>;
  /** Whether any event carried edge decisions (a release with an edge tier). */
  hasEdge: boolean;
  unavailable: number;
  /** Requests routed to each head's route, by any tier (for routes whose id is the head's name). */
  routed: Record<string, number>;
}

/**
 * Folds router events into a window. `labelOf` maps a rule id to its label (from the release's rule
 * set); events from other releases should be filtered out first.
 */
export function windowOf(events: readonly RouterEvent[], heads: readonly string[], labelOf: (ruleId: string) => string | undefined): TelemetryWindow {
  const zero = () => Object.fromEntries(heads.map((h) => [h, 0]));
  const w: TelemetryWindow = { requests: 0, scored: Object.fromEntries(heads.map((h) => [h, []])), sampled: Object.fromEntries(heads.map((h) => [h, []])), settled: 0, fired: zero(), edgeFired: zero(), edgeDismissed: zero(), hasEdge: false, unavailable: 0, routed: zero() };
  for (const e of events) {
    if (e.type === 'decision') {
      w.requests++;
      if (e.outcome === 'unavailable') { w.unavailable++; continue; }
      if (e.rulesSettled) w.settled++;
      if (e.outcome === 'route' && e.routeId !== null && e.routeId in w.routed) w.routed[e.routeId]++;
      const label = e.firedRule && !e.firedRule.startsWith('edge:') ? labelOf(e.firedRule) : undefined;
      if (label !== undefined && label in w.fired) w.fired[label]++;
      if (e.edge) {
        w.hasEdge = true;
        for (const h of e.edge.fired) if (h in w.edgeFired) w.edgeFired[h]++;
        for (const h of e.edge.dismissed) if (h in w.edgeDismissed) w.edgeDismissed[h]++;
      }
      if (e.calibratedProbabilities) for (const h of heads) if (e.calibratedProbabilities[h] !== undefined) w.scored[h].push(e.calibratedProbabilities[h]);
    } else if (e.type === 'diagnostic') {
      for (const h of heads) if (e.calibratedProbabilities[h] !== undefined) w.sampled[h].push(e.calibratedProbabilities[h]);
    }
  }
  return w;
}

export interface WindowCheck {
  name: string;
  alert: boolean;
  pValue: number;
  detail: string;
  /** What an alert means: the release no longer fits the traffic, or the service is failing. */
  kind: 'drift' | 'operational';
}

export interface WindowReport {
  alert: boolean;
  checks: WindowCheck[];
  /** Retrain (label a fresh sample, rebuild the release) when the traffic has drifted. */
  retrain: { recommended: boolean; reasons: string[] };
  /** Investigate the service (encoder, source, observer) when it fails; retraining won't help. */
  investigate: string[];
  /** Score strata with too few scores to test, live or reference. */
  skipped: string[];
}

/** A rate against the reference's rate: up and down tests, each at a/2. */
function rateShift(name: string, k: number, n: number, kr: number, nr: number, a: number, kind: WindowCheck['kind'], upOnly = false): WindowCheck[] {
  if (n === 0 || nr === 0) return [];
  const upper = clopperPearsonUpper(kr, nr, 1 - a / 4);
  const up = exceedanceTest(k, n, upper, a / 4);
  const out: WindowCheck[] = [{ name: `${name}-up`, alert: up.alert, pValue: up.pValue, kind, detail: `${k}/${n} vs reference ${kr}/${nr} (upper bound ${upper.toFixed(4)})` }];
  if (!upOnly) {
    const lower = clopperPearsonLower(kr, nr, 1 - a / 4);
    const p = binomialCdf(k, n, lower);
    out.push({ name: `${name}-down`, alert: p <= a / 4, pValue: p, kind, detail: `${k}/${n} vs reference ${kr}/${nr} (lower bound ${lower.toFixed(4)})` });
  }
  return out;
}

/**
 * Checks a live window against the reference. `heads` gives each head's threshold and review floor (from
 * the release's classifier). Strata with fewer than `minScores` scores (live or reference) are skipped
 * and listed; the family-wise false-alarm rate is `alpha` (default 0.01) over the checks run. A drift
 * alert recommends retraining; an alert on unavailable results says to investigate the service instead.
 */
export function checkWindow(reference: TelemetryWindow, live: TelemetryWindow, heads: Readonly<Record<string, { threshold: number; review_floor: number }>>, options: { alpha?: number; minScores?: number } = {}): WindowReport {
  const alpha = options.alpha ?? 0.01, minScores = options.minScores ?? 30;
  if (!(alpha > 0 && alpha < 1)) throw new RangeError('checkWindow: alpha must be in (0, 1)');
  const names = Object.keys(heads);
  const strata: Array<{ head: string; stratum: 'scored' | 'sampled' }> = [];
  const skipped: string[] = [];
  for (const head of names) for (const stratum of ['scored', 'sampled'] as const) {
    const nl = live[stratum][head]?.length ?? 0, nr = reference[stratum][head]?.length ?? 0;
    if (nl >= minScores && nr >= minScores) strata.push({ head, stratum });
    else skipped.push(`${head} (${stratum}): ${nl} live, ${nr} reference scores`);
  }
  // One budget: score strata, then settlement, each label's firing (rules, and the edge's fires and
  // dismissals when the release has an edge), and unavailability.
  const edge = reference.hasEdge || live.hasEdge;
  const families = strata.length + 1 + names.length * (edge ? 3 : 1) + 1;
  const a = alpha / families;
  const checks: WindowCheck[] = [];
  for (const { head, stratum } of strata) {
    const r = monitorWindow({ spec: heads[head]!, reference: reference[stratum][head]!, live: live[stratum][head]!, alpha: a });
    for (const c of r.checks) checks.push({ name: `${head} ${stratum} ${c.test}`, alert: c.alert, pValue: c.pValue, kind: 'drift', detail: c.detail });
  }
  const ok = live.requests - live.unavailable, okRef = reference.requests - reference.unavailable;
  checks.push(...rateShift('settled by the rules', live.settled, ok, reference.settled, okRef, a, 'drift'));
  for (const h of names) checks.push(...rateShift(`${h} rule firing`, live.fired[h] ?? 0, ok, reference.fired[h] ?? 0, okRef, a, 'drift'));
  if (edge) for (const h of names) {
    checks.push(...rateShift(`${h} edge firing`, live.edgeFired[h] ?? 0, ok, reference.edgeFired[h] ?? 0, okRef, a, 'drift'));
    checks.push(...rateShift(`${h} edge dismissal`, live.edgeDismissed[h] ?? 0, ok, reference.edgeDismissed[h] ?? 0, okRef, a, 'drift'));
  }
  checks.push(...rateShift('unavailable', live.unavailable, live.requests, reference.unavailable, reference.requests, a, 'operational', true));
  const drift = checks.filter((c) => c.alert && c.kind === 'drift');
  return {
    alert: checks.some((c) => c.alert), checks,
    retrain: { recommended: drift.length > 0, reasons: drift.map((c) => `${c.name}: ${c.detail}`) },
    investigate: checks.filter((c) => c.alert && c.kind === 'operational').map((c) => `${c.name}: ${c.detail}`),
    skipped,
  };
}
