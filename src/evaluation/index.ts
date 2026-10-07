/**
 * Offline evaluation, pure and platform-neutral (file I/O and CLI entry points live in the application):
 *
 *   evaluatePerHead   each head's positives under "rules OR classifier" (fired rule, dismissal, threshold),
 *                     for comparison with the classifier's stated per-head targets
 *   evaluateFinal     the routed outcome after rules, dismissals, priority and suppression: coverage,
 *                     review and abstention rates, confusion matrix, per-route precision/recall, rule
 *                     firing and settlement, slices - plus a check that the rules-settled path agrees
 *                     with the always-embed reference on every item
 *   checkAcceptance   configured criteria -> gates; no items is a failure (missing evidence can't pass)
 *   buildEvidence     the release evidence document for buildRelease
 *
 * Every number here is an empirical point estimate on the given items (Wilson intervals are reported
 * for rates); none is a guarantee. Ground truth for the final decision is explicit: each item lists the
 * routes that would be ACCEPTABLE. Routing to any one of them is correct for the item, but per-route
 * recall still counts the item against every acceptable route it was not sent to - choosing one of
 * several true labels does not satisfy each label's recall.
 */
import type { Candidate, DecisionEvaluator, DecisionModel, RuleFindings } from '../decision.ts';
import type { ClaimKind, EvidenceClaim, ReleaseEvidence } from '../evidence.ts';
import type { ReleaseGates } from '../manifest.ts';

export interface Rate {
  count: number;
  of: number;
  /** count / of, or null when of = 0. */
  rate: number | null;
  /** 95% Wilson score interval, or null when of = 0. */
  wilson95: [number, number] | null;
}

export function rate(count: number, of: number): Rate {
  if (of === 0) return { count, of, rate: null, wilson95: null };
  const z = 1.959963984540054, p = count / of, z2 = z * z;
  const centre = (p + z2 / (2 * of)) / (1 + z2 / of);
  const half = (z / (1 + z2 / of)) * Math.sqrt((p * (1 - p)) / of + z2 / (4 * of * of));
  return { count, of, rate: p, wilson95: [Math.max(0, centre - half), Math.min(1, centre + half)] };
}

// ---------- per head ----------

export interface PerHeadItem {
  rules: RuleFindings;
  /** Calibrated probabilities from the classifier (always-embed). */
  scores: Readonly<Record<string, number>>;
  /** Per-head labels: 1 positive, 0 negative, null or absent unlabelled. */
  labels: Readonly<Record<string, 0 | 1 | null | undefined>>;
}

export interface HeadReport {
  labelled: number;
  positives: number;
  recall: Rate;
  precision: Rate;
  falseAlarmRate: Rate;
  /** Items whose first firing rule named this head. */
  ruleFired: number;
  /** Items where a dismissal rule cleared this head. */
  dismissed: number;
  /** Positive decisions made by the rule alone (the classifier was below threshold). */
  ruleOnlyPositives: number;
}

export interface PerHeadReport {
  kind: 'per-head';
  bounds: 'empirical';
  items: number;
  heads: Record<string, HeadReport>;
}

/**
 * Each head's own positive decision, before priority and suppression: a fired rule (not dismissed), or
 * a probability at/above the threshold (not dismissed). This is what per-head targets describe; it is
 * NOT the final routing decision.
 */
export function evaluatePerHead(model: DecisionModel, items: readonly PerHeadItem[]): PerHeadReport {
  const heads: Record<string, HeadReport> = {};
  for (const [h, proj] of Object.entries(model.heads)) {
    let tp = 0, fp = 0, fn = 0, tn = 0, ruleFired = 0, dismissed = 0, ruleOnly = 0;
    for (const it of items) {
      if (it.rules.fired && it.rules.dismissed.includes(it.rules.fired.label)) throw new Error(`conflicting findings: ${it.rules.fired.label} both fires and is dismissed (rule-miner's matcher never reports both)`);
      const off = it.rules.dismissed.includes(h);
      const fired = it.rules.fired?.label === h && !off;
      const score = it.scores[h];
      if (!off && !fired && (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)) throw new Error(`item has no valid probability for head ${h}`);
      const byModel = !off && typeof score === 'number' && score >= proj.threshold;
      const positive = fired || byModel;
      if (it.rules.fired?.label === h) ruleFired++;
      if (off) dismissed++;
      if (fired && !byModel) ruleOnly++;
      const y = it.labels[h];
      if (y === 1) positive ? tp++ : fn++;
      else if (y === 0) positive ? fp++ : tn++;
    }
    heads[h] = {
      labelled: tp + fp + fn + tn, positives: tp + fn, recall: rate(tp, tp + fn), precision: rate(tp, tp + fp), falseAlarmRate: rate(fp, fp + tn),
      ruleFired, dismissed, ruleOnlyPositives: ruleOnly,
    };
  }
  return { kind: 'per-head', bounds: 'empirical', items: items.length, heads };
}

// ---------- final routing ----------

export interface FinalItem {
  id: string;
  rules: RuleFindings;
  /** Always-embed calibrated probabilities (the reference decision needs them). */
  scores: Readonly<Record<string, number>>;
  /** Routes that would be acceptable for this item; empty: it should not be routed. */
  acceptable: readonly string[];
  /** Optional slice name for performance slices. */
  slice?: string;
}

export interface RouteReport {
  routed: number;
  /** Routed here and acceptable. */
  precision: Rate;
  /** Of items listing this route as acceptable, how many were routed here. */
  recall: Rate;
}

export interface FinalReport {
  kind: 'final-routing';
  bounds: 'empirical';
  items: number;
  outcomes: { route: number; review: number; abstain: number };
  coverage: Rate;
  reviewRate: Rate;
  abstainRate: Rate;
  /** Of routed items, the share routed to an acceptable route. */
  routedAccuracy: Rate;
  /** Of items that should not be routed, the share routed anyway. */
  falseRouteRate: Rate;
  mechanism: { rule: number; classifier: number };
  /** Items where some firing rule matched. */
  ruleFiringRate: Rate;
  /** Items the rules tier settled without scores. */
  rulesSettlementRate: Rate;
  /** Items where the rules-settled decision differed from the always-embed reference (must be 0). */
  pathMismatches: string[];
  perRoute: Record<string, RouteReport>;
  /** truth key ('a+b' or '(none)') -> predicted ('route:x', 'review', 'abstain') -> count. */
  confusion: Record<string, Record<string, number>>;
  slices: Record<string, { items: number; coverage: Rate; routedAccuracy: Rate; reviewRate: Rate }>;
}

const predictedKey = (c: Candidate) => (c.outcome === 'route' ? `route:${c.routeId}` : c.outcome);
const sameCandidate = (a: Candidate, b: Candidate) => JSON.stringify(a) === JSON.stringify(b);

export function evaluateFinal(evaluator: DecisionEvaluator, items: readonly FinalItem[]): FinalReport {
  const routes = evaluator.priority;
  const outcomes = { route: 0, review: 0, abstain: 0 };
  const mechanism = { rule: 0, classifier: 0 };
  let correct = 0, shouldNot = 0, routedWrongly = 0, fired = 0, settledCount = 0;
  const mismatches: string[] = [];
  const perRoute = new Map(routes.map((r) => [r, { routed: 0, correct: 0, support: 0, hit: 0 }]));
  const confusion: Record<string, Record<string, number>> = {};
  const slices = new Map<string, { items: number; routed: number; correct: number; review: number }>();
  for (const it of items) {
    for (const r of it.acceptable) if (!perRoute.has(r)) throw new Error(`item ${it.id} lists unknown route ${r}`);
    const reference = evaluator.decide(it.rules, it.scores);
    const settled = evaluator.settle(it.rules);
    if (settled) settledCount++;
    if (settled && !sameCandidate(settled, reference)) mismatches.push(it.id);
    if (it.rules.fired) fired++;
    const c = reference;
    outcomes[c.outcome]++;
    const ok = c.outcome === 'route' && it.acceptable.includes(c.routeId);
    if (c.outcome === 'route') {
      mechanism[c.mechanism]++;
      if (ok) correct++;
      const pr = perRoute.get(c.routeId)!;
      pr.routed++;
      if (ok) pr.correct++;
    }
    if (!it.acceptable.length) { shouldNot++; if (c.outcome === 'route') routedWrongly++; }
    for (const r of it.acceptable) { const pr = perRoute.get(r)!; pr.support++; if (c.outcome === 'route' && c.routeId === r) pr.hit++; }
    const truth = it.acceptable.length ? [...it.acceptable].sort().join('+') : '(none)';
    (confusion[truth] ??= {})[predictedKey(c)] = (confusion[truth][predictedKey(c)] ?? 0) + 1;
    if (it.slice !== undefined) {
      const s = slices.get(it.slice) ?? { items: 0, routed: 0, correct: 0, review: 0 };
      s.items++;
      if (c.outcome === 'route') s.routed++;
      if (ok) s.correct++;
      if (c.outcome === 'review') s.review++;
      slices.set(it.slice, s);
    }
  }
  const n = items.length;
  return {
    kind: 'final-routing', bounds: 'empirical', items: n, outcomes,
    coverage: rate(outcomes.route, n), reviewRate: rate(outcomes.review, n), abstainRate: rate(outcomes.abstain, n),
    routedAccuracy: rate(correct, outcomes.route), falseRouteRate: rate(routedWrongly, shouldNot), mechanism,
    ruleFiringRate: rate(fired, n), rulesSettlementRate: rate(settledCount, n), pathMismatches: mismatches,
    perRoute: Object.fromEntries([...perRoute].map(([r, v]) => [r, { routed: v.routed, precision: rate(v.correct, v.routed), recall: rate(v.hit, v.support) }])),
    confusion,
    slices: Object.fromEntries([...slices].map(([k, s]) => [k, { items: s.items, coverage: rate(s.routed, s.items), routedAccuracy: rate(s.correct, s.routed), reviewRate: rate(s.review, s.items) }])),
  };
}

// ---------- acceptance ----------

export interface AcceptanceCriteria {
  /** Minimum evaluated items (default 1): no evidence never passes. */
  minItems?: number;
  minCoverage?: number;
  maxReviewRate?: number;
  minRoutedAccuracy?: number;
  maxFalseRouteRate?: number;
  routes?: Record<string, { minPrecision?: number; minRecall?: number }>;
  heads?: Record<string, { minRecall?: number; minPrecision?: number; maxFalseAlarmRate?: number }>;
  /** Point estimates by default; 'wilson-lower' compares the conservative interval end instead. */
  compare?: 'point' | 'wilson-lower';
}

/** True when the criteria name at least one performance target (minItems alone gates nothing). */
export function hasPerformanceCriteria(c: AcceptanceCriteria): boolean {
  const scalar = ['minCoverage', 'maxReviewRate', 'minRoutedAccuracy', 'maxFalseRouteRate'] as const;
  const nested = (o: Record<string, Record<string, number | undefined> | undefined> | undefined) => Object.values(o ?? {}).some((v) => Object.values(v ?? {}).some((x) => x !== undefined));
  return scalar.some((k) => c[k] !== undefined) || nested(c.routes) || nested(c.heads);
}

/**
 * Gates from configured criteria. Always fails: invalid criteria, no performance criterion, an empty
 * evaluation, a measurement that is missing or not a number, and any rules-path mismatch. Pure and
 * deterministic: the loader recomputes it from the evidence's recorded criteria and reports.
 */
export function checkAcceptance(final: FinalReport | null, perHead: PerHeadReport | null, criteria: AcceptanceCriteria): ReleaseGates {
  const failures = acceptanceCriteriaProblems(criteria).map((c) => `invalid acceptance criteria: ${c}`), warnings: string[] = [];
  if (!failures.length && !hasPerformanceCriteria(criteria)) failures.push('no performance criterion is configured: acceptance would gate nothing');
  // At least one item always, whatever minItems says: an empty evaluation never passes.
  const minItems = Math.max(1, Number.isInteger(criteria?.minItems) ? criteria.minItems! : 1);
  const conservative = criteria?.compare === 'wilson-lower';
  const measured = (r: Rate | undefined, end: 0 | 1): number | null => {
    const v = conservative ? r?.wilson95?.[end] : r?.rate;
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };
  const atLeast = (name: string, r: Rate | undefined, min: number) => {
    const v = measured(r, 0);
    if (v === null) failures.push(`${name}: not measured (needs >= ${min})`);
    else if (v < min) failures.push(`${name} ${v.toFixed(4)} < ${min}`);
  };
  const atMost = (name: string, r: Rate | undefined, max: number) => {
    const v = measured(r, 1);
    if (v === null) failures.push(`${name}: not measured (needs <= ${max})`);
    else if (v > max) failures.push(`${name} ${v.toFixed(4)} > ${max}`);
  };
  if (!final) failures.push('no final routing evaluation');
  else {
    if (!(typeof final.items === 'number' && final.items >= minItems)) failures.push(`final routing evaluated ${final.items} items, fewer than ${minItems}`);
    if (!Array.isArray(final.pathMismatches) || final.pathMismatches.length) failures.push(`rules-settled decisions differ from the always-embed reference on ${final.pathMismatches?.length ?? 'unknown'} item(s)`);
    if (criteria.minCoverage !== undefined) atLeast('coverage', final.coverage, criteria.minCoverage);
    if (criteria.maxReviewRate !== undefined) atMost('review rate', final.reviewRate, criteria.maxReviewRate);
    if (criteria.minRoutedAccuracy !== undefined) atLeast('routed accuracy', final.routedAccuracy, criteria.minRoutedAccuracy);
    if (criteria.maxFalseRouteRate !== undefined) atMost('false route rate', final.falseRouteRate, criteria.maxFalseRouteRate);
    for (const [r, c] of Object.entries(criteria.routes ?? {})) {
      const rep = final.perRoute?.[r];
      if (!rep) { failures.push(`route ${r} is not in the evaluation`); continue; }
      if (c.minPrecision !== undefined) atLeast(`route ${r} precision`, rep.precision, c.minPrecision);
      if (c.minRecall !== undefined) atLeast(`route ${r} recall`, rep.recall, c.minRecall);
    }
  }
  if (criteria?.heads && Object.keys(criteria.heads).length) {
    if (!perHead) failures.push('per-head criteria are configured but there is no per-head evaluation');
    else for (const [h, c] of Object.entries(criteria.heads)) {
      const rep = perHead.heads?.[h];
      if (!rep) { failures.push(`head ${h} is not in the evaluation`); continue; }
      if (c.minRecall !== undefined) atLeast(`head ${h} recall`, rep.recall, c.minRecall);
      if (c.minPrecision !== undefined) atLeast(`head ${h} precision`, rep.precision, c.minPrecision);
      if (c.maxFalseAlarmRate !== undefined) atMost(`head ${h} false alarm rate`, rep.falseAlarmRate, c.maxFalseAlarmRate);
    }
  }
  if (!conservative) warnings.push('acceptance compares empirical point estimates, not confidence bounds');
  return { passed: failures.length === 0, failures, warnings };
}

const CRITERIA_KEYS = ['minItems', 'minCoverage', 'maxReviewRate', 'minRoutedAccuracy', 'maxFalseRouteRate', 'routes', 'heads', 'compare'];

/** Problems with acceptance criteria: known fields, minItems a positive integer, every target a finite probability, a known comparison. */
export function acceptanceCriteriaProblems(c: unknown): string[] {
  if (!c || typeof c !== 'object' || Array.isArray(c)) return ['criteria must be an object'];
  const k = c as AcceptanceCriteria & Record<string, unknown>;
  const p: string[] = [];
  for (const key of Object.keys(k)) if (!CRITERIA_KEYS.includes(key)) p.push(`unknown criterion ${key}`);
  if (k.minItems !== undefined && !(Number.isInteger(k.minItems) && k.minItems >= 1)) p.push(`minItems must be a positive integer (got ${k.minItems})`);
  if (k.compare !== undefined && k.compare !== 'point' && k.compare !== 'wilson-lower') p.push(`compare must be 'point' or 'wilson-lower' (got ${String(k.compare)})`);
  const prob = (name: string, v: unknown) => { if (v !== undefined && !(typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1)) p.push(`${name} must be a finite probability in [0, 1] (got ${String(v)})`); };
  for (const key of ['minCoverage', 'maxReviewRate', 'minRoutedAccuracy', 'maxFalseRouteRate'] as const) prob(key, k[key]);
  const nested = (what: 'routes' | 'heads', fields: string[]) => {
    const o = k[what];
    if (o === undefined) return;
    if (!o || typeof o !== 'object' || Array.isArray(o)) { p.push(`${what} must be an object`); return; }
    for (const [name, v] of Object.entries(o as Record<string, Record<string, unknown>>)) {
      if (!v || typeof v !== 'object' || Array.isArray(v)) { p.push(`${what}.${name} must be an object`); continue; }
      for (const f of Object.keys(v)) if (!fields.includes(f)) p.push(`${what}.${name} has unknown criterion ${f}`);
      for (const f of fields) prob(`${what}.${name}.${f}`, v[f]);
    }
  };
  nested('routes', ['minPrecision', 'minRecall']);
  nested('heads', ['minRecall', 'minPrecision', 'maxFalseAlarmRate']);
  return p;
}

/** Process exit code for an evaluation command: non-zero on any acceptance failure unless explicitly diagnostics-only. */
export function acceptanceExitCode(gates: ReleaseGates, options: { diagnosticsOnly?: boolean } = {}): number {
  return options.diagnosticsOnly || gates.passed ? 0 : 1;
}

// ---------- evidence ----------

export interface EvidenceDataset {
  id: string;
  split: string;
  items: number;
  description?: string;
}

export interface BuildEvidenceInput {
  /** Digests of the exact documents evaluated (documentDigest of their bytes) and the rule set's semantic hash. */
  inputs: ReleaseEvidence['inputs'];
  /** The held-out dataset the final routing (and per-head) reports were measured on. */
  dataset: { id: string; split: string; description?: string };
  /** Other datasets that caller claims cite, e.g. the calibration split behind per-head guarantees. */
  otherDatasets?: EvidenceDataset[];
  final: FinalReport;
  perHead?: PerHeadReport;
  /** Recorded in the evidence; the gates are computed from them here and recomputed by the loader. */
  criteria: AcceptanceCriteria;
  /** Additional claims (e.g. the classifier's per-head guarantees), each declaring its kind, method and dataset. */
  claims?: EvidenceClaim[];
  /** A non-gating run: the evidence records it and can't support enforcement. */
  diagnosticsOnly?: boolean;
  /**
   * For document releases ('liquidau-router-evidence/2'): the envelope's feature identity and the
   * manifest's serving input limit the evaluation ran under.
   */
  document?: { featureIdentitySha256: string; servingMaxInputUtf8Bytes: number };
}

/** The evidence document, with gates computed from the recorded criteria: empirical final-routing claims are added from the report. */
export function buildEvidence(input: BuildEvidenceInput): ReleaseEvidence {
  const empirical = (id: string, statement: string, subject?: string): EvidenceClaim => ({
    id, scope: 'final-routing', kind: 'empirical' satisfies ClaimKind, method: 'held-out point estimate with 95% Wilson interval', statement, dataset: input.dataset.id, ...(subject ? { subject } : {}),
  });
  const fmt = (r: Rate) => (r.rate === null ? 'n/a' : `${r.rate.toFixed(4)} (95% Wilson ${r.wilson95!.map((v) => v.toFixed(4)).join('-')}, ${r.count}/${r.of})`);
  const f = input.final;
  const claims: EvidenceClaim[] = [
    empirical('final.coverage', `coverage ${fmt(f.coverage)}`),
    empirical('final.review_rate', `review rate ${fmt(f.reviewRate)}`),
    empirical('final.routed_accuracy', `routed accuracy ${fmt(f.routedAccuracy)}`),
    ...Object.entries(f.perRoute).map(([r, rep]) => empirical(`final.route.${r}`, `precision ${fmt(rep.precision)}, recall ${fmt(rep.recall)}`, r)),
    ...(input.claims ?? []),
  ];
  const gates = checkAcceptance(f, input.perHead ?? null, input.criteria);
  return {
    schema: input.document ? 'liquidau-router-evidence/2' : 'liquidau-router-evidence/1',
    inputs: { ...input.inputs, ...(input.document ? { featureIdentitySha256: input.document.featureIdentitySha256, servingMaxInputUtf8Bytes: input.document.servingMaxInputUtf8Bytes } : {}) },
    datasets: [{ ...input.dataset, items: f.items }, ...(input.otherDatasets ?? []).map((d) => ({ ...d }))],
    finalDataset: input.dataset.id,
    criteria: JSON.parse(JSON.stringify(input.criteria)) as Record<string, unknown>,
    gates,
    claims,
    ...(input.perHead ? { perHead: JSON.parse(JSON.stringify(input.perHead)) as Record<string, unknown> } : {}),
    final: JSON.parse(JSON.stringify(f)) as Record<string, unknown>,
    ...(input.diagnosticsOnly ? { diagnosticsOnly: true } : {}),
  };
}
