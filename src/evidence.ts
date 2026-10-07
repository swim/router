/**
 * Evaluation evidence: what was evaluated (the exact classifier, rules and policy digests and the rule
 * set's semantic hash), on which held-out data, with which outcome, and what each claim means. Claims
 * say whether they are exact, approximate, heuristic or empirical, and whether they describe one
 * classifier head or the final routing decision. The evidence never contains its own digest: the
 * manifest hashes it.
 */
import { canonicalJson, isSha256Hex } from './bytes.ts';
import { RouterLoadError } from './errors.ts';
import { acceptanceCriteriaProblems, checkAcceptance, hasPerformanceCriteria, rate, type AcceptanceCriteria, type FinalReport, type PerHeadReport, type Rate } from './evaluation/index.ts';
import { gatesProblems, type ReleaseGates } from './manifest.ts';

export const EVIDENCE_SCHEMA = 'liquidau-router-evidence/1';
/**
 * Evidence for document releases: inputs also bind the document feature identity (chunk encoder and
 * pipeline) and the manifest's serving input limit, and the acceptance criteria are required.
 */
export const EVIDENCE_SCHEMA_DOCUMENT = 'liquidau-router-evidence/2';

export const CLAIM_KINDS = ['exact', 'approximate', 'heuristic', 'empirical'] as const;
export type ClaimKind = (typeof CLAIM_KINDS)[number];

export interface EvidenceClaim {
  id: string;
  /** 'per-head': one classifier head's behaviour; 'final-routing': the decision after rules, priority and suppression. */
  scope: 'per-head' | 'final-routing';
  /** The head (per-head) or route (final-routing) the claim is about, if any. */
  subject?: string;
  kind: ClaimKind;
  /** How it was obtained, e.g. 'conformal-pac (calibration split)' or 'held-out point estimate'. */
  method: string;
  statement: string;
  /** The dataset id the claim was measured on. */
  dataset: string;
}

export interface ReleaseEvidence {
  schema: typeof EVIDENCE_SCHEMA | typeof EVIDENCE_SCHEMA_DOCUMENT;
  /** Digests of exactly what was evaluated ('/2' adds the feature identity and serving input limit). */
  inputs: { classifierSha256: string; rulesSha256: string; policySha256: string; rulesSemanticHash: string; featureIdentitySha256?: string; servingMaxInputUtf8Bytes?: number };
  datasets: Array<{ id: string; split: string; items: number; description?: string }>;
  /** The dataset `final` was measured on; its `items` must equal the report's. Required with `final`. */
  finalDataset?: string;
  /** The acceptance criteria `gates` were computed from (checkAcceptance). Required to enforce. */
  criteria?: Record<string, unknown>;
  /** The final-system acceptance outcome (evaluation gates). To enforce, it must equal checkAcceptance(final, perHead, criteria). */
  gates: ReleaseGates;
  claims: EvidenceClaim[];
  /** Per-head report (evaluatePerHead), if evaluated. */
  perHead?: Record<string, unknown>;
  /** Final routing report (evaluateFinal). Required to enforce. */
  final?: Record<string, unknown>;
  /** True when the gates were computed in a diagnostics-only (non-gating) run. Such evidence can't be enforced. */
  diagnosticsOnly?: boolean;
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** Validates parsed evidence; throws RouterLoadError (UNSUPPORTED_SCHEMA or EVIDENCE_INVALID). */
export function validateEvidence(raw: unknown): ReleaseEvidence {
  if (!isObject(raw)) throw new RouterLoadError('EVIDENCE_INVALID', 'the evidence is not an object');
  if (raw.schema !== EVIDENCE_SCHEMA && raw.schema !== EVIDENCE_SCHEMA_DOCUMENT) throw new RouterLoadError('UNSUPPORTED_SCHEMA', `unsupported evidence schema ${JSON.stringify(raw.schema)} (this router reads '${EVIDENCE_SCHEMA}' and '${EVIDENCE_SCHEMA_DOCUMENT}')`);
  const documentEvidence = raw.schema === EVIDENCE_SCHEMA_DOCUMENT;
  const p: string[] = [];
  for (const k of Object.keys(raw)) if (!['schema', 'inputs', 'datasets', 'finalDataset', 'criteria', 'gates', 'claims', 'perHead', 'final', 'diagnosticsOnly'].includes(k)) p.push(`unknown field ${k}`);
  const inputs = raw.inputs;
  if (!isObject(inputs)) p.push('inputs must be an object');
  else {
    const hashes = ['classifierSha256', 'rulesSha256', 'policySha256', 'rulesSemanticHash', ...(documentEvidence ? ['featureIdentitySha256'] : [])];
    for (const k of Object.keys(inputs)) if (![...hashes, ...(documentEvidence ? ['servingMaxInputUtf8Bytes'] : [])].includes(k)) p.push(`inputs has unknown field ${k}`);
    for (const k of hashes) if (!isSha256Hex(inputs[k])) p.push(`inputs.${k} must be lowercase hex SHA-256`);
    if (documentEvidence && !(Number.isInteger(inputs.servingMaxInputUtf8Bytes) && (inputs.servingMaxInputUtf8Bytes as number) >= 1)) p.push('inputs.servingMaxInputUtf8Bytes must be a positive integer');
  }
  const datasetIds = new Set<string>();
  const datasetItems = new Map<string, number>();
  if (!Array.isArray(raw.datasets)) p.push('datasets must be an array');
  else raw.datasets.forEach((d: unknown, i: number) => {
    if (!isObject(d) || typeof d.id !== 'string' || !d.id || typeof d.split !== 'string' || !d.split || !Number.isInteger(d.items) || (d.items as number) < 0) p.push(`datasets[${i}] must be { id, split, items }`);
    else if (datasetIds.has(d.id)) p.push(`dataset ${d.id} is listed twice`);
    else { datasetIds.add(d.id); datasetItems.set(d.id, d.items as number); }
  });
  if (raw.criteria !== undefined) p.push(...acceptanceCriteriaProblems(raw.criteria).map((c) => `criteria: ${c}`));
  else if (documentEvidence) p.push('criteria are required in document evidence');
  p.push(...gatesProblems(raw.gates, 'evidence'));
  if (!Array.isArray(raw.claims)) p.push('claims must be an array');
  else raw.claims.forEach((c: unknown, i: number) => {
    if (!isObject(c)) { p.push(`claims[${i}] is not an object`); return; }
    if (typeof c.id !== 'string' || !c.id) p.push(`claims[${i}].id must be a non-empty string`);
    if (c.scope !== 'per-head' && c.scope !== 'final-routing') p.push(`claims[${i}].scope must be 'per-head' or 'final-routing'`);
    if (!(CLAIM_KINDS as readonly unknown[]).includes(c.kind)) p.push(`claims[${i}].kind must be one of ${CLAIM_KINDS.join(', ')}`);
    if (typeof c.method !== 'string' || !c.method) p.push(`claims[${i}].method must be a non-empty string`);
    if (typeof c.statement !== 'string' || !c.statement) p.push(`claims[${i}].statement must be a non-empty string`);
    if (c.subject !== undefined && typeof c.subject !== 'string') p.push(`claims[${i}].subject must be a string`);
    if (typeof c.dataset !== 'string' || !datasetIds.has(c.dataset)) p.push(`claims[${i}].dataset must name a listed dataset`);
  });
  if (raw.perHead !== undefined) p.push(...perHeadReportProblems(raw.perHead));
  if (raw.finalDataset !== undefined && (typeof raw.finalDataset !== 'string' || !datasetIds.has(raw.finalDataset))) p.push('finalDataset must name a listed dataset');
  if (raw.final !== undefined) {
    if (raw.finalDataset === undefined) p.push('finalDataset is required with a final routing report');
    p.push(...finalReportProblems(raw.final, typeof raw.finalDataset === 'string' ? datasetItems.get(raw.finalDataset) : undefined));
  }
  if (raw.diagnosticsOnly !== undefined && typeof raw.diagnosticsOnly !== 'boolean') p.push('diagnosticsOnly must be a boolean');
  if (p.length) throw new RouterLoadError('EVIDENCE_INVALID', 'the evidence is invalid', p);
  return raw as unknown as ReleaseEvidence;
}

const count = (v: unknown) => Number.isInteger(v) && (v as number) >= 0;

/**
 * A recorded rate that is exactly what `rate(count, of)` gives: whole counts, count <= of, and the
 * point estimate and Wilson interval recomputed from them (the gates read these, so they must follow
 * from the counts rather than be taken on trust).
 */
const isRate = (r: unknown): r is Rate => isObject(r) && count(r.count) && count(r.of) && (r.count as number) <= (r.of as number) && canonicalJson(r) === canonicalJson(rate(r.count as number, r.of as number));

/** Problems with one recorded rate: consistent in itself and, where given, with the report's own counts. */
function rateProblems(r: unknown, name: string, expected: { count?: number; of?: number } = {}): string[] {
  if (!isRate(r)) return [`${name} must be { count, of, rate, wilson95 } with rate and interval computed from the counts`];
  const p: string[] = [];
  if (expected.count !== undefined && r.count !== expected.count) p.push(`${name}.count is ${r.count}, the report's counts give ${expected.count}`);
  if (expected.of !== undefined && r.of !== expected.of) p.push(`${name}.of is ${r.of}, the report's counts give ${expected.of}`);
  return p;
}

/**
 * Structural problems with a final routing report (evaluateFinal's output): the right kind, whole
 * counts that add up, an item count matching its own dataset, a list of path mismatches, and every
 * rate the gates read consistent with those counts. A structurally valid report may still record a
 * failed evaluation; that is for the gates and shadow mode.
 */
function finalReportProblems(f: unknown, datasetItems: number | undefined): string[] {
  if (!isObject(f)) return ['final must be a final routing report object'];
  const p: string[] = [];
  if (f.kind !== 'final-routing') p.push("final.kind must be 'final-routing'");
  if (!count(f.items)) p.push('final.items must be a non-negative integer');
  const o = f.outcomes;
  const outcomesOk = isObject(o) && count(o.route) && count(o.review) && count(o.abstain);
  if (!outcomesOk) p.push('final.outcomes must hold route, review and abstain counts');
  else if (count(f.items) && (o.route as number) + (o.review as number) + (o.abstain as number) !== f.items) p.push(`final.outcomes add up to ${(o.route as number) + (o.review as number) + (o.abstain as number)}, not final.items ${f.items}`);
  if (!Array.isArray(f.pathMismatches) || !f.pathMismatches.every((m) => typeof m === 'string')) p.push('final.pathMismatches must be an array of item ids');
  if (count(f.items) && datasetItems !== undefined && f.items !== datasetItems) p.push(`final.items ${f.items} does not match the ${datasetItems} items of its dataset`);
  if (!outcomesOk || !count(f.items)) return p;
  const items = f.items as number, routed = (o as Record<string, number>).route;
  p.push(...rateProblems(f.coverage, 'final.coverage', { count: routed, of: items }));
  p.push(...rateProblems(f.reviewRate, 'final.reviewRate', { count: (o as Record<string, number>).review, of: items }));
  p.push(...rateProblems(f.abstainRate, 'final.abstainRate', { count: (o as Record<string, number>).abstain, of: items }));
  p.push(...rateProblems(f.routedAccuracy, 'final.routedAccuracy', { of: routed }));
  p.push(...rateProblems(f.falseRouteRate, 'final.falseRouteRate'));
  p.push(...rateProblems(f.ruleFiringRate, 'final.ruleFiringRate', { of: items }));
  p.push(...rateProblems(f.rulesSettlementRate, 'final.rulesSettlementRate', { of: items }));
  const m = f.mechanism;
  if (!isObject(m) || !count(m.rule) || !count(m.classifier) || (m.rule as number) + (m.classifier as number) !== routed) p.push(`final.mechanism must split the ${routed} routed items into rule and classifier counts`);
  if (!isObject(f.perRoute)) p.push('final.perRoute must be an object');
  else {
    let sum = 0;
    for (const [r, rep] of Object.entries(f.perRoute)) {
      if (!isObject(rep) || !count(rep.routed)) { p.push(`final.perRoute.${r} must hold a routed count and precision and recall rates`); continue; }
      sum += rep.routed as number;
      p.push(...rateProblems(rep.precision, `final.perRoute.${r}.precision`, { of: rep.routed as number }));
      p.push(...rateProblems(rep.recall, `final.perRoute.${r}.recall`));
    }
    if (sum !== routed) p.push(`final.perRoute routes ${sum} items, the outcomes route ${routed}`);
  }
  return p;
}

/** Structural problems with a per-head report (evaluatePerHead's output), its rates consistent with one confusion matrix per head. */
function perHeadReportProblems(f: unknown): string[] {
  if (!isObject(f)) return ['perHead must be a per-head report object'];
  const p: string[] = [];
  if (f.kind !== 'per-head') p.push("perHead.kind must be 'per-head'");
  if (!count(f.items)) p.push('perHead.items must be a non-negative integer');
  if (!isObject(f.heads)) p.push('perHead.heads must be an object');
  else for (const [h, r] of Object.entries(f.heads)) {
    if (!isObject(r) || !isRate(r.recall) || !isRate(r.precision) || !isRate(r.falseAlarmRate)) { p.push(`perHead.heads.${h} must hold recall, precision and falseAlarmRate rates computed from their counts`); continue; }
    // recall = tp/(tp+fn), precision = tp/(tp+fp), false alarms = fp/(fp+tn).
    const tp = r.recall.count, fp = r.precision.of - r.precision.count;
    if (r.precision.count !== tp || r.falseAlarmRate.count !== fp || r.positives !== r.recall.of || r.labelled !== r.recall.of + r.falseAlarmRate.of) {
      p.push(`perHead.heads.${h}: recall, precision, false alarms, positives and labelled counts don't come from one confusion matrix`);
    }
  }
  return p;
}

/** Why this evidence can't support enforcement (empty when it can). Gate failures are reported separately. */
export function evidenceInsufficiency(e: ReleaseEvidence): string[] {
  const out: string[] = [];
  if (e.diagnosticsOnly) out.push('the evidence comes from a diagnostics-only (non-gating) evaluation');
  const final = e.final as { items?: number; pathMismatches?: unknown[] } | undefined;
  if (!final) out.push('the evidence has no final routing evaluation');
  else {
    if (!final.items) out.push('the final routing evaluation has no items');
    if (final.pathMismatches?.length) out.push(`the rules-settled path disagreed with the always-embed reference on ${final.pathMismatches.length} item(s)`);
  }
  if (!e.datasets.some((d) => d.items > 0)) out.push('the evidence lists no evaluated items');
  // The recorded gates must be what the recorded criteria give on the recorded reports.
  const criteria = e.criteria as AcceptanceCriteria | undefined;
  if (!criteria) out.push('the evidence records no acceptance criteria, so its gates cannot be checked');
  else if (!hasPerformanceCriteria(criteria)) out.push('the recorded acceptance criteria name no performance target');
  else if (final) {
    let recomputed: ReturnType<typeof checkAcceptance> | null = null;
    try {
      recomputed = checkAcceptance(e.final as unknown as FinalReport, (e.perHead ?? null) as unknown as PerHeadReport | null, criteria);
    } catch {
      out.push('the recorded reports cannot be checked against the recorded criteria');
    }
    if (recomputed && canonicalJson(recomputed) !== canonicalJson(e.gates)) {
      out.push(`the recorded gates differ from the recorded criteria applied to the recorded reports${recomputed.failures.length ? ` (${recomputed.failures.join('; ')})` : ''}`);
    }
  }
  return out;
}
