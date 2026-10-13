/**
 * The decision evaluator: rules findings + (when needed) calibrated probabilities -> one candidate.
 *
 * It wraps embedding-classifier's `decide` and `settleWithRules`, so the router and
 * existing callers decide identically:
 *   - the first firing rule in rule-set order names the fired head; rule-miner's matcher reports no
 *     dismissals whenever a rule fires, so firing beats dismissal. Findings that both fire and dismiss
 *     the same head can't come from the matcher and are rejected as a caller error
 *   - a fired head counts as at its threshold (even a disabled threshold above 1); its probability is
 *     never replaced by a threshold
 *   - dismissed heads need no score, never fire and never suppress
 *   - suppression is computed once from the unsuppressed evidence, then the first eligible head in
 *     priority order fires; otherwise review-band evidence (suppressed heads included) means review
 *   - otherwise `onAbstain` applies
 *
 * `decide` is the always-embed reference. `settle` is the optimised path: a candidate when the rules
 * alone fix the outcome for every possible valid score, else null. Where `settle` returns a candidate it
 * equals `decide` for any scores (tested exhaustively against an independent oracle).
 */
import { decide as classifierDecide, settleWithRules, type Decision, type DecisionHeads, type DecisionPolicy } from '@liquidau/embedding-classifier';

import { deepFreeze, jsonClone } from './bytes.ts';
import { validatePolicy, type RouterPolicy } from './policy.ts';
import type { ReviewAnswerFields } from './review.ts';

/** A head's decision-relevant numbers: the restricted projection of a classifier head. */
export interface HeadProjection {
  threshold: number;
  reviewFloor: number;
}

export interface DecisionModel {
  heads: Readonly<Record<string, HeadProjection>>;
}

/** What the rules tier found for one message: rule-miner's `ruleSetMatcher(set).evaluate(text)`. */
export interface RuleFindings {
  /**
   * `source` 'edge': the edge tier fired this label (id 'edge:<label>'); 'middle': the middle tier did
   * (id 'middle:<label>'); absent or 'rule': a rule did.
   */
  fired: { id: string; label: string; source?: 'rule' | 'edge' | 'middle' } | null;
  /**
   * Every head fired, by a rule, the edge or the middle tier, highest priority first (`fired` is its
   * first entry). Each counts as at its threshold, as certification counted it. Absent: `fired` alone.
   */
  firing?: ReadonlyArray<{ id: string; label: string; source?: 'rule' | 'edge' | 'middle' }>;
  dismissed: readonly string[];
}

export type ReviewReason = 'near_threshold' | 'abstention_policy';
export type AbstainReason = 'no_match' | 'all_dismissed';

export type Candidate =
  | { outcome: 'route'; routeId: string; destination: string; mechanism: 'rule' | 'edge' | 'middle' | 'classifier' }
  | ({ outcome: 'review'; reason: ReviewReason } & ReviewAnswerFields)
  | { outcome: 'abstain'; reason: AbstainReason };

/** A probability the policy needs is missing, non-finite or outside [0, 1]. Maps to INVALID_SCORE. */
export class ScoreError extends Error {
  override readonly name = 'ScoreError';
}

export interface DecisionEvaluator {
  /** Route ids in priority order. */
  readonly priority: readonly string[];
  /** The rules-only decision when it can't depend on any score, else null. Throws on conflicting findings. */
  settle(rules: RuleFindings): Candidate | null;
  /** Heads whose probabilities `decide` reads for these findings (not dismissed, not rule-fired). */
  requiredScores(rules: RuleFindings): string[];
  /** The always-embed reference decision. Throws ScoreError on an invalid needed probability. */
  decide(rules: RuleFindings, scores: Readonly<Record<string, number | undefined>>): Candidate;
}

export interface Settler {
  /** Route ids in priority order. */
  readonly priority: readonly string[];
  /** The rules-only decision when it can't depend on any score, else null. Throws on conflicting findings. */
  settle(rules: RuleFindings): Candidate | null;
}

/** Every fired head of a finding: `firing`, or `fired` alone. */
export function firingOf(rules: RuleFindings): ReadonlyArray<NonNullable<RuleFindings['fired']>> {
  return rules.firing ?? (rules.fired ? [rules.fired] : []);
}

/** The policy-only half shared by the settler and the evaluator: label checks, settlement and result shapes. */
function policyParts(policy: RouterPolicy) {
  const p = deepFreeze(jsonClone(policy));
  const classifierPolicy: DecisionPolicy<string> = deepFreeze({ priority: [...p.priority], suppress: p.suppress.map((s) => ({ when: [...s.when], heads: [...s.heads] })) });
  const destination = new Map(p.routes.map((r) => [r.id, r.destination]));
  const routes = new Set(p.priority);
  const check = (rules: RuleFindings) => {
    for (const f of firingOf(rules)) {
      if (!routes.has(f.label)) throw new Error(`rule ${f.id} fired ${f.label}, which is not a route`);
      if (rules.dismissed.includes(f.label)) throw new Error(`conflicting findings: ${f.id} fires ${f.label}, which is also dismissed`);
    }
    for (const d of rules.dismissed) if (!routes.has(d)) throw new Error(`dismissed label ${d} is not a route`);
    if (rules.fired !== null && rules.dismissed.includes(rules.fired.label)) {
      throw new Error(`conflicting findings: rule ${rules.fired.id} fires ${rules.fired.label}, which is also dismissed (rule-miner's matcher never reports both)`);
    }
  };
  const candidate = (d: Decision<string>, rules: RuleFindings): Candidate => {
    if (d.head !== null) {
      const source = d.reason === 'rule' ? firingOf(rules).find((f) => f.label === d.head)?.source ?? 'rule' : 'classifier';
      return { outcome: 'route', routeId: d.head, destination: destination.get(d.head)!, mechanism: source };
    }
    if (d.reason === 'near_threshold') return { outcome: 'review', reason: 'near_threshold' };
    if (p.onAbstain === 'review') return { outcome: 'review', reason: 'abstention_policy' };
    const off = new Set(rules.dismissed);
    return { outcome: 'abstain', reason: p.priority.every((h) => off.has(h)) ? 'all_dismissed' : 'no_match' };
  };
  const settle = (rules: RuleFindings) => {
    check(rules);
    const s = settleWithRules(classifierPolicy, rules);
    return s.settled ? candidate(s.decision, rules) : null;
  };
  return { p, classifierPolicy, check, candidate, settle };
}

/**
 * Settlement alone, from the policy (no classifier): what a rules tier needs. Settlement never reads a
 * score or threshold, so it equals createDecisionEvaluator(...).settle for the same policy.
 */
export function createSettler(policy: RouterPolicy): Settler {
  const parts = policyParts(policy);
  validatePolicy(parts.p);
  return Object.freeze({ priority: parts.p.priority, settle: parts.settle });
}

/**
 * Builds a frozen evaluator from a head projection and a policy (both copied, then validated: the
 * policy's routes must be exactly the model's heads). Usable offline without a release.
 */
export function createDecisionEvaluator(model: DecisionModel, policy: RouterPolicy): DecisionEvaluator {
  const { p, classifierPolicy, check, candidate, settle } = policyParts(policy);
  validatePolicy(p, { heads: Object.keys(model.heads) });
  const heads: Record<string, { threshold: number; review_floor: number }> = {};
  for (const [h, proj] of Object.entries(model.heads)) {
    const { threshold, reviewFloor } = proj ?? ({} as HeadProjection);
    if (!Number.isFinite(threshold) || !Number.isFinite(reviewFloor) || !(reviewFloor >= 0 && reviewFloor <= threshold)) {
      throw new Error(`head ${h} needs finite 0 <= reviewFloor <= threshold (got ${reviewFloor}, ${threshold})`);
    }
    heads[h] = { threshold, review_floor: reviewFloor };
  }
  const artifact: DecisionHeads<string> = deepFreeze({ heads });
  const required = (rules: RuleFindings) => {
    const off = new Set(rules.dismissed);
    const on = new Set(firingOf(rules).map((f) => f.label));
    return p.priority.filter((h) => !off.has(h) && !on.has(h));
  };

  return Object.freeze({
    priority: p.priority,
    settle,
    requiredScores(rules: RuleFindings) {
      check(rules);
      return required(rules);
    },
    decide(rules: RuleFindings, scores: Readonly<Record<string, number | undefined>>) {
      check(rules);
      for (const h of required(rules)) {
        const s = scores[h];
        if (typeof s !== 'number' || !Number.isFinite(s) || s < 0 || s > 1) throw new ScoreError(`head ${h} has no valid calibrated probability (${s})`);
      }
      let d: Decision<string>;
      try {
        d = classifierDecide(artifact, scores, classifierPolicy, rules.dismissed, firingOf(rules).map((f) => f.label));
      } catch (e) {
        throw new ScoreError((e as Error).message);
      }
      return candidate(d, rules);
    },
  });
}
