/**
 * The edge tier: the rules plus the release's edge document (embedding-classifier's 'liquidau-edge/1'),
 * evaluated on the original text before any embedding. Its output is ordinary rule findings, so the
 * policy's settlement and decision logic is unchanged:
 *
 *   dismissed   the rules' dismissals plus the heads the edge dismissed, except a label a rule fired
 *               (the rule stands: the head's certification counted that row as a miss, so firing
 *               it only adds recall)
 *   fired       the highest-priority label among the rule that fired and the heads the edge fired
 *               (source 'edge' when the edge's), excluding dismissed labels: a rule's dismissal beats
 *               an edge fire, as in certification (trainHeads scores dismissed rows 0 first)
 *
 * The edge's 'rules' scorer reads whether any certified firing rule of the head's label fired (not
 * only the first, which is all the matcher reports). A static scorer needs the host's static encoder,
 * whose identity must equal the document's.
 */
import { scoreEdge, scoreMiddle, type EdgeDocument, type MiddleDocument } from '@liquidau/embedding-classifier';
import { compileRule, canonicalSegments, ruleSetMatcher, type RuleSet } from '@liquidau/rule-miner';
import type { StaticEmbedder, StaticIdentity } from '@liquidau/text-preprocessing';

import { canonicalJson } from './bytes.ts';
import { firingOf, type RuleFindings } from './decision.ts';
import { RouterLoadError } from './errors.ts';

export interface EdgeFindings {
  findings: RuleFindings;
  /** What the edge decided (null without an edge document). */
  edge: { fired: string[]; dismissed: string[] } | null;
}

export interface EdgeTier {
  readonly hasEdge: boolean;
  evaluate(text: string): EdgeFindings;
}

/** Problems with a host static encoder against the identity an edge document needs (empty when it fits). */
export function staticEncoderProblems(needed: StaticIdentity | null, encoder: StaticEmbedder | undefined): string[] {
  if (!needed) return [];
  if (!encoder) return [`the edge needs a static encoder (${needed.modelId}); pass staticEncoder`];
  return canonicalJson(encoder.identity) === canonicalJson(needed) ? [] : [`the static encoder is ${canonicalJson(encoder.identity)}, the edge needs ${canonicalJson(needed)}`];
}

/**
 * A static encoder, or a function that loads one. The loader is called only when the release's edge
 * uses a static embedding, so hosts can pass it for every release without paying for the weight table
 * where it isn't needed.
 */
export type StaticEncoderOption = StaticEmbedder | (() => StaticEmbedder | Promise<StaticEmbedder>);

/** The static encoder an edge document needs, loading it only then (undefined when it needs none). */
export async function resolveStaticEncoder(edge: EdgeDocument | null, option: StaticEncoderOption | undefined): Promise<StaticEmbedder | undefined> {
  if (!edge?.static || option === undefined) return undefined;
  return typeof option === 'function' ? await option() : option;
}

export function createEdgeTier(ruleSet: RuleSet, edge: EdgeDocument | null, priority: readonly string[], staticEncoder?: StaticEmbedder): EdgeTier {
  const matcher = ruleSetMatcher(ruleSet);
  const vectorHeads = edge ? Object.entries(edge.heads).filter(([, m]) => m?.scorers.includes('vector')).map(([h]) => h) : [];
  if (vectorHeads.length) throw new RouterLoadError('CAPABILITY_MISSING', "the edge uses the 'vector' scorer, which the router's edge tier cannot supply (it runs before any encoder): retrain the edge without it", vectorHeads);
  const problems = edge ? staticEncoderProblems(edge.static, staticEncoder) : [];
  if (problems.length) throw new RouterLoadError('STATIC_ENCODER_MISMATCH', 'the static encoder does not match the edge', problems);
  const edgeHeads = edge ? Object.keys(edge.heads) : [];
  const needsRules = edge ? Object.values(edge.heads).some((m) => m?.scorers.includes('rules')) : false;
  // Firing rules per label, compiled once.
  const byLabel = new Map<string, Array<(segments: readonly string[]) => boolean>>();
  if (needsRules) for (const r of ruleSet.rules) if (r.effect !== 'dismiss') {
    const list = byLabel.get(r.label) ?? [];
    list.push(compileRule(r));
    byLabel.set(r.label, list);
  }
  const rank = new Map(priority.map((h, i) => [h, i]));
  return Object.freeze({
    hasEdge: !!edge,
    evaluate(text: string): EdgeFindings {
      const found = matcher.evaluate(text);
      const ruleFiring = (found.firing ?? (found.fired ? [found.fired] : [])).map((f) => ({ ...f, source: 'rule' as const }));
      if (!edge) return { findings: { fired: found.fired, firing: ruleFiring, dismissed: found.dismissed }, edge: null };
      const segments = needsRules ? canonicalSegments(text, ruleSet.lexicon) : null;
      const ruleFired = (head: string) => (byLabel.get(head) ?? []).some((f) => f(segments!));
      const scored = scoreEdge(edge, { text, ruleFired, ...(edge.static ? { static: staticEncoder!.embed(text) } : {}) });
      const fired = edgeHeads.filter((h) => scored[h]?.decision === 'fire');
      const dismissedByEdge = edgeHeads.filter((h) => scored[h]?.decision === 'dismiss');
      const ruleLabels = new Set(ruleFiring.map((f) => f.label));
      const dismissed = [...new Set([...found.dismissed, ...dismissedByEdge.filter((h) => !ruleLabels.has(h))])];
      const off = new Set(dismissed);
      // Every fired head, highest priority first (a rule before the edge on the same label, which then appears once).
      const candidates: Array<NonNullable<RuleFindings['fired']>> = ruleFiring.filter((f) => !off.has(f.label));
      for (const h of fired) if (!off.has(h) && !ruleLabels.has(h)) candidates.push({ id: `edge:${h}`, label: h, source: 'edge' });
      candidates.sort((a, b) => (rank.get(a.label) ?? Infinity) - (rank.get(b.label) ?? Infinity) || (a.source === 'rule' ? -1 : 1));
      return { findings: { fired: candidates[0] ?? null, firing: candidates, dismissed }, edge: { fired, dismissed: dismissedByEdge } };
    },
  });
}

/**
 * The heads the middle tier decides for a message: those it has a model for that the rules and the edge
 * left undecided (not dismissed, not fired by a rule or the edge). Certification pinned the edge's
 * decisions before the middle's, and a rule's dismissal before both, so serving skips the same heads.
 */
export function middleHeads(middle: MiddleDocument, found: EdgeFindings): string[] {
  const decided = new Set([...found.findings.dismissed, ...(found.findings.fired ? [found.findings.fired.label] : []), ...(found.edge?.fired ?? []), ...(found.edge?.dismissed ?? [])]);
  return Object.keys(middle.heads).filter((h) => !decided.has(h));
}

/**
 * The rules tier's findings with the middle tier's decisions added, by the edge tier's precedence: its
 * dismissals join `dismissed` (never a label a rule fired), and its fires compete with the rule's or the
 * edge's fire by priority (source 'middle', id 'middle:<label>'; at equal rank a rule wins, then the
 * edge). Offline evaluation builds its items with the same function, so serving and evidence agree.
 */
export function mergeMiddle(found: EdgeFindings, decided: { fired: readonly string[]; dismissed: readonly string[] }, priority: readonly string[]): RuleFindings {
  const prior = found.findings;
  const priorFiring = firingOf(prior);
  const ruleFired = new Set(priorFiring.filter((f) => (f.source ?? 'rule') === 'rule').map((f) => f.label));
  const dismissed = [...new Set([...prior.dismissed, ...decided.dismissed.filter((h) => !ruleFired.has(h))])];
  const off = new Set(dismissed);
  const rank = new Map(priority.map((h, i) => [h, i]));
  const order = { rule: 0, edge: 1, middle: 2 } as const;
  const candidates: Array<NonNullable<RuleFindings['fired']>> = priorFiring.filter((f) => !off.has(f.label));
  const already = new Set(candidates.map((f) => f.label));
  for (const h of decided.fired) if (!off.has(h) && !already.has(h)) candidates.push({ id: `middle:${h}`, label: h, source: 'middle' });
  candidates.sort((a, b) => (rank.get(a.label) ?? Infinity) - (rank.get(b.label) ?? Infinity) || order[a.source ?? 'rule'] - order[b.source ?? 'rule']);
  return { fired: candidates[0] ?? null, firing: candidates, dismissed };
}

/** What the middle tier decides for a message: scoreMiddle on its vector, for `heads` only. */
export function middleDecisions(middle: MiddleDocument, vector: readonly number[], heads: readonly string[]): { fired: string[]; dismissed: string[] } {
  const scored = scoreMiddle(middle, vector) as Record<string, { decision: 'fire' | 'dismiss' | null } | undefined>;
  return { fired: heads.filter((h) => scored[h]?.decision === 'fire'), dismissed: heads.filter((h) => scored[h]?.decision === 'dismiss') };
}
