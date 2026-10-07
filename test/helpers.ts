/**
 * A complete, valid release built in memory: three linear heads on a 4-dimensional fake embedding,
 * a rule set with firing and dismissal rules, a policy with suppression, and evidence from the
 * evaluation subpath. `tweak` hooks mutate each document before it is serialised, so each test breaks
 * exactly one thing.
 */
import { buildArtifact, scoreEmbedding, type ClassifierArtifact, type HeadSpec } from '@liquidau/embedding-classifier';
import { buildRuleSet, ruleSetHash, ruleSetMatcher, type RuleSet } from '@liquidau/rule-miner';

import { buildRelease, createDecisionEvaluator, documentDigest, encodeJson, functionEncoder, loadRouter, memorySource, type BuiltRelease, type EncoderIdentity, type LoadRouterOptions, type ReleaseEvidence, type RouterPolicy } from '../src/index.ts';
import { buildEvidence, evaluateFinal, evaluatePerHead, type FinalItem } from '../src/evaluation/index.ts';

export const HEADS = ['urgent', 'refund', 'other'] as const;
export type Head = (typeof HEADS)[number];

export const IDENTITY: EncoderIdentity = {
  schema: 'liquidau-encoder/1', modelId: 'fake/model', revision: 'sha256:0000', dimensions: 4, precision: 'fp32', inputType: null, layers: null,
  pooling: 'mean', normalization: 'none', tokenizerRevision: 'fake-tokenizer/1', maxChars: null, maxTokens: null, truncation: 'none',
};

/** p = sigmoid(10 * x[i] - 5): x[i] = 0.5 gives 0.5, 1 gives 0.9933, 0 gives 0.0067. */
const head = (i: number, threshold: number, review_floor: number): HeadSpec => ({
  weights: [0, 1, 2, 3].map((k) => (k === i ? 10 : 0)), bias: -5, calibration: { method: 'platt', a: 1, c: 0 }, threshold, review_floor,
});

/** The vector giving head i the probability p (others near 0). */
export function vectorFor(probabilities: Partial<Record<Head, number>>): number[] {
  return HEADS.map((h) => { const p = probabilities[h] ?? 0.0001; return (Math.log(p / (1 - p)) + 5) / 10; }).concat([0]);
}

export function ruleSet(): RuleSet {
  return buildRuleSet('rules-1', [
    { label: 'refund', rules: [{ id: 'r1', pattern: { kind: 'phrase', tokens: ['refund', 'please'] } }] },
    { label: 'urgent', rules: [{ id: 'u1', pattern: { kind: 'phrase', tokens: ['emergency'] } }] },
    { label: 'other', effect: 'dismiss', rules: [{ id: 'd1', pattern: { kind: 'phrase', tokens: ['hello'] } }] },
    { label: 'urgent', effect: 'dismiss', rules: [{ id: 'd2', pattern: { kind: 'phrase', tokens: ['hello'] } }] },
    { label: 'refund', effect: 'dismiss', rules: [{ id: 'd3', pattern: { kind: 'phrase', tokens: ['hello'] } }] },
  ], { createdAt: '2026-10-07T00:00:00Z' });
}

export function artifact(rules: RuleSet, gatesPassed = true): ClassifierArtifact<Head> {
  return buildArtifact<Head>({
    heads: { urgent: head(0, 0.8, 0.3), refund: head(1, 0.7, 0.4), other: head(2, 0.6, 0.5) },
    failures: gatesPassed ? [] : ['urgent recall below target'], warnings: [], evaluation: {}, headChoice: {}, convergence: {}, weakLabels: {}, testProbabilities: {},
  } as never, {
    version: 'clf-1', createdAt: '2026-10-07T00:00:00Z', embedding: { model_id: 'fake/model', dimensions: 4, normalize: false, precision: 'fp32' },
    router: { ruleSetHash: ruleSetHash(rules) },
  });
}

export const POLICY: RouterPolicy = {
  schema: 'liquidau-router-policy/1',
  routes: [
    { id: 'urgent', destination: 'queue:urgent', description: 'Needs a human now' },
    { id: 'refund', destination: 'queue:refunds' },
    { id: 'other', destination: 'reply:out-of-scope' },
  ],
  priority: ['urgent', 'refund', 'other'],
  suppress: [{ when: ['urgent'], heads: ['other'] }],
  onAbstain: 'none',
};

/** Labelled evaluation items in the router's own terms. */
export const EVAL: Array<{ text: string; p: Partial<Record<Head, number>>; acceptable: Head[] }> = [
  { text: 'refund please', p: { refund: 0.2 }, acceptable: ['refund'] },
  { text: 'I want my money back', p: { refund: 0.9 }, acceptable: ['refund'] },
  { text: 'emergency at home', p: { urgent: 0.5 }, acceptable: ['urgent'] },
  { text: 'my card is stuck and I am scared', p: { urgent: 0.95 }, acceptable: ['urgent'] },
  { text: 'what is the weather', p: { other: 0.9 }, acceptable: ['other'] },
  { text: 'hello', p: {}, acceptable: [] },
];

export interface Tweaks {
  artifact?: (a: ClassifierArtifact<Head>) => void;
  rules?: (r: RuleSet) => void;
  policy?: (p: RouterPolicy) => void;
  evidence?: (e: ReleaseEvidence) => void;
  /** The release encoder identity (default IDENTITY); set the artifact's embedding to match. */
  identity?: EncoderIdentity;
  gatesPassed?: boolean;
  evaluationPasses?: boolean;
}

export async function makeRelease(tweaks: Tweaks = {}): Promise<BuiltRelease> {
  const rules = ruleSet();
  tweaks.rules?.(rules);
  const art = artifact(ruleSet(), tweaks.gatesPassed ?? true);
  tweaks.artifact?.(art);
  const policy: RouterPolicy = JSON.parse(JSON.stringify(POLICY));
  tweaks.policy?.(policy);
  const documents = { classifier: encodeJson(art), rules: encodeJson(rules), policy: encodeJson(policy) };

  // Evidence is computed on the untweaked fixture: only its digests and gates matter to these tests.
  const base = artifact(ruleSet());
  const matcher = ruleSetMatcher(ruleSet());
  const items: FinalItem[] = EVAL.map((e, i) => ({ id: String(i), rules: matcher.evaluate(e.text), scores: scoreEmbedding(base, vectorFor(e.p)) as Record<string, number>, acceptable: e.acceptable }));
  const model = { heads: Object.fromEntries(Object.entries(base.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) };
  const final = evaluateFinal(createDecisionEvaluator(model, POLICY), items);
  const perHead = evaluatePerHead(model, items.map((it, i) => ({ rules: it.rules, scores: it.scores, labels: Object.fromEntries(HEADS.map((h) => [h, EVAL[i].acceptable.includes(h) ? 1 : 0])) })));
  const criteria = { minCoverage: tweaks.evaluationPasses === false ? 0.99 : 0.5, compare: 'point' as const };
  const evidence = buildEvidence({
    inputs: { classifierSha256: await documentDigest(documents.classifier), rulesSha256: await documentDigest(documents.rules), policySha256: await documentDigest(documents.policy), rulesSemanticHash: ruleSetHash(rules) },
    dataset: { id: 'fixture-test', split: 'test' }, final, perHead, criteria,
  });
  tweaks.evidence?.(evidence);
  return buildRelease({
    releaseId: 'release-1', createdAt: '2026-10-07T00:00:00Z', documents: { ...documents, evidence: encodeJson(evidence) }, embedding: tweaks.identity ?? IDENTITY,
    serving: { maxInputUtf8Bytes: 4096 }, packages: { '@liquidau/embedding-classifier': '0.8.0', '@liquidau/rule-miner': '0.5.1', '@liquidau/router': '0.1.0' },
  });
}

/** A fake encoder over a text -> probabilities table; counts calls. */
export function fakeEncoder(table: Record<string, Partial<Record<Head, number>>> = Object.fromEntries(EVAL.map((e) => [e.text, e.p])), identity: EncoderIdentity = IDENTITY) {
  const calls: string[] = [];
  const encoder = functionEncoder(identity, (t) => { calls.push(t); return vectorFor(table[t] ?? {}); });
  return { encoder, calls };
}

export function options(built: BuiltRelease, overrides: Partial<LoadRouterOptions> = {}): LoadRouterOptions {
  return {
    source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce',
    encoder: fakeEncoder().encoder, timeoutMs: 1000, monitoring: { sampleRate: 0, samplingSalt: 'salt' }, ...overrides,
  };
}

export const load = async (tweaks: Tweaks = {}, overrides: Partial<LoadRouterOptions> = {}) => loadRouter(options(await makeRelease(tweaks), overrides));
