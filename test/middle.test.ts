/**
 * Releases with a middle tier: the middle encoder decides the heads the rules and the edge left, a request
 * it settles never reaches the main encoder, its decisions merge with the edge tier's precedence, and the
 * release pairs the middle file with the classifier, the edge and the evidence.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { edgeHeadDigest, middleHeadDigest, scoreEmbedding, type ClassifierArtifact, type EdgeDocument, type MiddleDocument } from '@liquidau/embedding-classifier';
import { ruleSetHash } from '@liquidau/rule-miner';
import { DEFAULT_LEXICAL, lexicalFeatures } from '@liquidau/text-preprocessing';

import {
  buildRelease, createDecisionEvaluator, createEdgeTier, documentDigest, encodeJson, functionEncoder, loadRouter, loadRulesTier, memorySource, mergeMiddle, middleDecisions, middleHeads, RouterLoadError,
  type BuiltRelease, type EncoderIdentity, type RouterEvent,
} from '../src/index.ts';
import { buildEvidence, evaluateFinal, evaluatePerHead, type FinalItem } from '../src/evaluation/index.ts';
import { artifact, EVAL, fakeEncoder, HEADS, IDENTITY, options, POLICY, ruleSet, vectorFor, type Head } from './helpers.ts';

const code = (c: string) => (e: unknown) => e instanceof RouterLoadError && e.code === c;

/** An edge that fires `urgent` on "stolen" and decides nothing else. */
function edge(): EdgeDocument<Head> {
  const f = lexicalFeatures('stolen');
  return {
    schema: 'liquidau-edge/1', lexical: { ...DEFAULT_LEXICAL }, static: null, vector: null,
    heads: { urgent: { scorers: ['lexical'], lexical: { bias: 0, index: [...f.index], weight: [...f.index].map(() => 1) }, blend: { coef: [10], intercept: -1 }, fire: 0.5, dismiss: null, claims: { max_false_alarms: 0.005, max_dismissed: 0, delta: 0.05 } } },
  };
}

const MIDDLE_IDENTITY: EncoderIdentity = { ...IDENTITY, modelId: 'fake/middle', dimensions: 2 };
/** The middle encoder's vector per text: x[0] > 0.7 fires urgent, x[0] < 0.3 dismisses it. */
const MIDDLE_TABLE: Record<string, [number, number]> = {
  'my card is stuck and I am scared': [0.9, 0], 'what is the weather': [0.1, 0], 'I want my money back': [0.1, 0], 'refund please': [0.5, 0], 'emergency at home': [0.5, 0], hello: [0.5, 0],
};
const middleEncoder = (identity = MIDDLE_IDENTITY) => {
  const calls: string[] = [];
  return { calls, encoder: functionEncoder(identity, (t) => { calls.push(t); return MIDDLE_TABLE[t] ?? [0.5, 0]; }) };
};
function middle(e: EdgeDocument<Head>): MiddleDocument<Head> {
  return {
    schema: 'liquidau-middle/1', embedding: { model_id: 'fake/middle', dimensions: 2, normalize: false, precision: 'fp32' },
    heads: { urgent: { model: { coef: [10, 0], intercept: -5 }, fire: 2, dismiss: -2, claims: { max_false_alarms: 0.01, max_dismissed: 0.0375, delta: 0.05, prior_edge_sha256: edgeHeadDigest(e, 'urgent')! } } },
  };
}

/** A release whose `urgent` head was certified behind the edge and the middle tier, evaluated through both. */
async function middleRelease(tweak: { middle?: (m: MiddleDocument<Head>) => void; artifact?: (a: ClassifierArtifact<Head>) => void; evidenceMiddle?: 'omit'; middleEmbedding?: EncoderIdentity | null } = {}): Promise<BuiltRelease> {
  const rules = ruleSet();
  const e = edge(), m = middle(e);
  const art = artifact(rules);
  art.heads.urgent = { ...art.heads.urgent!, edge: { model_sha256: edgeHeadDigest(e, 'urgent')! }, middle: { model_sha256: middleHeadDigest(m, 'urgent')! } };
  tweak.artifact?.(art);
  tweak.middle?.(m);
  const documents = { classifier: encodeJson(art), rules: encodeJson(rules), policy: encodeJson(POLICY), edge: encodeJson(e), middle: encodeJson(m) };
  const tier = createEdgeTier(rules, e, POLICY.priority);
  const items: FinalItem[] = EVAL.map((x, i) => {
    const found = tier.evaluate(x.text);
    const merged = mergeMiddle(found, middleDecisions(m, MIDDLE_TABLE[x.text], middleHeads(m, found)), POLICY.priority);
    return { id: String(i), rules: merged, beforeMiddle: found.findings, scores: scoreEmbedding(art, vectorFor(x.p)) as Record<string, number>, acceptable: x.acceptable };
  });
  const model = { heads: Object.fromEntries(Object.entries(art.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) };
  const final = evaluateFinal(createDecisionEvaluator(model, POLICY), items);
  const perHead = evaluatePerHead(model, items.map((it, i) => ({ rules: it.rules, scores: it.scores, labels: Object.fromEntries(HEADS.map((h) => [h, EVAL[i].acceptable.includes(h) ? 1 : 0])) })));
  const evidence = buildEvidence({
    inputs: {
      classifierSha256: await documentDigest(documents.classifier), rulesSha256: await documentDigest(documents.rules), policySha256: await documentDigest(documents.policy), rulesSemanticHash: ruleSetHash(rules),
      edgeSha256: await documentDigest(documents.edge), ...(tweak.evidenceMiddle === 'omit' ? {} : { middleSha256: await documentDigest(documents.middle) }),
    },
    dataset: { id: 'fixture-test', split: 'test' }, final, perHead, criteria: { minCoverage: 0.5, compare: 'point' },
  });
  return buildRelease({
    releaseId: 'release-middle', createdAt: '2026-10-11T00:00:00Z', documents: { ...documents, evidence: encodeJson(evidence) }, embedding: IDENTITY,
    ...(tweak.middleEmbedding === null ? {} : { middleEmbedding: tweak.middleEmbedding ?? MIDDLE_IDENTITY }),
    serving: { maxInputUtf8Bytes: 4096 }, packages: { '@liquidau/embedding-classifier': '0.9.0', '@liquidau/rule-miner': '0.5.2', '@liquidau/router': '0.3.0', '@liquidau/text-preprocessing': '0.2.0' },
  });
}

test('a request the middle tier settles skips the main encoder; one it leaves is scored as usual', async () => {
  const built = await middleRelease();
  assert.equal(built.manifest.files.middle?.key, 'middle.json');
  const events: RouterEvent[] = [];
  const main = fakeEncoder(), mid = middleEncoder();
  const router = await loadRouter(options(built, { encoder: main.encoder, middleEncoder: mid.encoder, observer: (e) => { events.push(e); } }));
  // The middle fires urgent (top priority, nothing suppresses it): settled without the main encoder.
  const r1 = await router.route({ text: 'my card is stuck and I am scared', requestId: 'a' });
  assert.equal(r1.outcome, 'route');
  assert.equal(r1.outcome === 'route' && r1.mechanism, 'middle');
  assert.deepEqual(main.calls, []);
  assert.deepEqual(mid.calls, ['my card is stuck and I am scared']);
  // The middle dismisses urgent; refund still needs its score, so the main encoder runs.
  const r2 = await router.route({ text: 'I want my money back', requestId: 'b' });
  assert.equal(r2.outcome === 'route' && r2.routeId, 'refund');
  assert.equal(r2.outcome === 'route' && r2.mechanism, 'classifier');
  assert.deepEqual(main.calls, ['I want my money back']);
  // The edge fires urgent on "stolen": the rules tier settles, and the middle encoder never runs.
  const r3 = await router.route({ text: 'my card was stolen', requestId: 'c' });
  assert.equal(r3.outcome === 'route' && r3.mechanism, 'edge');
  assert.equal(mid.calls.length, 2);
  await router.flush();
  const decisions = events.filter((e) => e.type === 'decision') as Array<Extract<RouterEvent, { type: 'decision' }>>;
  const [a, b, c] = ['a', 'b', 'c'].map((id) => decisions.find((d) => d.requestId === id)!);
  assert.equal(a.settledBy, 'middle');
  assert.equal(a.rulesSettled, false);
  assert.equal(a.embedded, false);
  assert.deepEqual(a.middle, { fired: ['urgent'], dismissed: [] });
  assert.equal(a.firedRule, 'middle:urgent');
  assert.ok(typeof a.timings.middleMs === 'number');
  assert.equal(b.settledBy, null);
  assert.equal(b.embedded, true);
  assert.deepEqual(b.middle, { fired: [], dismissed: ['urgent'] });
  assert.ok(b.dismissed.includes('urgent'));
  assert.equal(c.settledBy, 'rules');
  assert.equal(c.middle, undefined);
  assert.equal(c.timings.middleMs, undefined);
  await router.close();
});

test('the middle tier merges with the edge tier’s precedence', () => {
  const found = (fired: { id: string; label: string; source?: 'rule' | 'edge' } | null, dismissed: string[] = [], edge: { fired: string[]; dismissed: string[] } | null = null) => ({ findings: { fired, dismissed }, edge });
  // A rule's dismissal beats a middle fire; a middle dismissal never removes a rule's fire.
  assert.deepEqual(mergeMiddle(found(null, ['urgent']), { fired: ['urgent'], dismissed: [] }, POLICY.priority), { fired: null, firing: [], dismissed: ['urgent'] });
  assert.deepEqual(mergeMiddle(found({ id: 'r1', label: 'refund' }), { fired: [], dismissed: ['refund'] }, POLICY.priority).fired?.id, 'r1');
  // By priority: a middle fire on urgent outranks a rule fire on refund; a rule fire on urgent outranks a middle fire on other.
  assert.equal(mergeMiddle(found({ id: 'r1', label: 'refund' }), { fired: ['urgent'], dismissed: [] }, POLICY.priority).fired?.id, 'middle:urgent');
  assert.equal(mergeMiddle(found({ id: 'u1', label: 'urgent' }), { fired: ['other'], dismissed: [] }, POLICY.priority).fired?.id, 'u1');
  // The middle scores only heads the rules and the edge left undecided.
  const m = middle(edge());
  assert.deepEqual(middleHeads(m, found(null, [], { fired: [], dismissed: [] })), ['urgent']);
  assert.deepEqual(middleHeads(m, found(null, [], { fired: ['urgent'], dismissed: [] })), []);
  assert.deepEqual(middleHeads(m, found(null, ['urgent'])), []);
});

test('evaluation tells rules- from middle-settled items apart, and the evidence records it', async () => {
  const built = await middleRelease();
  const ev = JSON.parse(new TextDecoder().decode(built.files.get('evidence.json')!));
  assert.ok(ev.final.middleSettlementRate.count >= 1, JSON.stringify(ev.final.middleSettlementRate));
  assert.ok(ev.final.mechanism.middle >= 1);
  assert.deepEqual(ev.final.pathMismatches, []);
});

test('loading checks: the middle encoder is required, must match, and a middle file must pair with the classifier, the edge and the evidence', async () => {
  const built = await middleRelease();
  await assert.rejects(loadRouter(options(built)), code('CAPABILITY_MISSING'));
  await assert.rejects(loadRouter(options(built, { middleEncoder: middleEncoder({ ...MIDDLE_IDENTITY, revision: 'sha256:1111' }).encoder })), code('ENCODER_IDENTITY_MISMATCH'));
  await assert.rejects(loadRouter(options(await (await import('./helpers.ts')).makeRelease(), { middleEncoder: middleEncoder().encoder })), code('OPTIONS_INVALID'));
  // Moved thresholds: the head was certified with other middle thresholds.
  await assert.rejects(middleRelease({ middle: (m) => { m.heads.urgent!.fire = 3; } }), (e: unknown) => e instanceof RouterLoadError && e.code === 'PAIRING_MISMATCH' && e.details.some((d) => /certified with middle model/.test(d)));
  // Certified behind another edge model.
  await assert.rejects(middleRelease({ middle: (m) => { m.heads.urgent!.claims.prior_edge_sha256 = 'e'.repeat(64); } }), (e: unknown) => e instanceof RouterLoadError && e.details.some((d) => /certified behind edge model/.test(d)));
  // A middle model for a head that wasn't certified with it.
  await assert.rejects(middleRelease({ artifact: (a) => { a.heads.urgent = { ...a.heads.urgent!, middle: undefined }; } }), (e: unknown) => e instanceof RouterLoadError && e.details.some((d) => /not certified with it/.test(d)));
  await assert.rejects(middleRelease({ evidenceMiddle: 'omit' }), (e: unknown) => e instanceof RouterLoadError && e.details.some((d) => /different middle document/.test(d)));
  await assert.rejects(middleRelease({ middleEmbedding: null }), code('MANIFEST_INVALID'));
  await assert.rejects(middleRelease({ middleEmbedding: { ...MIDDLE_IDENTITY, dimensions: 3 } }), code('EMBEDDING_MISMATCH'));
  await assert.rejects(middleRelease({ middle: (m) => { (m as unknown as { schema: string }).schema = 'liquidau-middle/9'; } }), code('MIDDLE_INVALID'));
});

test('a split deployment: the rules tier loads a middle release, and forwarded requests run the middle tier in the model tier', async () => {
  const built = await middleRelease();
  const main = fakeEncoder(), mid = middleEncoder();
  const model = await loadRouter(options(built, { encoder: main.encoder, middleEncoder: mid.encoder }));
  const rulesTier = await loadRulesTier({
    source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000,
    monitoring: { sampleRate: 0, samplingSalt: 'salt' },
  });
  // The rules tier can't settle it (the edge leaves urgent), so it forwards; the model tier's middle tier settles it.
  const outcome = await rulesTier.handle({ text: 'my card is stuck and I am scared', requestId: 's1' });
  assert.equal(outcome.answered, false);
  const r = await model.routeForwarded(!outcome.answered ? outcome.forward : (null as never));
  assert.equal(r?.outcome === 'route' && r.mechanism, 'middle');
  // The edge settles "stolen" in the rules tier itself.
  const edgeOutcome = await rulesTier.handle({ text: 'my card was stolen', requestId: 's2' });
  assert.equal(edgeOutcome.answered, true);
  assert.deepEqual(main.calls, []);
  await rulesTier.close();
  await model.close();
});

test('several fired heads: every one counts as fired, so serving decides what was certified', async () => {
  // The review's case: priority [a, b], b suppresses a; the edge fires both, and b's model score is low.
  const policy = { ...POLICY, priority: ['urgent', 'refund', 'other'], suppress: [{ when: ['refund'], heads: ['urgent'] }] };
  const model = { heads: { urgent: { threshold: 0.8, reviewFloor: 0.3 }, refund: { threshold: 0.7, reviewFloor: 0.4 }, other: { threshold: 0.6, reviewFloor: 0.5 } } };
  const ev = createDecisionEvaluator(model, policy as typeof POLICY);
  const findings = { fired: { id: 'edge:urgent', label: 'urgent', source: 'edge' as const }, firing: [{ id: 'edge:urgent', label: 'urgent', source: 'edge' as const }, { id: 'edge:refund', label: 'refund', source: 'edge' as const }], dismissed: [] };
  // refund fired suppresses urgent for certain: refund routes, settled without any score.
  const settled = ev.settle(findings);
  assert.deepEqual(settled, { outcome: 'route', routeId: 'refund', destination: 'queue:refunds', mechanism: 'edge' });
  assert.deepEqual(ev.decide(findings, { other: 0.01 }), settled, 'decide agrees with settle');
  assert.deepEqual(ev.requiredScores(findings), ['other'], 'fired heads need no score');
  // The edge tier reports every fired head.
  const f = lexicalFeatures('stolen');
  const both: EdgeDocument<Head> = { ...edge(), heads: { urgent: edge().heads.urgent!, refund: { scorers: ['lexical'], lexical: { bias: 0, index: [...f.index], weight: [...f.index].map(() => 1) }, blend: { coef: [10], intercept: -1 }, fire: 0.5, dismiss: null, claims: { max_false_alarms: 0.005, max_dismissed: 0, delta: 0.05 } } } };
  const tier = createEdgeTier(ruleSet(), both, policy.priority);
  assert.deepEqual(tier.evaluate('card stolen').findings.firing?.map((x) => x.label), ['urgent', 'refund']);
});
