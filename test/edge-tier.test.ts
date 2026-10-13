/**
 * Releases with an edge tier: the edge's fires and dismissals become rule findings, the release pairs
 * the edge file with the classifier and the evidence, a static edge needs the matching static encoder,
 * and a split deployment's edge function answers exactly as one router.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { edgeHeadDigest, scoreEmbedding, type ClassifierArtifact, type EdgeDocument } from '@liquidau/embedding-classifier';
import { ruleSetHash, type RuleSet } from '@liquidau/rule-miner';
import { createStaticEmbedder, DEFAULT_LEXICAL, lexicalFeatures, STATIC_SCHEMA, type StaticEmbedder } from '@liquidau/text-preprocessing';

import {
  buildRelease, createDecisionEvaluator, createEdgeTier, documentDigest, encodeJson, loadRouter, loadRulesTier, memorySource, RouterLoadError,
  type BuiltRelease, type LoadRulesTierOptions, type RouterEvent, type RouteResult,
} from '../src/index.ts';
import { buildEvidence, evaluateFinal, evaluatePerHead, type FinalItem } from '../src/evaluation/index.ts';
import { artifact, EVAL, fakeEncoder, HEADS, options, POLICY, ruleSet, vectorFor, type Head } from './helpers.ts';

const code = (c: string) => (e: unknown) => e instanceof RouterLoadError && e.code === c;

/** An edge whose lexical model fires `urgent` on "stolen" and dismisses it on everything else. */
function lexicalEdge(): EdgeDocument<Head> {
  const f = lexicalFeatures('stolen');
  return {
    schema: 'liquidau-edge/1', lexical: { ...DEFAULT_LEXICAL }, static: null, vector: null,
    heads: { urgent: { scorers: ['lexical'], lexical: { bias: 0, index: [...f.index], weight: [...f.index].map(() => 1) }, blend: { coef: [10], intercept: -1 }, fire: 0.5, dismiss: -0.5, claims: { max_false_alarms: 0.005, max_dismissed: 0.075, delta: 0.05 } } },
  };
}

const TOKENIZER = {
  normalizer: { type: 'BertNormalizer', clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
  pre_tokenizer: { type: 'BertPreTokenizer' },
  model: { type: 'WordPiece', unk_token: '[UNK]', continuing_subword_prefix: '##', max_input_chars_per_word: 100, vocab: { '[UNK]': 0, money: 1, back: 2, want: 3 } },
};
const staticEmbedder = (weights = [0, 0, 1, 0, 1, 0, 0, 1]) => createStaticEmbedder({ modelId: 'fixture-static', tokenizerBytes: new TextEncoder().encode(JSON.stringify(TOKENIZER)), weights: new Float32Array(weights), dimensions: 2 });

/** A release whose `urgent` head was certified with `edge` (unless tweaked), evaluated through the edge tier. */
async function edgeRelease(edge: EdgeDocument<Head>, tweak: { artifact?: (a: ClassifierArtifact<Head>) => void; evidenceEdge?: 'omit' | 'wrong'; edgeBytes?: Uint8Array; staticEncoder?: StaticEmbedder; minCoverage?: number } = {}): Promise<BuiltRelease> {
  const rules: RuleSet = ruleSet();
  const sha = edgeHeadDigest(edge, 'urgent')!;
  const art = artifact(rules);
  art.heads.urgent = { ...art.heads.urgent!, edge: { model_sha256: sha } };
  tweak.artifact?.(art);
  const edgeBytes = tweak.edgeBytes ?? encodeJson(edge);
  const documents = { classifier: encodeJson(art), rules: encodeJson(rules), policy: encodeJson(POLICY), edge: edgeBytes };
  const tier = createEdgeTier(rules, edge, POLICY.priority, tweak.staticEncoder);
  const items: FinalItem[] = EVAL.map((e, i) => ({ id: String(i), rules: tier.evaluate(e.text).findings, scores: scoreEmbedding(art, vectorFor(e.p)) as Record<string, number>, acceptable: e.acceptable }));
  const model = { heads: Object.fromEntries(Object.entries(art.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) };
  const final = evaluateFinal(createDecisionEvaluator(model, POLICY), items);
  const perHead = evaluatePerHead(model, items.map((it, i) => ({ rules: it.rules, scores: it.scores, labels: Object.fromEntries(HEADS.map((h) => [h, EVAL[i].acceptable.includes(h) ? 1 : 0])) })));
  const edgeSha256 = tweak.evidenceEdge === 'omit' ? undefined : tweak.evidenceEdge === 'wrong' ? 'f'.repeat(64) : await documentDigest(edgeBytes);
  const evidence = buildEvidence({
    inputs: { classifierSha256: await documentDigest(documents.classifier), rulesSha256: await documentDigest(documents.rules), policySha256: await documentDigest(documents.policy), rulesSemanticHash: ruleSetHash(rules), ...(edgeSha256 ? { edgeSha256 } : {}) },
    dataset: { id: 'fixture-test', split: 'test' }, final, perHead, criteria: { minCoverage: tweak.minCoverage ?? 0.5, compare: 'point' },
  });
  return buildRelease({
    releaseId: 'release-edge', createdAt: '2026-10-09T00:00:00Z', documents: { ...documents, evidence: encodeJson(evidence) }, embedding: fakeEncoder().encoder.identity,
    serving: { maxInputUtf8Bytes: 4096 }, packages: { '@liquidau/embedding-classifier': '0.9.0', '@liquidau/rule-miner': '0.5.2', '@liquidau/router': '0.3.0', '@liquidau/text-preprocessing': '0.2.0' },
  });
}

test('the edge fires and dismisses: an edge route needs no embedding, a dismissal reaches the model tier', async () => {
  const built = await edgeRelease(lexicalEdge());
  assert.equal(built.manifest.files.edge?.key, 'edge.json');
  const { encoder, calls } = fakeEncoder({ 'I want my money back': { refund: 0.9 }, 'my card was stolen': { urgent: 0.1 } });
  const events: RouterEvent[] = [];
  const router = await loadRouter(options(built, { encoder, observer: (e) => { events.push(e); } }));
  const stolen = await router.route({ text: 'my card was stolen', requestId: 'a' });
  assert.deepEqual([stolen.outcome, (stolen as { mechanism?: string }).mechanism, stolen.embedded], ['route', 'edge', false]);
  assert.equal(calls.length, 0, 'settled at the edge: nothing embedded');
  const money = await router.route({ text: 'I want my money back', requestId: 'b' });
  assert.deepEqual([money.outcome, (money as { routeId?: string }).routeId, (money as { mechanism?: string }).mechanism], ['route', 'refund', 'classifier']);
  await router.flush();
  const ea = events.find((e) => e.type === 'decision' && e.requestId === 'a')!, eb = events.find((e) => e.type === 'decision' && e.requestId === 'b')!;
  assert.deepEqual([(ea as { firedRule: string }).firedRule, (ea as { edge?: unknown }).edge], ['edge:urgent', { fired: ['urgent'], dismissed: [] }]);
  assert.deepEqual((eb as unknown as { edge?: unknown }).edge, { fired: [], dismissed: ['urgent'] });
  assert.ok((eb as unknown as { dismissed: string[] }).dismissed.includes('urgent'));
  // A rule beats the edge's dismissal of its own label; a rule's dismissal beats an edge fire.
  const tier = createEdgeTier(ruleSet(), lexicalEdge(), POLICY.priority);
  assert.deepEqual(tier.evaluate('emergency at home').findings.fired, { id: 'urgent.u1', label: 'urgent', source: 'rule' });
  assert.ok(!tier.evaluate('emergency at home').findings.dismissed.includes('urgent'));
  assert.deepEqual(tier.evaluate('hello stolen').findings, { fired: null, firing: [], dismissed: ['other', 'urgent', 'refund'] });
});

test('pairing: the classifier, the evidence and the manifest must all name the same edge', async () => {
  await assert.rejects(edgeRelease(lexicalEdge(), { artifact: (a) => { delete a.heads.urgent!.edge; } }), (e: unknown) => code('PAIRING_MISMATCH')(e) && /not certified with it/.test(JSON.stringify((e as RouterLoadError).details)));
  await assert.rejects(edgeRelease(lexicalEdge(), { artifact: (a) => { a.heads.urgent!.edge = { model_sha256: 'c'.repeat(64) }; } }), (e: unknown) => code('PAIRING_MISMATCH')(e) && /certified with edge model/.test(JSON.stringify((e as RouterLoadError).details)));
  await assert.rejects(edgeRelease(lexicalEdge(), { evidenceEdge: 'omit' }), (e: unknown) => code('PAIRING_MISMATCH')(e) && /different edge/.test(JSON.stringify((e as RouterLoadError).details)));
  await assert.rejects(edgeRelease(lexicalEdge(), { evidenceEdge: 'wrong' }), code('PAIRING_MISMATCH'));
  const bad = { ...lexicalEdge(), heads: { urgent: { ...lexicalEdge().heads.urgent!, blend: { coef: [1, 2], intercept: 0 } } } } as EdgeDocument<Head>;
  await assert.rejects(edgeRelease(lexicalEdge(), { edgeBytes: encodeJson(bad) }), code('EDGE_INVALID'));
  // A corrupted edge file fails the digest check at load.
  const built = await edgeRelease(lexicalEdge());
  const files = new Map(built.files);
  files.set('edge.json', new TextEncoder().encode(JSON.stringify(lexicalEdge()) + ' '));
  await assert.rejects(loadRouter({ ...options(built), source: memorySource(files) }), code('FILE_DIGEST_MISMATCH'));
});

test('a static edge needs the exact static encoder it was trained with', async () => {
  const st = await staticEmbedder();
  const edge: EdgeDocument<Head> = {
    schema: 'liquidau-edge/1', lexical: null, static: { ...st.identity }, vector: null,
    heads: { urgent: { scorers: ['static'], static: { coef: [-5, 5], intercept: 0 }, blend: { coef: [1], intercept: 0 }, fire: 1, dismiss: -1, claims: { max_false_alarms: 0.005, max_dismissed: 0.075, delta: 0.05 } } },
  };
  const built = await edgeRelease(edge, { staticEncoder: st, minCoverage: 0.3 });
  await assert.rejects(loadRouter(options(built)), code('STATIC_ENCODER_MISMATCH'));
  await assert.rejects(loadRouter(options(built, { staticEncoder: await staticEmbedder([0, 0, 1, 0, 1, 0, 0, 2]) })), code('STATIC_ENCODER_MISMATCH'));
  const { encoder, calls } = fakeEncoder();
  const router = await loadRouter(options(built, { encoder, staticEncoder: st }));
  // "money back": both tokens point at dimension 0, so the static score is -5: urgent dismissed.
  const r = await router.route({ text: 'I want my money back', requestId: 'm' });
  assert.equal((r as { routeId?: string }).routeId, 'refund');
  assert.equal(calls.length, 1);
  assert.equal(st.identity.schema, STATIC_SCHEMA);
  // A loader in place of the encoder: called for a static edge, never for an edge without one.
  let loads = 0;
  const lazy = await loadRouter(options(built, { encoder: fakeEncoder().encoder, staticEncoder: () => { loads++; return st; } }));
  assert.equal(((await lazy.route({ text: 'I want my money back', requestId: 'n' })) as { routeId?: string }).routeId, 'refund');
  await loadRouter(options(await edgeRelease(lexicalEdge()), { staticEncoder: () => { loads++; return st; } }));
  assert.equal(loads, 1);
});

test('an edge that fails on a message answers unavailable, and an edge needing a host vector is refused at load', async () => {
  const st = await staticEmbedder();
  const edge: EdgeDocument<Head> = {
    schema: 'liquidau-edge/1', lexical: null, static: { ...st.identity }, vector: null,
    heads: { urgent: { scorers: ['static'], static: { coef: [-5, 5], intercept: 0 }, blend: { coef: [1], intercept: 0 }, fire: 1, dismiss: -1, claims: { max_false_alarms: 0.005, max_dismissed: 0.075, delta: 0.05 } } },
  };
  const built = await edgeRelease(edge, { staticEncoder: st, minCoverage: 0.3 });
  const broken: StaticEmbedder = { identity: st.identity, embed: () => { throw new Error('weights unavailable'); } } as StaticEmbedder;
  const router = await loadRouter(options(built, { staticEncoder: broken }));
  const r = await router.route({ text: 'I want my money back', requestId: 'x' });
  assert.equal(r.outcome, 'unavailable');
  assert.equal((r as { code?: string }).code, 'PREPROCESSING_FAILURE');
  const rules = await loadRulesTier({ source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000, monitoring: { sampleRate: 0, samplingSalt: 's' }, staticEncoder: broken });
  const o = await rules.handle({ text: 'I want my money back', requestId: 'y' });
  assert.equal(o.answered && o.result.outcome, 'unavailable');
  // The 'vector' scorer needs a host vector the edge tier never has (it runs before any encoder).
  const vectorEdge: EdgeDocument<Head> = { ...lexicalEdge(), vector: { model_id: 'v', dimensions: 2 }, heads: { urgent: { scorers: ['vector'], vector: { coef: [1, 1], intercept: 0 }, blend: { coef: [1], intercept: 0 }, fire: 1, dismiss: null, claims: { max_false_alarms: 0.005, max_dismissed: 0, delta: 0.05 } } } };
  assert.throws(() => createEdgeTier(ruleSet(), vectorEdge, POLICY.priority), code('CAPABILITY_MISSING'));
});

test('split deployment with an edge: the edge function answers exactly as one router', async () => {
  const built = await edgeRelease(lexicalEdge());
  const table = { ...Object.fromEntries(EVAL.map((e) => [e.text, e.p])), 'my card was stolen': { urgent: 0.1 }, 'stolen money back please': { refund: 0.95 } };
  const router = await loadRouter(options(built, { encoder: fakeEncoder(table).encoder }));
  const model = await loadRouter(options(built, { encoder: fakeEncoder(table).encoder }));
  const rulesOptions: LoadRulesTierOptions = { source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000, monitoring: { sampleRate: 0, samplingSalt: 'salt' } };
  const edgeFn = await loadRulesTier(rulesOptions);
  let answered = 0;
  for (const [i, text] of [...EVAL.map((e) => e.text), 'my card was stolen', 'stolen money back please', 'nothing'].entries()) {
    const request = { text, requestId: `s${i}` };
    const expected = await router.route(request);
    const out = await edgeFn.handle(request);
    let actual: RouteResult | null;
    if (out.answered) { answered++; actual = out.result; } else actual = await model.routeForwarded(JSON.parse(JSON.stringify(out.forward)));
    assert.deepEqual(actual, expected, text);
  }
  assert.ok(answered >= 2, `the edge function answered ${answered}`);
});
