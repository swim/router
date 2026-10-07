/**
 * Document releases ('liquidau-router/2'): a classifier trained on pooled document features through
 * embedding-classifier's document API, rules, policy and '/2' evidence, built, loaded and served.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildDocumentArtifact, fullEncoderIdentityProblems, prepareDocumentFeatures, scoreDocument, trainDocumentHeads, type DocumentClassifierArtifact, type HeadPolicy, type Split,
} from '@liquidau/embedding-classifier';
import { buildRuleSet, ruleSetHash, ruleSetMatcher } from '@liquidau/rule-miner';
import { createPipeline, fixtureTokenizer } from '@liquidau/text-preprocessing';

import {
  buildRelease, createDecisionEvaluator, documentDigest, encodeJson, encoderIdentityProblems, loadRouter, memorySource, RouterLoadError, sha256Hex,
  type BuiltRelease, type Encoder, type EncoderIdentity, type LoadRouterOptions, type ReleaseEvidence, type RouteResult, type RouterEvent, type RouterPolicy,
} from '../src/index.ts';
import { buildEvidence, evaluateFinal, type FinalItem } from '../src/evaluation/index.ts';

const tokenizer = fixtureTokenizer();
const IDENTITY: EncoderIdentity = {
  schema: 'liquidau-encoder/1', modelId: 'fake/doc-model', revision: 'sha256:abc', dimensions: 4, precision: 'fp32', inputType: null, layers: null, pooling: 'mean',
  normalization: 'none', tokenizerRevision: 'fixture-tokenizer/1', maxChars: null, maxTokens: 64, truncation: 'none',
};
const pipeline = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24, maxChunks: 16 });
const vec = (t: string) => {
  let h = 7;
  for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return [/refund/i.test(t) ? 1 : 0, ((h % 1000) / 1000 - 0.5) * 0.6, /urgent/i.test(t) ? 1 : 0, 1];
};

/** A fake chunk encoder; counts calls and texts; optional per-call delay or failure. */
function encoder(options: { delayMs?: number; failCall?: number } = {}): Encoder & { calls: string[][] } {
  const calls: string[][] = [];
  return {
    identity: IDENTITY, calls,
    async embed(texts, { signal }) {
      const call = calls.push([...texts]) - 1;
      if (options.delayMs) await new Promise((r) => setTimeout(r, options.delayMs));
      if (signal.aborted) throw new Error('aborted');
      if (call === options.failCall) throw new Error('provider 503');
      return texts.map(vec);
    },
  };
}

type H = 'urgent' | 'refund';
const precision: HeadPolicy = { kind: 'precision', mode: 'heuristic', targetPrecision: 0.8 };
const POLICY: RouterPolicy = {
  schema: 'liquidau-router-policy/1', routes: [{ id: 'urgent', destination: 'queue:urgent' }, { id: 'refund', destination: 'queue:refunds' }],
  priority: ['urgent', 'refund'], suppress: [], onAbstain: 'none',
};
const RULES = buildRuleSet('rules-doc', [{ label: 'urgent', rules: [{ id: 'e', pattern: { kind: 'phrase', tokens: ['emergency'] } }] }], { createdAt: '2026-10-07T00:00:00Z' });

function corpus(n: number) {
  const docs = [], labels: Array<Record<H, 0 | 1>> = [], split: Split[] = [];
  for (let i = 0; i < n; i++) {
    const refund = i % 3 === 0, urgent = i % 5 === 2;
    const s = Array.from({ length: 2 + (i % 6) }, (_, k) => `Filler sentence ${k} about the weather today.`);
    if (refund) s.splice((i * 7) % s.length, 0, 'Please refund my order.');
    if (urgent) s.splice((i * 3) % s.length, 0, 'This is urgent, help.');
    docs.push({ id: `d${i}`, groupId: `g${Math.floor(i / 2)}`, text: s.join(' ') });
    labels.push({ urgent: urgent ? 1 : 0, refund: refund ? 1 : 0 });
    const g = Math.floor(i / 2) % 10;
    split.push(g < 6 ? 'train' : g < 8 ? 'calibration' : 'test');
  }
  return { docs, labels, split };
}

const cache = new Map<string, { artifact: DocumentClassifierArtifact<H>; testItems: Array<{ text: string; acceptable: H[] }> }>();
const BOUNDARY: EncoderIdentity = { ...IDENTITY, modelId: 'fake/boundary', revision: 'sha256:b' };
const semanticPipeline = createPipeline({
  splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24, maxChunks: 16,
  semantic: { threshold: 0.95, boundaryEncoderIdentity: BOUNDARY, boundaryTokenizer: tokenizer.identity, boundaryMaxInputTokens: 24 },
});
const boundaryEncoder = (): Encoder => ({ ...encoder(), identity: BOUNDARY });

async function trained(p = pipeline) {
  const key = JSON.stringify(p);
  if (cache.has(key)) return cache.get(key)!;
  const { docs, labels, split } = corpus(300);
  const semantic = p.grouping.algorithm === 'adjacent-cosine/1';
  const features = await prepareDocumentFeatures(docs, p, { tokenizer, encoder: encoder(), ...(semantic ? { boundaryEncoder: boundaryEncoder(), boundaryTokenizer: tokenizer } : {}) });
  const result = trainDocumentHeads<H>({
    features, encoderIdentity: IDENTITY, pipeline: p, split,
    heads: (['urgent', 'refund'] as const).map((h) => ({ name: h, y: labels.map((l) => l[h]), prevalence: h === 'refund' ? 1 / 3 : 1 / 5, policy: precision })),
  });
  assert.deepEqual(result.failures, [], result.failures.join('; '));
  const artifact = buildDocumentArtifact(result, { version: 'doc-1', createdAt: '2026-10-07T00:00:00Z', router: { ruleSetHash: ruleSetHash(RULES) } });
  const testItems = docs.flatMap((d, i) => (split[i] === 'test' ? [{ text: d.text, acceptable: (['urgent', 'refund'] as const).filter((h) => labels[i][h]) as H[] }] : []));
  cache.set(key, { artifact, testItems });
  return cache.get(key)!;
}

const SERVING = { maxInputUtf8Bytes: 20_000 };
const PACKAGES = { '@liquidau/embedding-classifier': '0.8.0', '@liquidau/rule-miner': '0.5.1', '@liquidau/router': '0.1.0', '@liquidau/text-preprocessing': '0.1.0' };

async function makeDocumentRelease(tweak: { evidence?: (e: ReleaseEvidence) => void; legacyEvidence?: boolean; pipeline?: typeof pipeline } = {}): Promise<BuiltRelease> {
  const { artifact, testItems } = await trained(tweak.pipeline);
  const semantic = artifact.pipeline.grouping.algorithm === 'adjacent-cosine/1';
  const documents = { classifier: encodeJson(artifact), rules: encodeJson(RULES), policy: encodeJson(POLICY) };
  const matcher = ruleSetMatcher(RULES);
  const items: FinalItem[] = [];
  for (const [i, it] of testItems.entries()) {
    // Offline evaluation calls the same document scorer the router serves with.
    const scored = await scoreDocument(artifact, it.text, { tokenizer, encoder: encoder(), ...(semantic ? { boundaryEncoder: boundaryEncoder(), boundaryTokenizer: tokenizer } : {}) });
    items.push({ id: String(i), rules: matcher.evaluate(it.text), scores: scored.scores as Record<string, number>, acceptable: it.acceptable });
  }
  const model = { heads: Object.fromEntries(Object.entries(artifact.classifier.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) };
  const final = evaluateFinal(createDecisionEvaluator(model, POLICY), items);
  const evidence = buildEvidence({
    inputs: { classifierSha256: await documentDigest(documents.classifier), rulesSha256: await documentDigest(documents.rules), policySha256: await documentDigest(documents.policy), rulesSemanticHash: ruleSetHash(RULES) },
    dataset: { id: 'doc-test', split: 'test' }, final, criteria: { minCoverage: 0.3 },
    ...(tweak.legacyEvidence ? {} : { document: { featureIdentitySha256: artifact.featureIdentitySha256, servingMaxInputUtf8Bytes: SERVING.maxInputUtf8Bytes } }),
  });
  tweak.evidence?.(evidence);
  return buildRelease({ releaseId: 'doc-release-1', createdAt: '2026-10-07T00:00:00Z', documents: { ...documents, evidence: encodeJson(evidence) }, embedding: IDENTITY, serving: SERVING, packages: PACKAGES });
}

const options = (built: BuiltRelease, over: Partial<LoadRouterOptions> = {}): LoadRouterOptions => ({
  source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', encoder: encoder(),
  tokenizer, timeoutMs: 2000, monitoring: { sampleRate: 0, samplingSalt: 's' }, ...over,
});
const semantic = (r: RouteResult) => (r.outcome === 'route' ? [r.outcome, r.routeId, r.mechanism] : r.outcome === 'unavailable' ? [r.outcome, r.code] : r.outcome === 'shadow' ? ['shadow', r.candidate] : [r.outcome, r.reason]);
const code = (c: string) => (e: unknown) => e instanceof RouterLoadError && e.code === c;

test('a document release builds as liquidau-router/2, enforces, and serves the shared document scorer', async () => {
  const built = await makeDocumentRelease();
  const { artifact, testItems } = await trained();
  assert.equal(built.manifest.schema, 'liquidau-router/2');
  assert.equal(built.manifest.featureIdentitySha256, artifact.featureIdentitySha256);
  assert.deepEqual(built.enforcement, []);
  const events: RouterEvent[] = [];
  const enc = encoder();
  const router = await loadRouter(options(built, { encoder: enc, observer: (e) => { events.push(e); } }));
  const long = testItems.find((t) => t.acceptable.includes('refund') && t.text.length > 200)!.text;
  const r = await router.route({ text: long, requestId: 'r1' });
  assert.deepEqual(semantic(r), ['route', 'refund', 'classifier']);
  const direct = await scoreDocument(artifact, long, { tokenizer, encoder: encoder() });
  const [e] = events;
  assert.ok(e.type === 'decision' && e.document && e.document.chunkCount > 1 && e.document.chunkCount === direct.chunkCount);
  assert.deepEqual(e.type === 'decision' && e.calibratedProbabilities, direct.scores, 'router and offline evaluation score identically');
  assert.equal(JSON.stringify(events).includes('weather'), false, 'no text or chunks in telemetry');
  assert.ok(enc.calls.length >= 1);
});

test('rules run first on the original text: a settled document is never chunked; monitoring shares 0 and 1 agree', async () => {
  const built = await makeDocumentRelease();
  const enc = encoder();
  const router = await loadRouter(options(built, { encoder: enc }));
  const r = await router.route({ text: 'Emergency! '.repeat(50), requestId: 's1' });
  assert.deepEqual(semantic(r), ['route', 'urgent', 'rule']);
  assert.equal(enc.calls.length, 0);
  const { testItems } = await trained();
  const texts = ['Emergency now.', ...testItems.slice(0, 12).map((t) => t.text), '   '];
  const out: unknown[][] = [];
  for (const sampleRate of [0, 1]) {
    const rt = await loadRouter(options(built, { mode: 'shadow', monitoring: { sampleRate, samplingSalt: 's' } }));
    out.push(await Promise.all(texts.map(async (t, i) => semantic(await rt.route({ text: t, requestId: String(i) })))));
  }
  assert.deepEqual(out[0], out[1]);
});

test('one deadline covers the whole document, not a fresh timeout per chunk', async () => {
  const built = await makeDocumentRelease();
  const { testItems } = await trained();
  const long = testItems.reduce((a, b) => (b.text.length > a.text.length ? b : a)).text;
  const chunks = (await scoreDocument((await trained()).artifact, long, { tokenizer, encoder: encoder() })).chunkCount;
  assert.ok(chunks >= 4, `${chunks} chunks`);
  // Each call takes 40 ms; one call per chunk; a 100 ms deadline must expire mid-document.
  const slow = await loadRouter(options(built, { encoder: encoder({ delayMs: 40 }), timeoutMs: 100, document: { batchSize: 1, concurrency: 1 } }));
  const t0 = performance.now();
  const r = await slow.route({ text: long, requestId: 'slow' });
  assert.deepEqual([r.outcome === 'unavailable' && r.code, r.outcome === 'unavailable' && r.retryable], ['PROCESSING_TIMEOUT', true]);
  assert.ok(performance.now() - t0 < 160, 'returned at the single deadline');
  const ok = await loadRouter(options(built, { encoder: encoder({ delayMs: 40 }), timeoutMs: 2000, document: { batchSize: 1, concurrency: 1 } }));
  assert.equal((await ok.route({ text: long, requestId: 'ok' })).outcome, 'route');
});

test('document failures are explicit and never route on partial chunk vectors', async () => {
  const built = await makeDocumentRelease();
  const { testItems } = await trained();
  const long = testItems.reduce((a, b) => (b.text.length > a.text.length ? b : a)).text;
  const outcome = async (o: Partial<LoadRouterOptions>, text = long, signal?: AbortSignal) => {
    const r = await (await loadRouter(options(built, o))).route({ text, requestId: 'x', signal });
    return r.outcome === 'unavailable' ? [r.code, r.retryable] : [r.outcome];
  };
  assert.deepEqual(await outcome({ encoder: encoder({ failCall: 1 }), document: { batchSize: 1 } }), ['ENCODER_FAILURE', true]);
  assert.deepEqual(await outcome({}, 'word. '.repeat(400)), ['INPUT_LIMIT_EXCEEDED', false], 'more chunks than the pipeline allows');
  assert.deepEqual(await outcome({}, ' \n\t '), ['INVALID_INPUT', false], 'whitespace-only documents are rejected before feature construction');
  const throwing = { identity: tokenizer.identity, countInput: () => { throw new Error('tokenizer bug'); }, cutOffsets: () => [] };
  assert.deepEqual(await outcome({ tokenizer: throwing }), ['PREPROCESSING_FAILURE', false]);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 20);
  assert.deepEqual(await outcome({ encoder: encoder({ delayMs: 40 }), document: { batchSize: 1, concurrency: 1 } }, long, controller.signal), ['ABORTED', false]);
});

test('loading: capabilities, tokenizer identity and every document binding are checked', async () => {
  const built = await makeDocumentRelease();
  await assert.rejects(loadRouter(options(built, { tokenizer: undefined })), code('CAPABILITY_MISSING'));
  await assert.rejects(loadRouter(options(built, { tokenizer: fixtureTokenizer({ special: 3 }) })), code('TOKENIZER_MISMATCH'));
  await assert.rejects(loadRouter(options(built, { encoder: { ...encoder(), identity: { ...IDENTITY, revision: 'sha256:other' } } })), code('ENCODER_IDENTITY_MISMATCH'));
  await assert.rejects(makeDocumentRelease({ evidence: (e) => { e.inputs.featureIdentitySha256 = 'f'.repeat(64); } }), code('PAIRING_MISMATCH'));
  await assert.rejects(makeDocumentRelease({ evidence: (e) => { e.inputs.servingMaxInputUtf8Bytes = 10_000; } }), code('PAIRING_MISMATCH'));
  await assert.rejects(makeDocumentRelease({ legacyEvidence: true }), code('UNSUPPORTED_SCHEMA'), "a document release needs '/2' evidence");
  await assert.rejects(makeDocumentRelease({ evidence: (e) => { delete e.criteria; } }), code('EVIDENCE_INVALID'));
  // An edited manifest feature identity, re-pinned, is still caught.
  const m = JSON.parse(JSON.stringify(built.manifest));
  m.featureIdentitySha256 = 'e'.repeat(64);
  const bytes = encodeJson(m);
  const files = new Map([...built.files, [built.manifestKey, bytes]]);
  await assert.rejects(loadRouter(options(built, { source: memorySource(files), expectedManifestSha256: await sha256Hex(bytes) })), code('PAIRING_MISMATCH'));
  // An envelope placed under a legacy '/1' manifest is refused by the legacy validator.
  const legacy = JSON.parse(JSON.stringify(built.manifest));
  legacy.schema = 'liquidau-router/1';
  delete legacy.featureIdentitySha256;
  const lb = encodeJson(legacy);
  await assert.rejects(loadRouter(options(built, { source: memorySource(new Map([...built.files, [built.manifestKey, lb]])), expectedManifestSha256: await sha256Hex(lb) })), code('CLASSIFIER_INVALID'));
});

test('conformance: the classifier and router encoder identity validators agree field for field', () => {
  const variants: unknown[] = [
    IDENTITY, { ...IDENTITY, layers: [2, 1], normalization: 'per-layer-unit' }, { ...IDENTITY, layers: [1, 1] }, { ...IDENTITY, dimensions: 0 }, { ...IDENTITY, extra: 1 },
    { ...IDENTITY, inputType: '' }, { ...IDENTITY, maxChars: 0 }, { ...IDENTITY, normalization: 'l2' }, { ...IDENTITY, normalization: 'per-layer-unit' }, { ...IDENTITY, schema: 'x' },
    { ...IDENTITY, layers: [1, 2, 3] }, { ...IDENTITY, maxTokens: null }, null, [],
  ];
  for (const v of variants) assert.equal(fullEncoderIdentityProblems(v).length === 0, encoderIdentityProblems(v).length === 0, JSON.stringify(v));
});

test('semantic document releases need and check the boundary adapters, then serve like any document release', async () => {
  const built = await makeDocumentRelease({ pipeline: semanticPipeline });
  const { artifact, testItems } = await trained(semanticPipeline);
  assert.equal(artifact.pipeline.grouping.algorithm, 'adjacent-cosine/1');
  await assert.rejects(loadRouter(options(built)), code('CAPABILITY_MISSING'));
  await assert.rejects(loadRouter(options(built, { boundaryEncoder: encoder(), boundaryTokenizer: tokenizer })), code('ENCODER_IDENTITY_MISMATCH'));
  await assert.rejects(loadRouter(options(built, { boundaryEncoder: boundaryEncoder(), boundaryTokenizer: fixtureTokenizer({ special: 3 }) })), code('TOKENIZER_MISMATCH'));
  const router = await loadRouter(options(built, { boundaryEncoder: boundaryEncoder(), boundaryTokenizer: tokenizer }));
  const long = testItems.find((t) => t.acceptable.includes('refund') && t.text.length > 200)!.text;
  const r = await router.route({ text: long, requestId: 'sem' });
  const direct = await scoreDocument(artifact, long, { tokenizer, encoder: encoder(), boundaryEncoder: boundaryEncoder(), boundaryTokenizer: tokenizer });
  const reference = createDecisionEvaluator({ heads: Object.fromEntries(Object.entries(artifact.classifier.heads).map(([h, s]) => [h, { threshold: s!.threshold, reviewFloor: s!.review_floor }])) }, POLICY)
    .decide(ruleSetMatcher(RULES).evaluate(long), direct.scores as Record<string, number>);
  assert.deepEqual(r.outcome === 'route' ? { outcome: 'route', routeId: r.routeId, destination: r.destination, mechanism: r.mechanism } : r, reference);
});

test('a synchronous planning stage that overruns the deadline still yields PROCESSING_TIMEOUT (review T4)', async () => {
  const built = await makeDocumentRelease();
  const busy = { identity: tokenizer.identity, countInput: (t: string) => { const end = performance.now() + 60; while (performance.now() < end) { /* blocks the event loop */ } return tokenizer.countInput(t); }, cutOffsets: tokenizer.cutOffsets };
  const router = await loadRouter(options(built, { tokenizer: busy, timeoutMs: 30, document: { yieldEvery: 1_000_000 } }));
  const r = await router.route({ text: 'Please refund my order. The weather is fine today.', requestId: 'busy' });
  assert.deepEqual([r.outcome, r.outcome === 'unavailable' && r.code], ['unavailable', 'PROCESSING_TIMEOUT']);
});
