/**
 * Releases with context features: the classifier scores the message's embedding followed by its
 * context's (embedding-classifier's embedding.context), each from its own encoder.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { ClassifierArtifact } from '@liquidau/embedding-classifier';

import { FORWARD_SCHEMA, functionEncoder, loadRouter, loadRulesTier, memorySource, type ForwardedRequest, type RouteResult } from '../src/index.ts';
import { IDENTITY, makeRelease, options, vectorFor, type Head } from './helpers.ts';

const CONTEXT = { ...IDENTITY, modelId: 'fake/context', dimensions: 2 };

/** Every head gets two context weights; refund's first is 10, so a context of [1.5, 0] puts refund over its threshold. */
const withContext = (a: ClassifierArtifact<Head>) => {
  a.embedding.context = { model_id: 'fake/context', dimensions: 2, normalize: false, precision: 'fp32' };
  for (const h of ['urgent', 'refund', 'other'] as const) a.heads[h] = { ...a.heads[h]!, weights: [...a.heads[h]!.weights, h === 'refund' ? 10 : 0, 0] };
};
const contextRelease = () => makeRelease({ artifact: withContext, contextEmbedding: CONTEXT });
const message = functionEncoder(IDENTITY, () => vectorFor({}));
const contextEncoder = functionEncoder(CONTEXT, (t) => (t === 'refund context' ? [1.5, 0] : [0, 0]));
const outcome = (r: RouteResult) => (r.outcome === 'route' ? r.routeId : r.outcome === 'unavailable' ? r.code : r.outcome);

test('a release with context features embeds the context with its own encoder and scores both', async () => {
  const router = await loadRouter(options(await contextRelease(), { encoder: message, contextEncoder }));
  assert.equal(outcome(await router.route({ text: 'nothing', context: 'refund context', requestId: '1' })), 'refund');
  assert.notEqual(outcome(await router.route({ text: 'nothing', context: 'other context', requestId: '2' })), 'refund');
  // Context is required, and limited like the text.
  assert.equal(outcome(await router.route({ text: 'nothing', requestId: '3' })), 'INVALID_INPUT');
  assert.equal(outcome(await router.route({ text: 'nothing', context: 'x'.repeat(5000), requestId: '4' })), 'INVALID_INPUT');
  // A forwarded request carries its context.
  const forward: ForwardedRequest = { schema: FORWARD_SCHEMA, manifestSha256: router.manifestSha256, requestId: '5', text: 'nothing', context: 'refund context', purpose: 'decide', inclusionProbability: 1 };
  assert.equal(outcome((await router.routeForwarded(forward))!), 'refund');
});

test('context encoders are required, checked against the release, and refused without context features', async () => {
  const built = await contextRelease();
  await assert.rejects(loadRouter(options(built, { encoder: message })), (e: { code?: string }) => e.code === 'CAPABILITY_MISSING');
  await assert.rejects(loadRouter(options(built, { encoder: message, contextEncoder: functionEncoder({ ...CONTEXT, modelId: 'other' }, () => [0, 0]) })), (e: { code?: string }) => e.code === 'ENCODER_IDENTITY_MISMATCH');
  const plain = await makeRelease();
  await assert.rejects(loadRouter(options(plain, { contextEncoder })), (e: { code?: string }) => e.code === 'OPTIONS_INVALID');
  const router = await loadRouter(options(plain));
  assert.equal(outcome(await router.route({ text: 'what is the weather', context: 'anything', requestId: '1' })), 'INVALID_INPUT', 'a release without context features refuses context');
});

test('the classifier\'s context spec and the release\'s context encoder must match', async () => {
  await assert.rejects(makeRelease({ artifact: withContext }), (e: { code?: string }) => e.code === 'EMBEDDING_MISMATCH');
  await assert.rejects(makeRelease({ contextEmbedding: CONTEXT }), (e: { code?: string }) => e.code === 'EMBEDDING_MISMATCH');
  await assert.rejects(makeRelease({ artifact: withContext, contextEmbedding: { ...CONTEXT, dimensions: 3 } }), (e: { code?: string }) => e.code === 'EMBEDDING_MISMATCH');
});

test('the rules tier forwards the context, and the model tier decides with it', async () => {
  const built = await contextRelease();
  const rules = await loadRulesTier({ source: memorySource(built.files), manifestKey: built.manifestKey, expectedManifestSha256: built.manifestSha256, mode: 'enforce', timeoutMs: 1000, monitoring: { sampleRate: 0, samplingSalt: 'salt' } });
  const model = await loadRouter(options(built, { encoder: message, contextEncoder }));
  const outcomeOf = await rules.handle({ text: 'nothing', context: 'refund context', requestId: '1' });
  assert.ok(!outcomeOf.answered);
  const forward = JSON.parse(JSON.stringify(outcomeOf.forward));
  assert.equal(forward.context, 'refund context');
  assert.equal(outcome((await model.routeForwarded(forward))!), 'refund');
  // The rules tier refuses a request without the context the release needs.
  const missing = await rules.handle({ text: 'nothing', requestId: '2' });
  assert.ok(missing.answered && missing.result.outcome === 'unavailable' && missing.result.code === 'INVALID_INPUT');
});
