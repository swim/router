/**
 * A release whose classifier has a finer head (embedding-classifier's finer-label route score): the
 * router serves it through scoreEmbedding, with the artifact's shared finer model.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { scoreEmbedding, type ClassifierArtifact } from '@liquidau/embedding-classifier';

import { functionEncoder, loadRouter } from '../src/index.ts';
import { IDENTITY, makeRelease, options, vectorFor, type Head } from './helpers.ts';

/** refund becomes a finer head: P(money | x) = σ(20·x[1] − 10), route rates [1, 0], so its logit is 20·x[1] − 10. */
const finerRefund = (a: ClassifierArtifact<Head>) => {
  a.finer = { classes: ['money', 'other'], coef: [[0, 20, 0, 0], [0, 0, 0, 0]], intercept: [-10, 0], C: 0.01, rows: 100 };
  a.heads.refund = { ...a.heads.refund!, weights: [0, 0, 0, 0, 4], bias: 0, features: { kind: 'finer', route_rates: [1, 0], scale: 0.25 } };
};

test('a finer head loads and routes through the artifact\'s finer model', async () => {
  const built = await makeRelease({ artifact: finerRefund });
  const route = async (p: Partial<Record<Head, number>>) => (await loadRouter(options(built, { encoder: functionEncoder(IDENTITY, () => vectorFor(p)) }))).route({ text: 'x', requestId: '1' });
  const money = await route({ refund: 0.9 });
  assert.equal(money.outcome, 'route');
  assert.equal(money.outcome === 'route' && money.routeId, 'refund');
  const weather = await route({ other: 0.9 });
  assert.ok(!(weather.outcome === 'route' && weather.routeId === 'refund'));
  // The served score is the library's: σ(20·x[1] − 10).
  const art = {} as ClassifierArtifact<Head>;
  Object.assign(art, { version: 'v', created_at: '', embedding: { model_id: 'fake/model', dimensions: 4, normalize: false }, heads: {} });
  finerRefund(Object.assign(art, { heads: { refund: { weights: [], bias: 0, calibration: { method: 'platt', a: 1, c: 0 }, threshold: 0.7, review_floor: 0.4 } } }));
  const x = vectorFor({ refund: 0.9 });
  assert.ok(Math.abs(scoreEmbedding(art, x).refund! - 1 / (1 + Math.exp(-(20 * x[1] - 10)))) < 1e-9);
});

test('a finer head without the artifact\'s finer model is refused when the release is built', async () => {
  await assert.rejects(makeRelease({ artifact: (a) => { finerRefund(a); delete a.finer; } }), (e: { code?: string; message?: string }) => e.code === 'CLASSIFIER_INVALID' && /no finer model/.test(e.message ?? ''));
});
