import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildRelease, encodeJson, functionEncoder, loadRouter, memorySource, RouterLoadError, sha256Hex, type BuiltRelease, type LoadErrorCode, type LoadRouterOptions } from '../src/index.ts';
import { fakeEncoder, IDENTITY, load, makeRelease, options, type Tweaks } from './helpers.ts';

/** A kNN reference in embedding-classifier's 'f32' format: row-major little-endian float32, base64. */
const f32Reference = (rows: number[][], labels: Record<string, Array<0 | 1 | null>>) => {
  const bytes = Buffer.alloc(rows.length * rows[0].length * 4);
  rows.flat().forEach((v, i) => bytes.writeFloatLE(v, i * 4));
  return { encoding: 'f32' as const, rows: rows.length, dims: rows[0].length, data: bytes.toString('base64'), labels };
};

async function rejects(p: Promise<unknown>, code: LoadErrorCode, pattern?: RegExp) {
  await assert.rejects(p, (e: unknown) => {
    assert.ok(e instanceof RouterLoadError, `expected RouterLoadError, got ${e}`);
    assert.equal(e.code, code, e.message);
    if (pattern) assert.match(e.message, pattern);
    return true;
  });
}

/** Replace one stored file's bytes, keeping everything else (including the manifest) unchanged. */
function withFile(built: BuiltRelease, key: string, bytes: Uint8Array): Map<string, Uint8Array> {
  return new Map([...built.files, [key, bytes]]);
}

test('a valid release loads in both modes and reports its id and digest', async () => {
  const built = await makeRelease();
  assert.deepEqual(built.enforcement, []);
  for (const mode of ['enforce', 'shadow'] as const) {
    const r = await loadRouter(options(built, { mode }));
    assert.equal(r.releaseId, 'release-1');
    assert.equal(r.manifestSha256, built.manifestSha256);
    assert.equal(r.mode, mode);
  }
});

test('options: mode is required, and timeouts, sample rates and the encoder identity are validated', async () => {
  const built = await makeRelease();
  const bad: Array<Partial<LoadRouterOptions>> = [
    { mode: undefined as never }, { timeoutMs: 0 }, { monitoring: { sampleRate: 1.5, samplingSalt: 's' } }, { monitoring: { sampleRate: -0.1, samplingSalt: 's' } },
    { encoder: functionEncoder({ ...IDENTITY, dimensions: 0 }, () => []) }, { expectedManifestSha256: 'ABC' },
  ];
  for (const o of bad) await rejects(loadRouter(options(built, o)), 'OPTIONS_INVALID');
});

test('changed bytes: the manifest digest, then each file digest', async () => {
  const built = await makeRelease();
  await rejects(loadRouter(options(built, { expectedManifestSha256: 'f'.repeat(64) })), 'MANIFEST_DIGEST_MISMATCH');
  for (const key of ['classifier.json', 'rules.json', 'policy.json', 'evidence.json']) {
    const bytes = built.files.get(key)!;
    const changed = new Uint8Array([...bytes.subarray(0, bytes.length - 1), 0x20]);
    await rejects(loadRouter(options(built, { source: memorySource(withFile(built, key, changed)) })), 'FILE_DIGEST_MISMATCH', new RegExp(key.replace('.', '\\.')));
  }
});

/** A manifest edited after building, re-pinned: every other check must still catch the problem. */
async function withManifest(edit: (m: Record<string, unknown>) => void): Promise<LoadRouterOptions> {
  const built = await makeRelease();
  const m = JSON.parse(JSON.stringify(built.manifest));
  edit(m);
  const bytes = encodeJson(m);
  return options(built, { source: memorySource(withFile(built, built.manifestKey, bytes)), expectedManifestSha256: await sha256Hex(bytes) });
}

test('malformed documents and unsupported schemas', async () => {
  const built = await makeRelease();
  const garbage = new TextEncoder().encode('{not json');
  await rejects(loadRouter(options(built, { source: memorySource(withFile(built, built.manifestKey, garbage)), expectedManifestSha256: await sha256Hex(garbage) })), 'DOCUMENT_MALFORMED');
  await rejects(loadRouter(await withManifest((m) => { m.schema = 'liquidau-router/3'; })), 'UNSUPPORTED_SCHEMA');
  await rejects(loadRouter(await withManifest((m) => { m.extra = 1; })), 'MANIFEST_INVALID', /unknown field extra/);
  await rejects(loadRouter(await withManifest((m) => { (m.gates as { passed: boolean }).passed = false; })), 'MANIFEST_INVALID', /record no failure/);
  await rejects(loadRouter(await withManifest((m) => { (m.serving as { ruleInput: string }).ruleInput = 'canonical'; })), 'MANIFEST_INVALID');
  await rejects(makeRelease({ policy: (p) => { (p as { schema: string }).schema = 'liquidau-router-policy/9'; } }), 'POLICY_INVALID', /unsupported policy schema/);
  await rejects(makeRelease({ evidence: (e) => { (e as { schema: string }).schema = 'x'; } }), 'UNSUPPORTED_SCHEMA');
});

test('keys may not escape the release prefix', async () => {
  await rejects(loadRouter(await withManifest((m) => { (m.files as Record<string, { key: string }>).rules.key = '../other/rules.json'; })), 'MANIFEST_INVALID', /safe relative key/);
  const built = await makeRelease();
  for (const key of ['/manifest.json', 'a/../manifest.json', 's3://bucket/manifest.json', '']) {
    await rejects(loadRouter(options(built, { manifestKey: key })), 'INVALID_KEY');
  }
});

test('package versions must be recorded and compatible', async () => {
  await rejects(loadRouter(await withManifest((m) => { (m.packages as Record<string, string>)['@liquidau/embedding-classifier'] = '0.6.2'; })), 'PACKAGE_INCOMPATIBLE', /0\.8\.x/);
  await rejects(loadRouter(await withManifest((m) => { delete (m.packages as Record<string, string>)['@liquidau/rule-miner']; })), 'PACKAGE_INCOMPATIBLE', /not recorded/);
});

test('pairing: rule set, training.router, dismissal records, evidence inputs and gates must all agree', async () => {
  const other = 'e'.repeat(64);
  const cases: Array<[string, Tweaks, RegExp]> = [
    ['training.router names another rule set', { artifact: (a) => { a.training!.router = { ruleSetHash: other }; } }, /trained with rule set/],
    ['no training.router (legacy artifact)', { artifact: (a) => { delete a.training!.router; } }, /no training\.router/],
    ['a head certified with another dismissal rule set', { artifact: (a) => { a.heads.other!.dismissal = { rule_set: other, max_rate: 0.05, certified: 1 }; } }, /dismissal pairing/],
    ['rules changed after training', { rules: (r) => { r.rules[0].id = 'refund.renamed'; } }, /trained with rule set/],
    ['evidence evaluated another policy', { evidence: (e) => { e.inputs.policySha256 = other; } }, /different policy/],
    ['evidence evaluated another classifier', { evidence: (e) => { e.inputs.classifierSha256 = other; } }, /different classifier/],
  ];
  for (const [name, tweaks, pattern] of cases) await rejects(makeRelease(tweaks), 'PAIRING_MISMATCH', pattern).catch((e) => { throw new Error(`${name}: ${e.message}`); });
  await rejects(loadRouter(await withManifest((m) => { m.rulesSemanticHash = other; })), 'PAIRING_MISMATCH', /semantic hash/);
  await rejects(loadRouter(await withManifest((m) => { (m.gates as { warnings: string[] }).warnings = ['added later']; })), 'PAIRING_MISMATCH', /gates/);
});

test('classifier and rules are validated by their owning libraries, including reference numbers', async () => {
  await rejects(makeRelease({ artifact: (a) => { a.heads.urgent!.weights = [1, 2]; } }), 'CLASSIFIER_INVALID', /weights/);
  await rejects(makeRelease({ artifact: (a) => {
    const rows = [[1, 0, 0, 0], [0, 1, 0, Number.NaN]];
    a.reference = f32Reference(rows, { urgent: [1, 0] });
    a.heads.urgent = { weights: [1], bias: 0, features: { kind: 'knn', k: 1 }, calibration: { method: 'platt', a: 1, c: 0 }, threshold: 0.5, review_floor: 0.2 };
  } }), 'CLASSIFIER_INVALID', /non-finite/);
  await rejects(makeRelease({ rules: (r) => { r.rules[0].regex = ['(a+)+$']; } }), 'RULES_INVALID', /regex/);
});

test('missing or extra heads and unmapped rule labels are refused at load', async () => {
  await rejects(makeRelease({ artifact: (a) => { delete a.heads.other; } }), 'POLICY_INVALID', /route other has no classifier head/);
  await rejects(makeRelease({ artifact: (a) => { (a.heads as Record<string, unknown>).spare = { ...a.heads.other! }; } }), 'POLICY_INVALID', /spare is not a route/);
  await rejects(makeRelease({ policy: (p) => { p.suppress.push({ when: ['other'], heads: ['urgent'] }); } }), 'POLICY_INVALID', /suppression cycle/);
  await rejects(makeRelease({ policy: (p) => { p.suppress.push({ when: ['refund'], heads: ['refund'] }); } }), 'POLICY_INVALID', /suppresses itself/);
  await rejects(makeRelease({ policy: (p) => { p.priority.pop(); } }), 'POLICY_INVALID', /missing from priority/);
  await rejects(makeRelease({ rules: (r) => { r.rules[1].label = 'billing'; } }), 'POLICY_INVALID', /rule label billing/);
});

test('encoder identity: every field must match, and the classifier must agree with the release identity', async () => {
  const built = await makeRelease();
  for (const change of [{ revision: 'sha256:1111' }, { precision: 'q8' }, { tokenizerRevision: 'x' }, { truncation: 'liquidau-truncate-utf16/1' }]) {
    await rejects(loadRouter(options(built, { encoder: fakeEncoder(undefined, { ...IDENTITY, ...change }).encoder })), 'ENCODER_IDENTITY_MISMATCH', new RegExp(Object.keys(change)[0]));
  }
  const layered = { ...IDENTITY, layers: [2, 1], normalization: 'per-layer-unit' as const };
  await rejects(buildRelease({ ...(await rebuildInput()), embedding: layered }), 'EMBEDDING_MISMATCH', /layers/);
  await rejects(makeRelease({ artifact: (a) => { delete a.embedding.precision; } }), 'EMBEDDING_MISMATCH', /precision/);
});

async function rebuildInput() {
  const built = await makeRelease();
  const f = built.files;
  return {
    releaseId: 'x', documents: { classifier: f.get('classifier.json')!, rules: f.get('rules.json')!, policy: f.get('policy.json')!, evidence: f.get('evidence.json')! },
    embedding: IDENTITY, serving: { maxInputUtf8Bytes: 100 }, packages: built.manifest.packages,
  };
}

test('gates: failures refuse enforcement but allow shadow; structural failures refuse both', async () => {
  for (const tweaks of [{ gatesPassed: false }, { evaluationPasses: false }, { evidence: (e: { final?: unknown }) => { delete e.final; } }] as Tweaks[]) {
    const built = await makeRelease(tweaks);
    assert.ok(built.enforcement.length);
    await rejects(loadRouter(options(built, { mode: 'enforce' })), 'GATES_FAILED');
    const shadow = await loadRouter(options(built, { mode: 'shadow' }));
    const r = await shadow.route({ text: 'refund please', requestId: '1' });
    assert.equal(r.actionable, false);
  }
  const built = await makeRelease({ gatesPassed: false });
  await rejects(loadRouter(options(built, { mode: 'shadow', source: memorySource(withFile(built, 'rules.json', new Uint8Array([1]))) })), 'FILE_DIGEST_MISMATCH');
  await rejects(loadRouter(options(built, { mode: 'shadow', encoder: fakeEncoder(undefined, { ...IDENTITY, revision: 'other' }).encoder })), 'ENCODER_IDENTITY_MISMATCH');
});

test('reads: size limits, deadlines, cancellation and failing sources', async () => {
  const built = await makeRelease();
  await rejects(loadRouter(options(built, { limits: { maxFileBytes: 64 } })), 'FILE_TOO_LARGE');
  await rejects(loadRouter(options(built, { limits: { maxManifestBytes: 64 } })), 'FILE_TOO_LARGE');
  await rejects(loadRouter(options(built, { timeoutMs: 20, source: { read: () => new Promise(() => {}) } })), 'READ_TIMEOUT');
  await rejects(loadRouter(options(built, { source: { read: async () => { throw new Error('denied'); } } })), 'READ_FAILED', /denied/);
  await rejects(loadRouter(options(built, { source: { read: async () => 'text' as never } })), 'READ_FAILED');
  const controller = new AbortController();
  controller.abort();
  await rejects(loadRouter(options(built, { signal: controller.signal })), 'ABORTED');
});

test('the loaded snapshot is private: changing source bytes after load changes nothing', async () => {
  const built = await makeRelease();
  const files = new Map([...built.files].map(([k, v]) => [k, v.slice()]));
  let returned: Uint8Array[] = [];
  const router = await loadRouter(options(built, { source: { read: async (k) => { const b = files.get(k)!; returned.push(b); return b; } } }));
  for (const b of returned) b.fill(0);
  returned = [];
  const r = await router.route({ text: 'refund please', requestId: '1' });
  assert.equal(r.outcome, 'route');
  assert.equal((await load()).releaseId, router.releaseId);
});
