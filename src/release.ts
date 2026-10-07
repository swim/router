/**
 * Reading and validating a complete release (load steps 1-7), shared by loadRouter and
 * buildRelease so the builder writes exactly what the loader accepts. Mode-dependent gate checks
 * (step 8) are separate: `enforcementProblems`.
 */
import { checkRuleSetPairing, routerRuleSetHash, validateArtifact, validateDocumentArtifact, type ClassifierArtifact, type DocumentClassifierArtifact } from '@liquidau/embedding-classifier';
import { ruleSetHash, validateRuleSet, type RuleSet } from '@liquidau/rule-miner';

import { canonicalJson, decodeJson, isSha256Hex, sha256Hex } from './bytes.ts';
import { RouterLoadError } from './errors.ts';
import { EVIDENCE_SCHEMA, EVIDENCE_SCHEMA_DOCUMENT, evidenceInsufficiency, validateEvidence, type ReleaseEvidence } from './evidence.ts';
import { classifierEmbeddingDiff, encoderIdentityDiff, type EncoderIdentity } from './identity.ts';
import { isSafeKey, MANIFEST_SCHEMA_DOCUMENT, validateManifest, type ReleaseGates, type RouterManifest } from './manifest.ts';
import { validatePolicy, type RouterPolicy } from './policy.ts';

export interface ReleaseSource {
  /** The exact stored bytes for a key relative to the source. Apply size limits before buffering. */
  read(key: string, options: { signal: AbortSignal }): Promise<Uint8Array>;
}

export interface ReadLimits {
  /** Default 1 MiB. */
  maxManifestBytes?: number;
  /** Per release file. Default 512 MiB. */
  maxFileBytes?: number;
}

export const DEFAULT_LIMITS = Object.freeze({ maxManifestBytes: 1 << 20, maxFileBytes: 512 * (1 << 20) });

export interface ReleaseDocuments {
  manifest: RouterManifest;
  manifestSha256: string;
  /** The classifier that scores: the legacy artifact, or the document envelope's embedded classifier. */
  artifact: ClassifierArtifact;
  /** Document releases ('liquidau-router/2'): the validated envelope. */
  document: DocumentClassifierArtifact | null;
  ruleSet: RuleSet;
  policy: RouterPolicy;
  evidence: ReleaseEvidence;
}

/** Reads one key within a deadline and size limit; the returned bytes are a private copy. */
export async function readBytes(source: ReleaseSource, key: string, maxBytes: number, timeoutMs: number, outer?: AbortSignal): Promise<Uint8Array> {
  if (!isSafeKey(key)) throw new RouterLoadError('INVALID_KEY', `release key ${JSON.stringify(key)} is not a safe relative key`);
  if (outer?.aborted) throw new RouterLoadError('ABORTED', `reading ${key} was aborted`);
  const controller = new AbortController();
  let reason: 'timeout' | 'aborted' | null = null;
  let rejectAbort!: (e: unknown) => void;
  const aborted = new Promise<never>((_, reject) => { rejectAbort = reject; });
  const stop = (why: 'timeout' | 'aborted') => {
    if (reason) return;
    reason = why;
    controller.abort();
    rejectAbort(why === 'timeout' ? new RouterLoadError('READ_TIMEOUT', `reading ${key} took longer than ${timeoutMs} ms`) : new RouterLoadError('ABORTED', `reading ${key} was aborted`));
  };
  const timer = setTimeout(() => stop('timeout'), timeoutMs);
  const onAbort = () => stop('aborted');
  outer?.addEventListener('abort', onAbort, { once: true });
  try {
    const reading = Promise.resolve().then(() => source.read(key, { signal: controller.signal }));
    reading.catch(() => {});
    let bytes: unknown;
    try {
      bytes = await Promise.race([reading, aborted]);
    } catch (e) {
      if (e instanceof RouterLoadError) throw e;
      throw new RouterLoadError('READ_FAILED', `reading ${key} failed`, [e instanceof Error ? e.message : String(e)]);
    }
    if (!(bytes instanceof Uint8Array)) throw new RouterLoadError('READ_FAILED', `the source returned ${typeof bytes} for ${key}, not a Uint8Array`);
    if (bytes.byteLength > maxBytes) throw new RouterLoadError('FILE_TOO_LARGE', `${key} is ${bytes.byteLength} bytes, over the ${maxBytes}-byte limit`);
    return bytes.slice();
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener('abort', onAbort);
  }
}

function parse(bytes: Uint8Array, what: string): unknown {
  try {
    return decodeJson(bytes);
  } catch (e) {
    throw new RouterLoadError('DOCUMENT_MALFORMED', `the ${what} is not UTF-8 JSON`, [(e as Error).message]);
  }
}

/** Complete-release gates: the classifier's gates combined with the evaluation evidence's. Deterministic. */
export function combineGates(classifier: ClassifierArtifact['gates'], evaluation: ReleaseGates): ReleaseGates {
  const failures = [
    ...(classifier === undefined ? ['classifier: gates were not recorded'] : classifier.failures.map((f) => `classifier: ${f}`)),
    ...evaluation.failures.map((f) => `evaluation: ${f}`),
  ];
  const warnings = [...(classifier?.warnings ?? []).map((w) => `classifier: ${w}`), ...evaluation.warnings.map((w) => `evaluation: ${w}`)];
  return { passed: classifier?.passed === true && evaluation.passed && failures.length === 0, failures, warnings };
}

/**
 * Steps 2-7 on documents already read: validate the manifest (unless given), verify file digests,
 * validate every document with its owning library, the policy against the classifier heads and rule
 * labels, and every pairing (semantic rules hash, training.router, dismissal records, evidence inputs,
 * classifier embedding vs release identity, recorded gates).
 */
export async function validateDocuments(manifestRaw: unknown, manifestSha256: string, files: Readonly<Record<'classifier' | 'rules' | 'policy' | 'evidence', Uint8Array>>): Promise<ReleaseDocuments> {
  const manifest = validateManifest(manifestRaw);
  const mismatched: string[] = [];
  for (const role of ['classifier', 'rules', 'policy', 'evidence'] as const) {
    const actual = await sha256Hex(files[role]);
    if (actual !== manifest.files[role].sha256) mismatched.push(`${role} (${manifest.files[role].key}) has SHA-256 ${actual}, the manifest says ${manifest.files[role].sha256}`);
  }
  if (mismatched.length) throw new RouterLoadError('FILE_DIGEST_MISMATCH', 'release files do not match the manifest', mismatched);

  const documentRelease = manifest.schema === MANIFEST_SCHEMA_DOCUMENT;
  let artifact: ClassifierArtifact;
  let document: DocumentClassifierArtifact | null = null;
  try {
    const raw = parse(files.classifier, 'classifier artifact');
    if (documentRelease) {
      document = validateDocumentArtifact(raw);
      artifact = document.classifier;
    } else artifact = validateArtifact(raw);
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError('CLASSIFIER_INVALID', 'the classifier artifact is invalid', [(e as Error).message]);
  }
  let ruleSet: RuleSet;
  try {
    ruleSet = validateRuleSet(parse(files.rules, 'rule set'));
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError('RULES_INVALID', 'the rule set is invalid', [(e as Error).message]);
  }
  const heads = Object.keys(artifact.heads);
  const policy = validatePolicy(parse(files.policy, 'policy'), { heads, ruleLabels: ruleSet.rules.map((r) => r.label) });
  const evidence = validateEvidence(parse(files.evidence, 'evidence'));
  const expectedEvidence = documentRelease ? EVIDENCE_SCHEMA_DOCUMENT : EVIDENCE_SCHEMA;
  if (evidence.schema !== expectedEvidence) throw new RouterLoadError('UNSUPPORTED_SCHEMA', `a ${manifest.schema} release needs '${expectedEvidence}' evidence, not '${evidence.schema}'`);

  const semantic = ruleSetHash(ruleSet);
  const pairing: string[] = [];
  if (semantic !== manifest.rulesSemanticHash) pairing.push(`the rule set's semantic hash is ${semantic}, the manifest says ${manifest.rulesSemanticHash}`);
  let trained: string | undefined;
  try {
    trained = routerRuleSetHash(artifact);
  } catch (e) {
    pairing.push((e as Error).message);
  }
  if (trained === undefined && !pairing.length) pairing.push('the classifier artifact has no training.router.ruleSetHash (convert legacy artifacts and re-evaluate; the router does not guess pairing from other metadata)');
  else if (trained !== undefined && trained !== semantic) pairing.push(`the classifier was trained with rule set ${trained}, the release runs ${semantic}`);
  pairing.push(...checkRuleSetPairing(artifact, semantic).map((d) => `dismissal pairing: ${d}`));
  const ins = evidence.inputs;
  if (ins.classifierSha256 !== manifest.files.classifier.sha256) pairing.push('the evidence evaluated a different classifier artifact');
  if (ins.rulesSha256 !== manifest.files.rules.sha256) pairing.push('the evidence evaluated a different rule set file');
  if (ins.policySha256 !== manifest.files.policy.sha256) pairing.push('the evidence evaluated a different policy');
  if (ins.rulesSemanticHash !== semantic) pairing.push('the evidence evaluated a different rule set');
  if (document) {
    if (manifest.featureIdentitySha256 !== document.featureIdentitySha256) pairing.push("the manifest's featureIdentitySha256 is not the classifier envelope's");
    if (ins.featureIdentitySha256 !== document.featureIdentitySha256) pairing.push('the evidence evaluated another document feature identity (encoder or pipeline)');
    if (ins.servingMaxInputUtf8Bytes !== manifest.serving.maxInputUtf8Bytes) pairing.push(`the evidence was evaluated with a ${ins.servingMaxInputUtf8Bytes}-byte input limit, the manifest serves ${manifest.serving.maxInputUtf8Bytes}`);
  }
  if (canonicalJson(combineGates(artifact.gates, evidence.gates)) !== canonicalJson(manifest.gates)) pairing.push("the manifest's gates are not the classifier's gates combined with the evidence's");
  if (pairing.length) throw new RouterLoadError('PAIRING_MISMATCH', 'the release documents are not a matched set', pairing);

  // Legacy: the classifier's embedding spec maps onto the release identity. Document: the envelope pins
  // the full chunk-encoder identity itself (its classifier describes pooled features, not the encoder).
  const embedding = document ? encoderIdentityDiff(manifest.embedding, document.encoderIdentity as EncoderIdentity) : classifierEmbeddingDiff(manifest.embedding, artifact.embedding);
  if (embedding.length) throw new RouterLoadError('EMBEDDING_MISMATCH', document ? "the envelope's encoder identity does not match the release encoder identity" : "the classifier's recorded embedding does not match the release encoder identity", embedding);

  return { manifest, manifestSha256, artifact, document, ruleSet, policy, evidence };
}

/** Steps 1-7: read and verify the manifest, then the referenced files (each read once), then validate. */
export async function readRelease(source: ReleaseSource, manifestKey: string, expectedManifestSha256: string, timeoutMs: number, limits: Required<ReadLimits>, signal?: AbortSignal): Promise<ReleaseDocuments> {
  if (!isSha256Hex(expectedManifestSha256)) throw new RouterLoadError('OPTIONS_INVALID', 'expectedManifestSha256 must be lowercase hex SHA-256');
  const manifestBytes = await readBytes(source, manifestKey, limits.maxManifestBytes, timeoutMs, signal);
  const actual = await sha256Hex(manifestBytes);
  if (actual !== expectedManifestSha256) throw new RouterLoadError('MANIFEST_DIGEST_MISMATCH', `the manifest's SHA-256 is ${actual}, deployment expects ${expectedManifestSha256}`);
  const manifest = validateManifest(parse(manifestBytes, 'manifest'));
  const roles = ['classifier', 'rules', 'policy', 'evidence'] as const;
  const bytes = await Promise.all(roles.map((role) => readBytes(source, manifest.files[role].key, limits.maxFileBytes, timeoutMs, signal)));
  return validateDocuments(manifest, actual, { classifier: bytes[0], rules: bytes[1], policy: bytes[2], evidence: bytes[3] });
}

/** Step 8 for enforcement: why this release must not act (empty when it may). Shadow mode needs none of these. */
export function enforcementProblems(docs: ReleaseDocuments): string[] {
  const out: string[] = [];
  const g = docs.artifact.gates;
  if (g?.passed !== true || g.failures.length) out.push(`classifier gates did not pass${g ? `: ${g.failures.join('; ')}` : ' (not recorded)'}`);
  if (!docs.manifest.gates.passed || docs.manifest.gates.failures.length) out.push(`complete-release gates did not pass: ${docs.manifest.gates.failures.join('; ')}`);
  out.push(...evidenceInsufficiency(docs.evidence));
  return out;
}
