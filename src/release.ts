/**
 * Reading and validating a complete release (load steps 1-7), shared by loadRouter and
 * buildRelease so the builder writes exactly what the loader accepts. Mode-dependent gate checks
 * (step 8) are separate: `enforcementProblems`.
 */
import { checkRuleSetPairing, checkTierPairing, routerRuleSetHash, validateArtifact, validateDocumentArtifact, validateEdge, validateMiddle, type ClassifierArtifact, type DocumentClassifierArtifact, type EdgeDocument, type MiddleDocument } from '@liquidau/embedding-classifier';
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
  /** The edge tier's document, when the release has one. */
  edge: EdgeDocument | null;
  /** The middle tier's document, when the release has one (the model tier runs it). */
  middle: MiddleDocument | null;
}

type Role = 'classifier' | 'rules' | 'policy' | 'evidence' | 'edge' | 'middle';

/** The middle document of a release (null without one), validated, with its heads checked against the policy's routes. */
function parseMiddle(bytes: Uint8Array | undefined, routes: readonly string[]): MiddleDocument | null {
  if (!bytes) return null;
  let middle: MiddleDocument;
  try {
    middle = validateMiddle(parse(bytes, 'middle document'));
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError('MIDDLE_INVALID', 'the middle document is invalid', [(e as Error).message]);
  }
  const unknown = Object.keys(middle.heads).filter((h) => !routes.includes(h));
  if (unknown.length) throw new RouterLoadError('MIDDLE_INVALID', 'the middle tier decides for heads that are not routes', unknown);
  return middle;
}

/** The edge document of a release (null without one), validated, with its heads checked against the policy's routes. */
function parseEdge(bytes: Uint8Array | undefined, routes: readonly string[]): EdgeDocument | null {
  if (!bytes) return null;
  let edge: EdgeDocument;
  try {
    edge = validateEdge(parse(bytes, 'edge document'));
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError('EDGE_INVALID', 'the edge document is invalid', [(e as Error).message]);
  }
  const unknown = Object.keys(edge.heads).filter((h) => !routes.includes(h));
  if (unknown.length) throw new RouterLoadError('EDGE_INVALID', 'the edge decides for heads that are not routes', unknown);
  return edge;
}

/** Edge pairing between the manifest, the evidence and the file (empty when matched). */
function edgeEvidencePairing(manifest: RouterManifest, evidence: ReleaseEvidence): string[] {
  const edgeSha = manifest.files.edge?.sha256;
  if (edgeSha && evidence.inputs.edgeSha256 !== edgeSha) return ['the evidence evaluated a different edge document (or none)'];
  if (!edgeSha && evidence.inputs.edgeSha256 !== undefined) return ['the evidence evaluated an edge the release does not have'];
  const middleSha = manifest.files.middle?.sha256;
  if (middleSha && evidence.inputs.middleSha256 !== middleSha) return ['the evidence evaluated a different middle document (or none)'];
  if (!middleSha && evidence.inputs.middleSha256 !== undefined) return ['the evidence evaluated a middle tier the release does not have'];
  return [];
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
export async function validateDocuments(manifestRaw: unknown, manifestSha256: string, files: Readonly<Record<'classifier' | 'rules' | 'policy' | 'evidence', Uint8Array> & { edge?: Uint8Array; middle?: Uint8Array }>): Promise<ReleaseDocuments> {
  const manifest = validateManifest(manifestRaw);
  const mismatched: string[] = [];
  if (!!manifest.files.edge !== !!files.edge) mismatched.push(manifest.files.edge ? 'the manifest names an edge file that was not supplied' : 'an edge file was supplied but the manifest names none');
  if (!!manifest.files.middle !== !!files.middle) mismatched.push(manifest.files.middle ? 'the manifest names a middle file that was not supplied' : 'a middle file was supplied but the manifest names none');
  for (const role of ['classifier', 'rules', 'policy', 'evidence', ...(manifest.files.edge && files.edge ? ['edge' as const] : []), ...(manifest.files.middle && files.middle ? ['middle' as const] : [])] as Role[]) {
    const actual = await sha256Hex(files[role]!);
    if (actual !== manifest.files[role]!.sha256) mismatched.push(`${role} (${manifest.files[role]!.key}) has SHA-256 ${actual}, the manifest says ${manifest.files[role]!.sha256}`);
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
  const edge = parseEdge(files.edge, policy.priority);
  const middle = parseMiddle(files.middle, policy.priority);

  const semantic = ruleSetHash(ruleSet);
  const pairing: string[] = [];
  if (semantic !== manifest.rulesSemanticHash) pairing.push(`the rule set's semantic hash is ${semantic}, the manifest says ${manifest.rulesSemanticHash}`);
  pairing.push(...checkTierPairing(artifact, edge, middle).map((d) => `tier pairing: ${d}`));
  pairing.push(...edgeEvidencePairing(manifest, evidence));
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
  // Context features: the classifier's context spec and the manifest's context encoder, both or neither.
  const ctx = document ? undefined : artifact.embedding.context, ctxId = manifest.contextEmbedding;
  if (ctx || ctxId) {
    const c = !ctx ? ['the release names a context encoder (contextEmbedding), but the classifier has no context features']
      : !ctxId ? ['the classifier was trained with context features (embedding.context), but the release names no context encoder (contextEmbedding)']
      : classifierEmbeddingDiff(ctxId, ctx).map((x) => `context ${x}`);
    if (c.length) throw new RouterLoadError('EMBEDDING_MISMATCH', "the classifier's recorded context embedding does not match the release's context encoder", c);
  }
  // The middle tier: its document's encoder identity and the manifest's middle encoder (validateManifest requires both or neither).
  if (middle) {
    const m = classifierEmbeddingDiff(manifest.middleEmbedding!, middle.embedding).map((x) => `middle ${x}`);
    if (m.length) throw new RouterLoadError('EMBEDDING_MISMATCH', "the middle document's embedding does not match the release's middle encoder (middleEmbedding)", m);
  }

  return { manifest, manifestSha256, artifact, document, ruleSet, policy, evidence, edge, middle };
}

/** Steps 1-7: read and verify the manifest, then the referenced files (each read once), then validate. */
export async function readRelease(source: ReleaseSource, manifestKey: string, expectedManifestSha256: string, timeoutMs: number, limits: Required<ReadLimits>, signal?: AbortSignal): Promise<ReleaseDocuments> {
  if (!isSha256Hex(expectedManifestSha256)) throw new RouterLoadError('OPTIONS_INVALID', 'expectedManifestSha256 must be lowercase hex SHA-256');
  const manifestBytes = await readBytes(source, manifestKey, limits.maxManifestBytes, timeoutMs, signal);
  const actual = await sha256Hex(manifestBytes);
  if (actual !== expectedManifestSha256) throw new RouterLoadError('MANIFEST_DIGEST_MISMATCH', `the manifest's SHA-256 is ${actual}, deployment expects ${expectedManifestSha256}`);
  const manifest = validateManifest(parse(manifestBytes, 'manifest'));
  const roles = ['classifier', 'rules', 'policy', 'evidence', ...(manifest.files.edge ? ['edge' as const] : []), ...(manifest.files.middle ? ['middle' as const] : [])] as Role[];
  const bytes = await Promise.all(roles.map((role) => readBytes(source, manifest.files[role]!.key, limits.maxFileBytes, timeoutMs, signal)));
  const byRole = Object.fromEntries(roles.map((role, i) => [role, bytes[i]])) as Record<Role, Uint8Array | undefined>;
  return validateDocuments(manifest, actual, {
    classifier: byRole.classifier!, rules: byRole.rules!, policy: byRole.policy!, evidence: byRole.evidence!,
    ...(byRole.edge ? { edge: byRole.edge } : {}), ...(byRole.middle ? { middle: byRole.middle } : {}),
  });
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

/** What a rules tier loads: everything but the classifier, each file verified against the pinned manifest. */
export interface RulesReleaseDocuments {
  manifest: RouterManifest;
  manifestSha256: string;
  ruleSet: RuleSet;
  policy: RouterPolicy;
  evidence: ReleaseEvidence;
  /** The edge tier's document (the rules tier runs it): its pairing with the classifier is checked by the model tier. */
  edge: EdgeDocument | null;
}

/**
 * Steps 1-6 for a rules tier: the manifest's exact bytes, then the rules, policy and evidence (never the
 * classifier, so loading stays small and fast), each checked against the manifest's digests; the rule
 * set's semantic hash; the policy against the rule labels; and the evidence's inputs against the
 * manifest. The model tier loads the same manifest digest and validates the classifier and its pairing,
 * so a deployment that pins one digest for both tiers serves one matched release.
 */
export async function readRulesRelease(source: ReleaseSource, manifestKey: string, expectedManifestSha256: string, timeoutMs: number, limits: Required<ReadLimits>, signal?: AbortSignal): Promise<RulesReleaseDocuments> {
  if (!isSha256Hex(expectedManifestSha256)) throw new RouterLoadError('OPTIONS_INVALID', 'expectedManifestSha256 must be lowercase hex SHA-256');
  const manifestBytes = await readBytes(source, manifestKey, limits.maxManifestBytes, timeoutMs, signal);
  const manifestSha256 = await sha256Hex(manifestBytes);
  if (manifestSha256 !== expectedManifestSha256) throw new RouterLoadError('MANIFEST_DIGEST_MISMATCH', `the manifest's SHA-256 is ${manifestSha256}, deployment expects ${expectedManifestSha256}`);
  const manifest = validateManifest(parse(manifestBytes, 'manifest'));
  const roles = ['rules', 'policy', 'evidence', ...(manifest.files.edge ? ['edge' as const] : [])] as Role[];
  const bytes = await Promise.all(roles.map((role) => readBytes(source, manifest.files[role]!.key, limits.maxFileBytes, timeoutMs, signal)));
  const mismatched: string[] = [];
  for (const [i, role] of roles.entries()) {
    const actual = await sha256Hex(bytes[i]);
    if (actual !== manifest.files[role]!.sha256) mismatched.push(`${role} (${manifest.files[role]!.key}) has SHA-256 ${actual}, the manifest says ${manifest.files[role]!.sha256}`);
  }
  if (mismatched.length) throw new RouterLoadError('FILE_DIGEST_MISMATCH', 'release files do not match the manifest', mismatched);
  let ruleSet: RuleSet;
  try {
    ruleSet = validateRuleSet(parse(bytes[0], 'rule set'));
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError('RULES_INVALID', 'the rule set is invalid', [(e as Error).message]);
  }
  const policy = validatePolicy(parse(bytes[1], 'policy'), { ruleLabels: ruleSet.rules.map((r) => r.label) });
  const evidence = validateEvidence(parse(bytes[2], 'evidence'));
  const edge = parseEdge(bytes[3], policy.priority);
  const documentRelease = manifest.schema === MANIFEST_SCHEMA_DOCUMENT;
  const expectedEvidence = documentRelease ? EVIDENCE_SCHEMA_DOCUMENT : EVIDENCE_SCHEMA;
  if (evidence.schema !== expectedEvidence) throw new RouterLoadError('UNSUPPORTED_SCHEMA', `a ${manifest.schema} release needs '${expectedEvidence}' evidence, not '${evidence.schema}'`);
  const semantic = ruleSetHash(ruleSet);
  const pairing: string[] = [];
  if (semantic !== manifest.rulesSemanticHash) pairing.push(`the rule set's semantic hash is ${semantic}, the manifest says ${manifest.rulesSemanticHash}`);
  const ins = evidence.inputs;
  if (ins.classifierSha256 !== manifest.files.classifier.sha256) pairing.push('the evidence evaluated a different classifier artifact');
  if (ins.rulesSha256 !== manifest.files.rules.sha256) pairing.push('the evidence evaluated a different rule set file');
  if (ins.policySha256 !== manifest.files.policy.sha256) pairing.push('the evidence evaluated a different policy');
  if (ins.rulesSemanticHash !== semantic) pairing.push('the evidence evaluated a different rule set');
  if (documentRelease) {
    if (ins.featureIdentitySha256 !== manifest.featureIdentitySha256) pairing.push('the evidence evaluated another document feature identity (encoder or pipeline)');
    if (ins.servingMaxInputUtf8Bytes !== manifest.serving.maxInputUtf8Bytes) pairing.push(`the evidence was evaluated with a ${ins.servingMaxInputUtf8Bytes}-byte input limit, the manifest serves ${manifest.serving.maxInputUtf8Bytes}`);
  }
  pairing.push(...edgeEvidencePairing(manifest, evidence));
  if (pairing.length) throw new RouterLoadError('PAIRING_MISMATCH', 'the release documents are not a matched set', pairing);
  return { manifest, manifestSha256, ruleSet, policy, evidence, edge };
}

/**
 * Step 8 for a rules tier. The manifest's gates are the classifier's gates combined with the evidence's
 * (the builder writes them so and the model tier checks it), so they stand in for the classifier here.
 */
export function rulesEnforcementProblems(docs: RulesReleaseDocuments): string[] {
  const out: string[] = [];
  if (!docs.manifest.gates.passed || docs.manifest.gates.failures.length) out.push(`complete-release gates did not pass: ${docs.manifest.gates.failures.join('; ')}`);
  out.push(...evidenceInsufficiency(docs.evidence));
  return out;
}
