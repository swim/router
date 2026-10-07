/**
 * The offline release builder: given the exact document bytes (classifier artifact, rule set, policy,
 * evaluation evidence) it writes the manifest and validates the whole release with the same code the
 * loader runs. Acceptance comes from the evidence's final-system gates combined with the classifier's
 * gates, never from the training result's failure list alone. File I/O and upload stay in the
 * application pipeline: this returns bytes.
 *
 * Order: buildArtifact(result, { router: { ruleSetHash } }) -> serialise -> evaluate with the
 * `evaluation` subpath (which records these digests) -> buildRelease.
 */
import { isDocumentArtifact, validateArtifact, validateDocumentArtifact } from '@liquidau/embedding-classifier';
import { ruleSetHash, validateRuleSet } from '@liquidau/rule-miner';

import { decodeJson, encodeJson, sha256Hex } from './bytes.ts';
import { RouterLoadError, type LoadErrorCode } from './errors.ts';
import { validateEvidence } from './evidence.ts';
import type { EncoderIdentity } from './identity.ts';
import { MANIFEST_SCHEMA, MANIFEST_SCHEMA_DOCUMENT, requiredPackages, type RouterManifest } from './manifest.ts';
import { combineGates, enforcementProblems, validateDocuments } from './release.ts';

export interface BuildReleaseInput {
  releaseId: string;
  /** ISO 8601 (default now). */
  createdAt?: string;
  documents: { classifier: Uint8Array; rules: Uint8Array; policy: Uint8Array; evidence: Uint8Array };
  /** File keys relative to the release prefix (defaults: classifier.json, rules.json, policy.json, evidence.json). */
  keys?: Partial<Record<'classifier' | 'rules' | 'policy' | 'evidence', string>>;
  embedding: EncoderIdentity;
  serving: { maxInputUtf8Bytes: number };
  /**
   * The exact installed versions the release was built with, recorded for provenance: at least the
   * packages requiredPackages lists (embedding-classifier, rule-miner, router; text-preprocessing for
   * document releases). The router doesn't check them; serve with the versions recorded here.
   */
  packages: Record<string, string>;
}

export interface BuiltRelease {
  manifest: RouterManifest;
  manifestBytes: Uint8Array;
  /** Pin this in deployment configuration (expectedManifestSha256). */
  manifestSha256: string;
  manifestKey: string;
  /** Every file to upload under the release prefix, manifest included. */
  files: ReadonlyMap<string, Uint8Array>;
  /** Why the release may be served only in shadow mode (empty: enforceable). */
  enforcement: string[];
}

/** Lowercase hex SHA-256 of a release document's bytes (what evidence records as its inputs). */
export const documentDigest = sha256Hex;

/** Builds and validates a release; throws RouterLoadError on anything the loader would refuse in shadow mode. */
export async function buildRelease(input: BuildReleaseInput): Promise<BuiltRelease> {
  const keys = { classifier: 'classifier.json', rules: 'rules.json', policy: 'policy.json', evidence: 'evidence.json', ...input.keys };
  const manifestKey = 'manifest.json';
  const taken = Object.entries(keys).filter(([, k]) => k === manifestKey).map(([role]) => `${role} uses the reserved key ${manifestKey}`);
  if (taken.length) throw new RouterLoadError('MANIFEST_INVALID', 'release file keys collide with the manifest', taken);
  // Owned copies: the result's bytes are exactly what was hashed, whatever the caller does next.
  const src = input.documents;
  const d = { classifier: src.classifier.slice(), rules: src.rules.slice(), policy: src.policy.slice(), evidence: src.evidence.slice() };
  const sha = {
    classifier: await sha256Hex(d.classifier), rules: await sha256Hex(d.rules), policy: await sha256Hex(d.policy), evidence: await sha256Hex(d.evidence),
  };
  // The semantic hash and gates are derived from the documents; validateDocuments re-checks everything.
  const ruleSet = derive('RULES_INVALID', 'rule set', () => validateRuleSet(decodeJson(d.rules)));
  // A document envelope makes a 'liquidau-router/2' release; anything else is a legacy '/1' release.
  const classifierRaw = derive('CLASSIFIER_INVALID', 'classifier artifact', () => decodeJson(d.classifier));
  const document = isDocumentArtifact(classifierRaw) ? derive('CLASSIFIER_INVALID', 'document classifier artifact', () => validateDocumentArtifact(classifierRaw)) : null;
  const artifact = document ? document.classifier : derive('CLASSIFIER_INVALID', 'classifier artifact', () => validateArtifact(classifierRaw));
  const evidence = derive('EVIDENCE_INVALID', 'evidence', () => validateEvidence(decodeJson(d.evidence)));
  const manifest: RouterManifest = {
    schema: document ? MANIFEST_SCHEMA_DOCUMENT : MANIFEST_SCHEMA,
    releaseId: input.releaseId,
    createdAt: input.createdAt ?? new Date().toISOString(),
    files: {
      classifier: { key: keys.classifier, sha256: sha.classifier },
      rules: { key: keys.rules, sha256: sha.rules },
      policy: { key: keys.policy, sha256: sha.policy },
      evidence: { key: keys.evidence, sha256: sha.evidence },
    },
    ...(document ? { featureIdentitySha256: document.featureIdentitySha256 } : {}),
    rulesSemanticHash: ruleSetHash(ruleSet),
    embedding: input.embedding,
    serving: { ruleInput: 'original', maxInputUtf8Bytes: input.serving.maxInputUtf8Bytes },
    gates: combineGates(artifact.gates, evidence.gates),
    packages: { ...input.packages },
  };
  const manifestBytes = encodeJson(manifest);
  const manifestSha256 = await sha256Hex(manifestBytes);
  const docs = await validateDocuments(JSON.parse(JSON.stringify(manifest)), manifestSha256, d);
  const files = new Map<string, Uint8Array>([[manifestKey, manifestBytes], [keys.classifier, d.classifier], [keys.rules, d.rules], [keys.policy, d.policy], [keys.evidence, d.evidence]]);
  return { manifest: docs.manifest, manifestBytes, manifestSha256, manifestKey, files, enforcement: enforcementProblems(docs) };
}

function derive<T>(code: LoadErrorCode, what: string, fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (e instanceof RouterLoadError) throw e;
    throw new RouterLoadError(code, `the ${what} is invalid`, [(e as Error).message]);
  }
}

/** The package names a legacy release must record (for building `packages`); see requiredPackages for document releases. */
export const REQUIRED_PACKAGES: readonly string[] = requiredPackages(MANIFEST_SCHEMA);
export { requiredPackages };
