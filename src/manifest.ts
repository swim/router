/**
 * The release manifest: one immutable document naming the four release files by exact-byte SHA-256,
 * the rule set's semantic hash, the encoder identity, serving limits, complete-release gates and the
 * build-time package versions. Its own digest is the deployment identity, supplied by trusted
 * deployment configuration; content hashes detect corruption and mixing, they don't authenticate.
 */
import { isSha256Hex } from './bytes.ts';
import { RouterLoadError } from './errors.ts';
import { encoderIdentityProblems, type EncoderIdentity } from './identity.ts';

export const MANIFEST_SCHEMA = 'liquidau-router/1';
/** Document-classifier releases: the classifier file is a 'liquidau-document-classifier/1' envelope. */
export const MANIFEST_SCHEMA_DOCUMENT = 'liquidau-router/2';

export interface FileRef {
  /** Relative to the release source; never escapes it. */
  key: string;
  /** Lowercase hex SHA-256 of the exact stored bytes. */
  sha256: string;
}

export interface ReleaseGates {
  passed: boolean;
  failures: string[];
  warnings: string[];
}

export interface RouterManifest {
  schema: typeof MANIFEST_SCHEMA | typeof MANIFEST_SCHEMA_DOCUMENT;
  releaseId: string;
  /** ISO 8601. */
  createdAt: string;
  /**
   * `edge`: the edge tier's document (embedding-classifier's 'liquidau-edge/1'), when the release has one.
   * `middle`: the middle tier's document ('liquidau-middle/1'; legacy releases with an edge only).
   */
  files: { classifier: FileRef; rules: FileRef; policy: FileRef; evidence: FileRef; edge?: FileRef; middle?: FileRef };
  /** '/2' only: the envelope's document feature identity (chunk encoder AND pipeline). */
  featureIdentitySha256?: string;
  /** rule-miner's ruleSetHash of the validated rule set: rule semantics, not bytes. */
  rulesSemanticHash: string;
  embedding: EncoderIdentity;
  /**
   * Legacy releases with context features: the context encoder's identity (the classifier's
   * embedding.context). Requests then carry `context`, embedded by an adapter implementing exactly this.
   */
  contextEmbedding?: EncoderIdentity;
  /** Releases with a middle tier: its encoder's identity (the middle document's embedding). */
  middleEmbedding?: EncoderIdentity;
  serving: { ruleInput: 'original'; maxInputUtf8Bytes: number };
  /** Complete-release gates: the classifier's gates combined with the final-system evaluation's. */
  gates: ReleaseGates;
  /** Exact build-time versions. */
  packages: Record<string, string>;
}

/**
 * @deprecated The router no longer checks recorded package versions against version lines: a release
 * records the versions it was built with, and serving it with compatible library versions is the
 * deployer's responsibility. Always empty.
 */
export const COMPATIBLE_PACKAGES: Readonly<Record<string, string>> = Object.freeze({});

/** Packages every manifest records (provenance), plus text-preprocessing for document releases. */
const RECORDED_PACKAGES = ['@liquidau/embedding-classifier', '@liquidau/rule-miner', '@liquidau/router'] as const;

/** Packages a manifest must record: text-preprocessing only for document releases. */
export const requiredPackages = (schema: RouterManifest['schema'], edge = false): string[] =>
  [...RECORDED_PACKAGES, ...(schema === MANIFEST_SCHEMA_DOCUMENT || edge ? ['@liquidau/text-preprocessing'] : [])];

const SEGMENT = /^[A-Za-z0-9_][A-Za-z0-9._-]*$/;

/** A relative key of plain segments: no absolute paths, '..', '.', backslashes, schemes or empty segments. */
export function isSafeKey(key: unknown): key is string {
  return typeof key === 'string' && key.length > 0 && key.length <= 1024 && key.split('/').every((s) => SEGMENT.test(s) && s !== '.' && s !== '..');
}

const SEMVER = /^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

/** Why a recorded build-time package version is unusable (not an exact version), or null. */
export function packageProblem(name: string, version: string): string | null {
  return SEMVER.test(version) ? null : `${name}: '${version}' is not an exact version`;
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every((s) => typeof s === 'string');
const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$/;

export function gatesProblems(g: unknown, what: string): string[] {
  if (!isObject(g) || typeof g.passed !== 'boolean' || !strings(g.failures) || !strings(g.warnings)) return [`${what} gates must be { passed, failures: string[], warnings: string[] }`];
  if (g.passed && g.failures.length) return [`${what} gates say passed but record ${g.failures.length} failure(s)`];
  if (!g.passed && !g.failures.length) return [`${what} gates say failed but record no failure`];
  return [];
}

/** Validates a parsed manifest; throws RouterLoadError (UNSUPPORTED_SCHEMA, MANIFEST_INVALID or PACKAGE_INCOMPATIBLE). */
export function validateManifest(raw: unknown): RouterManifest {
  if (!isObject(raw)) throw new RouterLoadError('MANIFEST_INVALID', 'the manifest is not an object');
  if (raw.schema !== MANIFEST_SCHEMA && raw.schema !== MANIFEST_SCHEMA_DOCUMENT) throw new RouterLoadError('UNSUPPORTED_SCHEMA', `unsupported manifest schema ${JSON.stringify(raw.schema)} (this router reads '${MANIFEST_SCHEMA}' and '${MANIFEST_SCHEMA_DOCUMENT}')`);
  const documentRelease = raw.schema === MANIFEST_SCHEMA_DOCUMENT;
  const p: string[] = [];
  const known = ['schema', 'releaseId', 'createdAt', 'files', 'rulesSemanticHash', 'embedding', 'serving', 'gates', 'packages', ...(documentRelease ? ['featureIdentitySha256'] : ['contextEmbedding', 'middleEmbedding'])];
  for (const k of Object.keys(raw)) if (!known.includes(k)) p.push(`unknown field ${k}`);
  if (typeof raw.releaseId !== 'string' || !raw.releaseId.length || raw.releaseId.length > 256) p.push('releaseId must be a non-empty string of at most 256 characters');
  if (typeof raw.createdAt !== 'string' || !ISO.test(raw.createdAt) || !Number.isFinite(Date.parse(raw.createdAt))) p.push('createdAt must be an ISO 8601 date-time');
  const files = raw.files;
  if (!isObject(files)) p.push('files must be an object');
  else {
    const roles = ['classifier', 'rules', 'policy', 'evidence', ...('edge' in files ? ['edge'] : []), ...('middle' in files ? ['middle'] : [])];
    if ('middle' in files && documentRelease) p.push('a document release cannot have a middle tier');
    if ('middle' in files && !('edge' in files)) p.push('files.middle needs files.edge (the middle tier sits behind the edge)');
    if ('middle' in files !== (raw.middleEmbedding !== undefined)) p.push('files.middle and middleEmbedding must be present together');
    for (const k of Object.keys(files)) if (!roles.includes(k)) p.push(`unknown file role ${k}`);
    const keys = new Set<string>();
    for (const role of roles) {
      const f = files[role];
      if (!isObject(f) || Object.keys(f).some((k) => k !== 'key' && k !== 'sha256')) { p.push(`files.${role} must be { key, sha256 }`); continue; }
      if (!isSafeKey(f.key)) p.push(`files.${role}.key ${JSON.stringify(f.key)} is not a safe relative key`);
      else if (keys.has(f.key)) p.push(`files.${role}.key ${f.key} is used twice`);
      else keys.add(f.key);
      if (!isSha256Hex(f.sha256)) p.push(`files.${role}.sha256 must be lowercase hex SHA-256`);
    }
  }
  if (!isSha256Hex(raw.rulesSemanticHash)) p.push('rulesSemanticHash must be lowercase hex SHA-256');
  if (documentRelease && !isSha256Hex(raw.featureIdentitySha256)) p.push('featureIdentitySha256 must be lowercase hex SHA-256 in a document release');
  p.push(...encoderIdentityProblems(raw.embedding).map((e) => `embedding ${e}`));
  if (raw.contextEmbedding !== undefined) p.push(...encoderIdentityProblems(raw.contextEmbedding).map((e) => `contextEmbedding ${e}`));
  if (raw.middleEmbedding !== undefined) p.push(...encoderIdentityProblems(raw.middleEmbedding).map((e) => `middleEmbedding ${e}`));
  const serving = raw.serving;
  if (!isObject(serving) || Object.keys(serving).some((k) => k !== 'ruleInput' && k !== 'maxInputUtf8Bytes')) p.push('serving must be { ruleInput, maxInputUtf8Bytes }');
  else {
    if (serving.ruleInput !== 'original') p.push("serving.ruleInput must be 'original'");
    if (!Number.isInteger(serving.maxInputUtf8Bytes) || (serving.maxInputUtf8Bytes as number) < 1) p.push('serving.maxInputUtf8Bytes must be a positive integer');
  }
  p.push(...gatesProblems(raw.gates, 'manifest'));
  if (!isObject(raw.packages) || !Object.values(raw.packages).every((v) => typeof v === 'string')) p.push('packages must map package names to exact versions');
  if (p.length) throw new RouterLoadError('MANIFEST_INVALID', 'the manifest is invalid', p);

  const packages = raw.packages as Record<string, string>;
  const pk: string[] = [];
  for (const name of requiredPackages(raw.schema as RouterManifest['schema'], isObject(raw.files) && 'edge' in raw.files)) if (packages[name] === undefined) pk.push(`${name}: build-time version not recorded`);
  for (const [name, version] of Object.entries(packages)) { const why = packageProblem(name, version); if (why) pk.push(why); }
  if (pk.length) throw new RouterLoadError('PACKAGE_INCOMPATIBLE', 'the release does not record its build-time package versions exactly', pk);
  return raw as unknown as RouterManifest;
}
