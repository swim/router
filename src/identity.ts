/**
 * Encoder identity: everything that determines what an embedding vector means. Two encoders are
 * interchangeable only when every field matches, layer order included. Host independence does not make
 * different embedding models interchangeable: a release trained on one identity is served only by an
 * adapter that implements exactly that identity.
 */
import type { EmbeddingSpec } from '@liquidau/embedding-classifier';
import { ENCODER_IDENTITY_FIELDS, ENCODER_IDENTITY_SCHEMA, fullEncoderIdentityProblems, vectorProblems, type EncoderAdapter, type FullEncoderIdentity } from '@liquidau/text-preprocessing';

import { canonicalJson, sha256Hex, utf8 } from './bytes.ts';

// The identity and its checks are text-preprocessing's, shared with embedding-classifier.
export { ENCODER_IDENTITY_SCHEMA, vectorProblems };
export type EncoderIdentity = FullEncoderIdentity;
export type Encoder = EncoderAdapter;

const FIELDS = ENCODER_IDENTITY_FIELDS;

/** Problems with an encoder identity (empty when valid). Unknown fields are refused, not ignored. */
export const encoderIdentityProblems = fullEncoderIdentityProblems;

/** Field-by-field differences between the identity a release was built for and an adapter's (empty when identical). */
export function encoderIdentityDiff(expected: Readonly<EncoderIdentity>, actual: Readonly<EncoderIdentity>): string[] {
  return FIELDS
    .filter((f) => canonicalJson(expected[f] ?? null) !== canonicalJson(actual[f] ?? null))
    .map((f) => `${f}: the release needs ${JSON.stringify(expected[f] ?? null)}, the encoder provides ${JSON.stringify(actual[f] ?? null)}`);
}

/** Canonical digest of a complete identity: the encoder half of an embedding cache key. */
export async function encoderIdentityDigest(identity: Readonly<EncoderIdentity>): Promise<string> {
  return sha256Hex(utf8(canonicalJson(FIELDS.map((f) => [f, identity[f]]))));
}

/**
 * An embedding cache key: the full identity digest plus a digest of the ACTUAL preprocessed input the
 * encoder embeds, within a caller-chosen scope (e.g. a tenant), so a changed model, setting or tenant
 * can never reuse a stale vector. No cache is needed for correctness.
 */
export async function embeddingCacheKey(identity: Readonly<EncoderIdentity>, preprocessedInput: string, scope: string): Promise<string> {
  const input = await sha256Hex(utf8(preprocessedInput));
  return sha256Hex(utf8(canonicalJson(['liquidau-embedding-cache/1', scope, await encoderIdentityDigest(identity), input])));
}

/**
 * How a release's encoder identity disagrees with the classifier artifact's recorded embedding spec,
 * on the fields both describe. Missing artifact fields that the identity requires (precision) are
 * reported, never inferred. Normalisation maps as:
 *   'unit'            no layers, normalize: true
 *   'per-layer-unit'  layers, layer_normalize: true (the classifier's `normalize` does not describe a
 *                     concatenation of layers, so it is not compared)
 *   'none'            normalize: false and layer_normalize not true
 */
export function classifierEmbeddingDiff(identity: Readonly<EncoderIdentity>, spec: EmbeddingSpec): string[] {
  const out: string[] = [];
  const cmp = (field: string, artifact: unknown, release: unknown) => {
    if (canonicalJson(artifact ?? null) !== canonicalJson(release ?? null)) out.push(`${field}: the classifier records ${JSON.stringify(artifact ?? null)}, the release identity says ${JSON.stringify(release ?? null)}`);
  };
  cmp('model_id/modelId', spec.model_id, identity.modelId);
  cmp('dimensions', spec.dimensions, identity.dimensions);
  if (spec.precision === undefined) out.push('precision: the classifier artifact does not record embedding.precision (required; convert the artifact rather than assume one)');
  else cmp('precision', spec.precision, identity.precision);
  cmp('input_type/inputType', spec.input_type, identity.inputType);
  cmp('layers', spec.layers, identity.layers);
  cmp('max_chars/maxChars', spec.max_chars, identity.maxChars);
  if (spec.layers !== undefined) cmp('pooling', spec.pooling, identity.pooling);
  const normalization = spec.layers !== undefined && spec.layer_normalize === true ? 'per-layer-unit'
    : spec.layers === undefined && spec.normalize === true ? 'unit'
    : spec.normalize === false && spec.layer_normalize !== true ? 'none'
    : null;
  if (normalization === null) out.push(`normalization: the classifier's normalize ${spec.normalize} / layer_normalize ${spec.layer_normalize ?? null} has no single meaning`);
  else if (normalization !== identity.normalization) out.push(`normalization: the classifier records ${normalization}, the release identity says ${identity.normalization}`);
  return out;
}
