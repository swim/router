/**
 * The stable error contract. Load failures throw RouterLoadError with a code from LOAD_ERROR_CODES;
 * request-time dependency failures are returned as `unavailable` results with a RuntimeErrorCode,
 * never thrown and never disguised as a negative label or an abstention.
 *
 * Programmer errors (calling the decision evaluator with labels outside the policy, passing an
 * invalid decision model) throw ordinary Errors.
 */

export const RUNTIME_ERROR_CODES = [
  'INVALID_INPUT', 'ENCODER_TIMEOUT', 'ENCODER_FAILURE', 'INVALID_EMBEDDING', 'INVALID_SCORE', 'ABORTED', 'CLOSED',
  // Split deployments only: a forwarded request names another release than this tier serves.
  'RELEASE_MISMATCH',
  // Document releases ('liquidau-router/2') only:
  'PREPROCESSING_FAILURE', 'INPUT_LIMIT_EXCEEDED', 'PROCESSING_TIMEOUT',
] as const;
export type RuntimeErrorCode = (typeof RUNTIME_ERROR_CODES)[number];

/** Whether retrying the same request (on this or a replacement router) can succeed. */
export const RETRYABLE: Readonly<Record<RuntimeErrorCode, boolean>> = Object.freeze({
  INVALID_INPUT: false,
  ENCODER_TIMEOUT: true,
  ENCODER_FAILURE: true,
  INVALID_EMBEDDING: false,
  INVALID_SCORE: false,
  ABORTED: false,
  CLOSED: true,
  /** The tiers serve different releases, e.g. mid-rollout: retry once both run the same manifest. */
  RELEASE_MISMATCH: true,
  /** The tokenizer adapter failed: deterministic and I/O-free by contract, so not retryable unchanged. */
  PREPROCESSING_FAILURE: false,
  /** The document exceeds a release or host limit (bytes, chunks, planning steps, token budget). */
  INPUT_LIMIT_EXCEEDED: false,
  /** The single document-processing deadline (planning, embedding batches and scoring) passed. */
  PROCESSING_TIMEOUT: true,
});

export const LOAD_ERROR_CODES = [
  'OPTIONS_INVALID',
  'PLATFORM_UNSUPPORTED',
  'INVALID_KEY',
  'READ_FAILED',
  'READ_TIMEOUT',
  'ABORTED',
  'FILE_TOO_LARGE',
  'MANIFEST_DIGEST_MISMATCH',
  'FILE_DIGEST_MISMATCH',
  'DOCUMENT_MALFORMED',
  'UNSUPPORTED_SCHEMA',
  'MANIFEST_INVALID',
  'PACKAGE_INCOMPATIBLE',
  'CLASSIFIER_INVALID',
  'RULES_INVALID',
  'POLICY_INVALID',
  'EVIDENCE_INVALID',
  'PAIRING_MISMATCH',
  'EMBEDDING_MISMATCH',
  'ENCODER_IDENTITY_MISMATCH',
  'GATES_FAILED',
  /** A document release needs an adapter the host didn't supply (the tokenizer). */
  'CAPABILITY_MISSING',
  /** The tokenizer adapter doesn't implement the release pipeline's tokenizer identity. */
  'TOKENIZER_MISMATCH',
] as const;
export type LoadErrorCode = (typeof LOAD_ERROR_CODES)[number];

/** Why a release (or the options to load it) was refused. `details` lists every problem found by the failing step. */
export class RouterLoadError extends Error {
  override readonly name = 'RouterLoadError';
  readonly code: LoadErrorCode;
  readonly details: readonly string[];

  constructor(code: LoadErrorCode, message: string, details: readonly string[] = []) {
    super(details.length ? `${message}: ${details.join('; ')}` : message);
    this.code = code;
    this.details = Object.freeze([...details]);
  }
}

/**
 * What an encoder adapter throws to translate a provider failure into the core contract. Any other
 * thrown value is treated as ENCODER_FAILURE (retryable). Provider diagnostics stay in host logs.
 */
export class EncoderError extends Error {
  override readonly name = 'EncoderError';
  readonly code: 'ENCODER_FAILURE' | 'ENCODER_TIMEOUT';
  readonly retryable: boolean;

  constructor(code: 'ENCODER_FAILURE' | 'ENCODER_TIMEOUT', message: string, options: { retryable?: boolean } = {}) {
    super(message);
    this.code = code;
    this.retryable = options.retryable ?? RETRYABLE[code];
  }
}
