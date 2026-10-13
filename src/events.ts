/**
 * Structured telemetry. Events never contain the request text or embedding vectors; calibrated
 * probabilities appear only where the classifier produced them. The observer is called in event order
 * but never delays a result: waiting for an asynchronous observer runs in the background (Router.flush()).
 * An observer that throws, rejects or exceeds its delivery budget can't change a decision: the event is
 * counted as dropped instead (Router.telemetry()), for the host to export as a metric.
 */
import type { RuntimeErrorCode } from './errors.ts';
import type { ReviewErrorCode } from './review.ts';

export interface EventBase {
  releaseId: string;
  requestId: string;
  mode: 'enforce' | 'shadow';
  /** Monitoring sample rate for rules-settled traffic. */
  samplingProbability: number;
}

/** Document releases: what the document scorer did. No text, chunks, spans or vectors. */
export interface DocumentTelemetry {
  chunkCount: number;
  /** Encoder input tokens over all chunks. */
  inputTokens: number;
  /** Classification-stage encoder calls actually made (shared-vector reuse excluded). */
  embedCalls: number;
  featureIdentitySha256: string;
  /** Planning time, excluding boundary embedding. */
  planningMs: number;
  embeddingMs: number;
  scoringMs: number;
  /** Semantic pipelines: the boundary stage's actual encoder calls, input tokens and time. */
  boundary?: { calls: number; inputTokens: number; ms: number };
}

export interface DecisionEvent extends EventBase {
  type: 'decision';
  outcome: 'route' | 'review' | 'abstain' | 'unavailable';
  routeId: string | null;
  mechanism: 'rule' | 'edge' | 'middle' | 'classifier' | null;
  reason: string | null;
  /** Decided by the rules tier without needing scores. */
  rulesSettled: boolean;
  /**
   * Which tier settled the request without the main encoder: 'rules' (rules and edge) or 'middle'
   * (after the middle encoder); null when the main encoder's scores decided it.
   */
  settledBy?: 'rules' | 'middle' | null;
  /** A rules-settled request selected for monitoring: embedded after the result is returned, reported by its own diagnostic or monitoring_error event. */
  sampled: boolean;
  /** The decision embedded this request (monitoring samples don't count: see `sampled`). */
  embedded: boolean;
  /**
   * Probability this request's scores were included in scored traffic: 1 for unresolved requests,
   * `samplingProbability` for rules-settled ones. Weight drift estimates by its inverse.
   */
  inclusionProbability: number;
  /** The fired rule's id, or 'edge:<label>' when the edge tier fired the route. */
  firedRule: string | null;
  /** What the edge tier decided (absent without an edge): labels it fired and dismissed. */
  edge?: { fired: string[]; dismissed: string[] };
  /** What the middle tier decided (absent when it did not run): labels it fired and dismissed. */
  middle?: { fired: string[]; dismissed: string[] };
  dismissed: readonly string[];
  errorCode: RuntimeErrorCode | null;
  /** reviewMs: the review adapter call (absent when not run). */
  /** middleMs: the middle encoder and its scoring (absent when the middle tier did not run). */
  timings: { totalMs: number; rulesMs: number; embedMs: number | null; scoreMs: number | null; middleMs?: number | null; reviewMs?: number | null };
  calibratedProbabilities?: Readonly<Record<string, number>>;
  document?: DocumentTelemetry;
  /** The review adapter's call for a review outcome: no text, no rationale. */
  review?: { answered: boolean; adapterSha256: string; routeId: string | null; confidence: number | null; errorCode: ReviewErrorCode | null; forHuman: boolean; reviewMs: number };
}

/** A rules-settled request embedded for monitoring: what the classifier says, without changing the decision. */
export interface DiagnosticEvent extends EventBase {
  type: 'diagnostic';
  kind: 'monitoring_sample';
  inclusionProbability: number;
  /** The decision the classifier alone (rules ignored) would make under the same policy. */
  classifierCandidate: { outcome: 'route' | 'review' | 'abstain'; routeId: string | null };
  calibratedProbabilities: Readonly<Record<string, number>>;
  timings: { embedMs: number; scoreMs: number };
  document?: DocumentTelemetry;
}

export interface MonitoringErrorEvent extends EventBase {
  type: 'monitoring_error';
  code: RuntimeErrorCode;
  timings: { embedMs: number };
}

export type RouterEvent = DecisionEvent | DiagnosticEvent | MonitoringErrorEvent;

export interface TelemetryCounters {
  delivered: number;
  /** Observer threw, rejected or exceeded observerTimeoutMs. */
  dropped: number;
}

/** Delivers one event within `timeoutMs`; never throws. Returns whether it was delivered. */
export async function deliver(observer: ((e: RouterEvent) => void | Promise<void>) | undefined, event: RouterEvent, timeoutMs: number): Promise<boolean | null> {
  if (!observer) return null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const r = observer(event);
    if (r && typeof (r as Promise<void>).then === 'function') {
      const timeout = new Promise<'timeout'>((resolve) => { timer = setTimeout(() => resolve('timeout'), timeoutMs); });
      const settled = (r as Promise<void>).then(() => 'ok' as const, () => 'failed' as const);
      return (await Promise.race([settled, timeout])) === 'ok';
    }
    return true;
  } catch {
    return false;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
