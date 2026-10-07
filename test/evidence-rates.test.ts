import assert from 'node:assert/strict';
import { test } from 'node:test';

import { RouterLoadError, type ReleaseEvidence } from '../src/index.ts';
import { checkAcceptance, rate, type FinalReport, type PerHeadReport } from '../src/evaluation/index.ts';
import { makeRelease } from './helpers.ts';

const invalid = (pattern: RegExp) => (e: unknown) => e instanceof RouterLoadError && e.code === 'EVIDENCE_INVALID' && pattern.test(e.details.join('; '));
/** Edits the recorded report, then records the gates its criteria give on the edited report. */
const forge = (edit: (final: FinalReport, perHead: PerHeadReport) => void) => (e: ReleaseEvidence) => {
  edit(e.final as unknown as FinalReport, e.perHead as unknown as PerHeadReport);
  e.gates = checkAcceptance(e.final as unknown as FinalReport, e.perHead as unknown as PerHeadReport, e.criteria as never);
};

test('recorded rates must follow from the report counts: a forged coverage cannot pass the recomputed gates', async () => {
  const honest = await makeRelease({ evaluationPasses: false });
  assert.ok(honest.enforcement.some((p) => /gates did not pass/.test(p)), 'the honest report fails its criteria');
  // Nothing routed, but coverage recorded as 100%: the recomputed gates would pass on the recorded rate.
  await assert.rejects(makeRelease({ evaluationPasses: false, evidence: forge((f) => {
    f.outcomes = { route: 0, review: 0, abstain: f.items };
    f.coverage = rate(f.items, f.items);
  }) }), invalid(/final\.coverage\.count is 6, the report's counts give 0/));
});

test('a rate whose estimate or interval is not computed from its counts is refused', async () => {
  await assert.rejects(makeRelease({ evidence: forge((f) => { f.routedAccuracy = { ...f.routedAccuracy, rate: 0.123 }; }) }), invalid(/final\.routedAccuracy must be/));
  await assert.rejects(makeRelease({ evidence: forge((f) => { const r = f.perRoute.urgent.recall; f.perRoute.urgent.recall = { ...r, wilson95: [0.99, 1] }; }) }), invalid(/perRoute\.urgent\.recall must be/));
  await assert.rejects(makeRelease({ evidence: forge((f) => { f.mechanism = { rule: f.mechanism.rule + 1, classifier: f.mechanism.classifier }; }) }), invalid(/final\.mechanism/));
  await assert.rejects(makeRelease({ evidence: forge((f) => { f.perRoute.urgent.routed += 1; f.perRoute.urgent.precision = rate(f.perRoute.urgent.precision.count, f.perRoute.urgent.routed); }) }), invalid(/perRoute routes/));
});

test('per-head rates must come from one confusion matrix', async () => {
  // One more labelled item than recall's and false alarms' denominators account for.
  await assert.rejects(makeRelease({ evidence: forge((_f, h) => { h.heads.urgent.labelled += 1; }) }), invalid(/perHead\.heads\.urgent: .*one confusion matrix/));
  // True positives that differ between recall and precision.
  await assert.rejects(makeRelease({ evidence: forge((_f, h) => { const r = h.heads.urgent; r.recall = rate(r.recall.count - 1, r.recall.of); }) }), invalid(/perHead\.heads\.urgent: .*one confusion matrix/));
});
