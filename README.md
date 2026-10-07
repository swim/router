# @liquidau/router

Serves a release built with [@liquidau/embedding-classifier](https://www.npmjs.com/package/@liquidau/embedding-classifier)
and [@liquidau/rule-miner](https://www.npmjs.com/package/@liquidau/rule-miner): a classifier, a rule
set, a routing policy and evaluation evidence, pinned by one manifest digest. Each request runs the
rules first and is embedded only when a classifier score could change the outcome. The router returns
a decision; your application acts on it.

```
buildEvidence ──► buildRelease ──► release: manifest, classifier, rules, policy, evidence
                                        │
                                        ▼
                  loadRouter: digests, pairing, encoder identity, gates
                                        │
request ──► rules ──► settled? ── yes ─────────────────────────────────► result
                         │ no                                              ▲
                         ▼                                                 │
                      encoder ──► classifier heads ──► policy ─────────────┘
                                                                    events ──► observer

split:  loadRulesTier (rules function) ── forward (JSON) ──► routeForwarded (model function)
```

It uses no Node built-ins or cloud SDKs, so it runs in Node, serverless functions and edge runtimes.
You supply the release files (a `ReleaseSource`) and the encoder that embeds text.

```sh
npm install @liquidau/router
```

## Usage

```ts
import { loadRouter } from '@liquidau/router';

const router = await loadRouter({
  source,                                  // ReleaseSource: read(key, { signal }) -> Uint8Array
  manifestKey: 'manifest.json',
  expectedManifestSha256: config.releaseSha256,
  mode: 'enforce',                         // or 'shadow'
  encoder,                                 // must implement the release's encoder identity exactly
  timeoutMs: 2000,
  monitoring: { sampleRate: 0.05, samplingSalt: 'per-deployment-secret' },
  observer: (event) => log(event),         // optional telemetry
});

const result = await router.route({ text, requestId });
if (result.mode === 'enforce' && result.actionable) dispatch(result.destination);
await router.flush();                      // finish monitoring and telemetry (see Serverless)
```

| Result | Meaning |
|---|---|
| `outcome: 'route'`, `actionable: true` | Act on `destination`. `mechanism` is `'rule'` or `'classifier'` |
| `outcome: 'review'` | Send to a person: `near_threshold` or `abstention_policy` |
| `outcome: 'abstain'` | No route: `no_match` or `all_dismissed` |
| `outcome: 'shadow'` | Shadow mode: `candidate` is what enforce mode would decide. Never act on it |
| `outcome: 'unavailable'` | No decision: `code` (`INVALID_INPUT`, `ENCODER_TIMEOUT`, `ENCODER_FAILURE`, `INVALID_EMBEDDING`, `INVALID_SCORE`, `ABORTED`, `CLOSED`, `RELEASE_MISMATCH`, and for document releases `PREPROCESSING_FAILURE`, `INPUT_LIMIT_EXCEEDED`, `PROCESSING_TIMEOUT`) and `retryable` |

Load failures throw `RouterLoadError` with a stable `code` and `details`. `close()` refuses new
requests and resolves when in-flight requests and background work finish. `RouterSlot` swaps in a new
release only after it loads, and keeps the old one if it doesn't.

## What loading checks

- The manifest's bytes against `expectedManifestSha256`, and every release file's SHA-256 before parsing.
- Each document with its own library, and the policy against the classifier's heads and the rule labels.
- That the documents belong together: the rule set the classifier was trained with, the dismissal
  rules it was certified with, and the exact files the evidence evaluated.
- That the encoder implements the release's identity: model, revision, layers, pooling, normalisation,
  tokenizer and limits.
- In `enforce` mode, that the classifier's and the release's gates passed and the evidence supports
  them: a final routing evaluation, recorded acceptance criteria, and gates that follow from both.
  Shadow mode accepts failed gates, never invalid or mismatched documents.

Releases record the library versions they were built with (pass the installed versions to
`buildRelease`). The router doesn't check them: serve a release with compatible versions of
embedding-classifier and rule-miner, ideally the ones it recorded.

## How a request is decided

The rules run on the original text. A request is settled by the rules alone when no score could
change the outcome: a rule fires the first-priority head and nothing can suppress it, or dismissal
rules clear every head. Otherwise the router embeds the text, checks the vector (width, finite values,
normalisation), scores every head, and applies the policy: priority, suppression, the review band,
then `onAbstain`.

## Monitoring

A deterministic sample of rules-settled requests (`monitoring.sampleRate`, selected by release, salt
and request id) is still embedded, so drift checks see all traffic. Monitoring never changes a decision
and runs after the result is returned. The observer receives one decision event per request (outcome,
mechanism, rule, probabilities, timings) and a diagnostic event per monitoring sample; events never
contain the request text. `inclusionProbability` on each event says how the scored traffic was
sampled. `telemetry()` counts delivered and dropped events.

## Split deployment

The rules can run in their own small function in front of the model's function:

```ts
import { loadRulesTier } from '@liquidau/router';

// Rules function: reads the manifest, rules, policy and evidence; no classifier, no encoder.
const rules = await loadRulesTier({ source, manifestKey, expectedManifestSha256, mode: 'enforce', timeoutMs: 2000, monitoring });
const outcome = await rules.handle({ text, requestId });
if (outcome.answered) {
  respond(outcome.result);
  if (outcome.monitor) await send(outcome.monitor);    // telemetry only
} else {
  respond(await send(outcome.forward));                 // JSON to the model function
}

// Model function: loadRouter on the same manifest digest.
const result = await router.routeForwarded(forward);   // null for a monitoring forward
```

Both functions run the same checks and rules, so the pair decides exactly as one router does. Use the
same manifest digest, mode and monitoring settings in both. The model function answers a forward from
another release with `RELEASE_MISMATCH` (retryable, e.g. mid-rollout). A split helps when the rules
settle a large share of traffic; the evaluation evidence reports that share (`rulesSettlementRate`).

## Serverless

- Load at module scope, once per instance. Loading reads and validates every release file.
- Read the manifest digest from your deployment configuration. Roll back by deploying the previous digest.
- Await `router.flush()` before the invocation ends, or pass it to `ctx.waitUntil` on Workers;
  otherwise monitoring and telemetry may not run.
- Size memory for the encoder's model plus any kNN reference (4 bytes × rows × dimensions). kNN and
  stack heads cost more per request than linear heads.
- The rules function has no model, so it starts quickly and fits edge runtimes. The model function's
  cold start is mostly loading the model.

## Building a release

```ts
import { buildRelease, createDecisionEvaluator, documentDigest, encodeJson } from '@liquidau/router';
import { buildEvidence, evaluateFinal, evaluatePerHead } from '@liquidau/router/evaluation';

const docs = { classifier: encodeJson(artifact), rules: encodeJson(ruleSet), policy: encodeJson(policy) };
const final = evaluateFinal(createDecisionEvaluator(model, policy), heldOutItems);
const perHead = evaluatePerHead(model, heldOutItems);
const evidence = buildEvidence({
  inputs: { classifierSha256: await documentDigest(docs.classifier), rulesSha256: ..., policySha256: ..., rulesSemanticHash },
  dataset, final, perHead, criteria: { routes: { urgent: { minRecall: 0.95 } } },
});
const release = await buildRelease({ releaseId, documents: { ...docs, evidence: encodeJson(evidence) }, embedding: encoderIdentity, serving, packages });
// release.files to upload; release.manifestSha256 to pin; release.enforcement lists why it can only run in shadow mode
```

Train the artifact with `buildArtifact(result, { ..., router: { ruleSetHash } })` so it records its
rule set. Evaluation numbers are point estimates with Wilson intervals on the held-out items, not
guarantees; the classifier's own guarantees come from its calibration.

## Document releases

A release whose classifier scores whole documents (embedding-classifier's `buildDocumentArtifact`)
also needs the pipeline's `tokenizer` (and, for semantic grouping, `boundaryEncoder` and
`boundaryTokenizer`) when loading. `timeoutMs` then covers the whole document: planning, every
embedding batch and scoring.

## Adapters

Release sources and encoders for specific platforms (filesystem, object storage, local or hosted
models) live in your application or separate packages. `encoderContractProblems` and
`releaseSourceContractProblems` check an adapter against the contract; `memorySource` and
`functionEncoder` are in-memory implementations for tests.

## Development

```sh
npm test
npm run typecheck        # includes a pass without Node types
npm run check:portable   # bundles for a neutral platform with no Node built-ins
```
