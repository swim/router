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
| `outcome: 'route'`, `actionable: true` | Act on `destination`. `mechanism` is `'rule'`, `'edge'`, `'middle'` or `'classifier'` |
| `outcome: 'review'` | Send to a person: `near_threshold` or `abstention_policy`. With a review adapter, also its advisory `answer` (or `answerError`) and `forHuman` |
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
  With context features, the same for the context encoder.
- In `enforce` mode, that the classifier's and the release's gates passed and the evidence supports
  them: a final routing evaluation, recorded acceptance criteria, and gates that follow from both.
  Shadow mode accepts failed gates, never invalid or mismatched documents.

Releases record the library versions they were built with (pass the installed versions to
`buildRelease`). The router doesn't check them: serve a release with compatible versions of
embedding-classifier and rule-miner, ideally the ones it recorded.

## How a request is decided

The rules run on the original text. Every head a rule (or the edge or middle tier) fires counts as at
its threshold, as certification counted it, so several fired heads are all honoured, suppression
included (`RuleFindings.firing`; `fired` and `firedRule` name the highest-priority one). A request is
settled by the rules alone when no score could change the outcome: the first head that can win is
fired and nothing can suppress it, every head that could win is suppressed by a fired head (review),
or dismissal rules clear every head. Otherwise the router embeds the text, checks the vector (width, finite values,
normalisation), scores every head, and applies the policy: priority, suppression, the review band,
then `onAbstain`.

**The edge tier.** A release may carry an edge document (`edge.json`, embedding-classifier's
`trainEdge`): per head, a blend of the rules, a hashed lexical model and a static embedding that fires
or dismisses the head before any embedding. Its decisions join the rule findings: dismissed heads are
added to the rules' dismissals (except a label a rule fired), and its fires join the fired heads
(route `mechanism: 'edge'`, `firedRule: 'edge:<label>'`). Settlement
is unchanged, so an edge that dismisses or fires decisively settles the request without the model. The
loader checks that every head the edge decides was certified with exactly that head's edge model
(`checkEdgePairing`, by per-head digest), that the evidence evaluated it, and, when it scores a static embedding, that
`staticEncoder` (text-preprocessing's `createStaticEmbedder` over the same tokenizer and weight bytes)
has the document's identity (`STATIC_ENCODER_MISMATCH` otherwise). `staticEncoder` may be a loader
(`() => Promise<StaticEmbedder>`), called only for releases whose edge needs it, so the weight table is
not held for releases that don't use it. Pass the same `staticEncoder` to `loadRulesTier`: the edge function runs the edge, and the model function recomputes it from the text.
An edge that fails on a message (its static encoder throwing, say) answers `unavailable` with
`PREPROCESSING_FAILURE`; an edge whose heads use the `vector` scorer is refused at load
(`CAPABILITY_MISSING`), since the edge tier runs before any encoder and has no host vector.
In the lab, regex + lexical + static settled 34-88% of traffic at the edge with end-to-end recall within
0.02 of the model alone (about 80 µs per message).

## Monitoring

A deterministic sample of rules-settled requests (`monitoring.sampleRate`, selected by release, salt
and request id) is still embedded, so drift checks see all traffic. Monitoring never changes a decision
and runs after the result is returned. The observer receives one decision event per request (outcome,
mechanism, rule, probabilities, timings) and a diagnostic event per monitoring sample; events never
contain the request text. `inclusionProbability` on each event says how the scored traffic was
sampled. `telemetry()` counts delivered and dropped events.

**Drift checks.** Store the events, and check each window (an hour, a day, N requests) against a
reference window: the release's own traffic, routed by the same release.

```ts
import { checkWindow, windowOf } from '@liquidau/router';

// From the release: its rule set, the head names, and each head's { threshold, review_floor }.
const thresholds = Object.fromEntries(Object.entries(classifier.heads).map(([h, s]) => [h, { threshold: s.threshold, review_floor: s.review_floor }]));
const heads = Object.keys(thresholds);
const labelOf = (ruleId: string) => rules.rules.find((r) => r.id === ruleId)?.label;
const reference = windowOf(referenceEvents, heads, labelOf);
const report = checkWindow(reference, windowOf(liveEvents, heads, labelOf), thresholds, { alpha: 0.01 });
if (report.retrain.recommended) { /* label a fresh sample, train, release, shadow, promote */ }
if (report.investigate.length) { /* the service is failing: retraining won't help */ }
```

- **What is compared.**
  - Scores, per head, in two strata kept apart so no weighting is needed: the requests the router scored,
    and the monitoring sample of rules-settled requests (with embedding-classifier's `monitorWindow`).
  - The rates the scores never see: the share settled without the model, each label's rule firing, the
    edge tier's fires and dismissals per label, and unavailable results.
  - Watch the edge's rates in particular. When the edge settles most traffic, a surge in one label shows
    up in its fires and dismissals, not in the scores. In the lab, a surge of account-access requests on
    ABCD raised no alert until these rates were checked.
- **The false-alarm rate.** All checks share one family-wise false-alarm rate (`alpha`, Bonferroni).
- **What an alert means.** A drift alert recommends retraining. An alert on unavailable results says to
  investigate the encoder or the release source instead.
- **Thin strata.** A score stratum with fewer than `minScores` (default 30) scores in either window is
  skipped and listed in `skipped`. At a 5% monitoring rate, that is about 600 rules-settled requests per
  window.
- **Dropped events.** `telemetry().dropped` weakens every check, so watch it too.

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

## Context features

A classifier can score a message together with its context, such as the conversation's next turn
or an assistant's reply. Its heads are then trained on the message's embedding followed by the
context's (embedding-classifier's `embedding.context`), each from its own encoder. Build the release
with `contextEmbedding` (the context encoder's full identity), load it with `contextEncoder`, and send
`context` with every request:

```ts
const release = await buildRelease({ ..., embedding: messageIdentity, contextEmbedding: contextIdentity });
const router = await loadRouter({ ..., encoder: messageEncoder, contextEncoder });
await router.route({ text: firstMessage, context: agentReply, requestId });
```

Loading checks the classifier's context spec against `contextEmbedding`, and the adapter against
both identities. A release with context features refuses requests without `context`; one without
refuses requests with it (`INVALID_INPUT`), so a host never believes context is used when it isn't.
The context has the same byte limit as the text. The two embeddings run in parallel under one
`timeoutMs`. Rules and the edge read only the text, and split deployments forward the context to the
model tier. Pass context exactly as training did, including whatever stood for "no context", and route at
the moment that context exists (for a next-turn context, after the agent's first reply).

In the lab, context was the largest single gain on ABCD `access` (false alarms at recall 0.85: 6.95%
to 2.63% on mpnet; 5.10% to 1.87% with EmbeddingGemma for the message and mpnet for the context). On
ToxicChat `toxic` (the assistant's reply as context) the gain was within the seed-to-seed spread
(re-checked 12 October with a stricter rule: the option helps ABCD `access`, one head). Embedding
the context with EmbeddingGemma as well gave no more than mpnet, at a second 140 ms pass
(RESEARCH.md, stages CX and GX).

## Middle tier

A release can put a cheaper encoder's head between the edge and the main encoder, per head:
embedding-classifier's `trainMiddle` and `HeadInput.cascade`. It fires and dismisses where it is sure,
and those requests never reach the main encoder. Training decides per head whether it pays, from the
training rows only. Build the release with the middle document and `middleEmbedding` (the middle
encoder's full identity), then load it with `middleEncoder`:

```ts
const release = await buildRelease({ ..., documents: { ..., edge, middle }, embedding: mainIdentity, middleEmbedding: middleIdentity });
const router = await loadRouter({ ..., encoder: mainEncoder, middleEncoder });
```

Per request, after the rules and the edge:
- **Which heads it scores:** the middle tier scores only the heads they left undecided.
- **How its decisions merge:** with the edge tier's precedence (`mergeMiddle`). A rule's dismissal beats
  a middle fire, and fires compete by priority.
- **Settling:** the request then settles again. A request the middle tier settles is routed without the
  main encoder. Its event carries `settledBy: 'middle'`, `middle: { fired, dismissed }` and
  `timings.middleMs`, and a middle fire routes with `mechanism: 'middle'`.
- **Deadline:** the middle encoder and the main embedding share one `timeoutMs`.

Loading refuses:
- a missing `middleEncoder` (`CAPABILITY_MISSING`), or one supplied without a middle tier
  (`OPTIONS_INVALID`);
- an adapter that isn't the middle identity (`ENCODER_IDENTITY_MISMATCH`);
- a middle document that doesn't match `middleEmbedding` (`EMBEDDING_MISMATCH`), or is invalid
  (`MIDDLE_INVALID`);
- any pairing hole (`PAIRING_MISMATCH`):
  - a middle model whose digest, thresholds included, isn't the one its head was certified with;
  - a middle model certified behind another edge model;
  - evidence that evaluated another middle file.

Split deployments:
- The rules tier reads the manifest's middle entry but not the file.
- The model tier runs the middle tier on forwarded requests.
- Offline evaluation builds its items with `middleHeads`, `middleDecisions` and `mergeMiddle`, and passes
  `beforeMiddle` so `evaluateFinal` reports `middleSettlementRate` separately.

Only legacy releases with an edge may have a middle tier.

In the lab, mpnet as the middle tier in front of EmbeddingGemma, kept where training chose it, cut the
cost per message from 81 to 26 ms on ToxicChat `toxic` and from 91 to 72 ms on ADE. False alarms were no
worse, and every head kept its guarantee. On Banking77 `urgent` it cut 16.7 to 12.6 ms at +0.10 points of
false alarms, within the seed-to-seed spread (RESEARCH.md, stage DY).

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

**Review adapters.** `loadRouter({ ..., review: { adapter, timeoutMs, humanSampleRate } })` sends every
`review` to an injected adapter: an LLM behind any provider (a direct API, Bedrock, Vertex) or a
synchronous human queue. The adapter has an identity (`provider`, `model`, `promptSha256`,
`outputSchemaSha256`) and one call, `review(input, { signal })`, which sees the text, the policy's
routes and the evidence (reason, rule findings, calibrated probabilities), never the request id or the
release. Its answer is advisory: the result stays a non-actionable `review` carrying `answer` (route id,
destination, confidence, the adapter's identity digest), or `answerError` when it refused, failed,
timed out or answered outside the schema. `forHuman` marks reviews to send to a person: every one
without an answer, plus a deterministic `humanSampleRate` share of answered ones, whose labels estimate
the adapter's accuracy per route. Telemetry records the answer's route, confidence and timing, never the
text or rationale. Give `loadRulesTier` the same `review` option: the rules alone can settle a request
to review (every head dismissed with `onAbstain: 'review'`), and both tiers must answer it alike.
`reviewAdapterContractProblems` checks an adapter (identity, in-schema answers, a timely response, an
already-aborted signal rejecting).

How much to trust an adapter, from the lab's Clef-backed adapter on Banking77 and ToxicChat reviews
(liquidau-lab RESEARCH.md, stage L10): it agreed with gold on 78-90% of reviews, and a host acting on
answers given at confidence 0.9 or more raised recall (refund 0.82 to 0.95, toxic 0.83 to 0.88) without
losing precision. Treat that as a per-deployment measurement, not a default. A ±0.05 interval on
agreement needs about 250 labelled reviews; at `humanSampleRate: 0.05` twenty windows of 1,000 requests
yielded only 4 to 38. Send every review to a person (`humanSampleRate: 1`) until about 250 per route
are labelled, then lower the rate to keep watching for drift.

## Development

```sh
npm test
npm run typecheck        # includes a pass without Node types
npm run check:portable   # bundles for a neutral platform with no Node built-ins
```
