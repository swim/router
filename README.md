# @liquidau/router

One validated entry point over
[@liquidau/embedding-classifier](../embedding-classifier) and [@liquidau/rule-miner](../rule-miner).
It loads an immutable bundle (classifier artifact, rule set, routing policy and evaluation evidence,
pinned by a manifest digest), checks everything once, then routes each request rules-first and embeds
only when a score could change the outcome.

The router returns decisions; it never executes them. It contains no Node built-ins, cloud SDKs,
inference engines or credential handling, so the same code and release bytes run in a local process,
a container, a function or an edge runtime. Hosts provide the release source, the encoder and telemetry.

It wraps the classifier's `decide` / `settleWithRules`, so the router and existing callers decide
identically. Requires embedding-classifier 0.8 and rule-miner 0.5.1. Until those are published, run
`npm run link:siblings` before `npm install` for local development, and don't commit a lockfile
produced that way.

## Serving

```ts
import { loadRouter } from '@liquidau/router';

const router = await loadRouter({
  source,                                   // ReleaseSource: read(key, { signal }) -> Uint8Array
  manifestKey: 'manifest.json',
  expectedManifestSha256: process.env.RELEASE_SHA256!, // host config; the core reads no env
  mode: 'enforce',                          // or 'shadow'; required
  encoder,                                  // implements the release's exact EncoderIdentity
  timeoutMs: 2000,
  monitoring: { sampleRate: 0.05, samplingSalt: 'per-deployment-secret' },
  observer: (event) => log(event),          // optional, bounded delivery
});

const result = await router.route({ text, requestId });
if (result.mode === 'enforce' && result.actionable) dispatch(result.destination); // check the union, not a boolean
await router.flush();                       // in a function host, before the invocation ends (or ctx.waitUntil)
```

Results are a discriminated union:

| Result | Meaning |
|---|---|
| `{ mode: 'enforce', actionable: true, outcome: 'route', routeId, destination, mechanism }` | Act on `destination`. `mechanism` is `'rule'` or `'classifier'` |
| `{ mode: 'enforce', actionable: false, outcome: 'review', reason }` | `near_threshold` or `abstention_policy` |
| `{ mode: 'enforce', actionable: false, outcome: 'abstain', reason }` | `no_match` or `all_dismissed` |
| `{ mode: 'shadow', actionable: false, outcome: 'shadow', candidate }` | What enforce would have decided; never act on it |
| `{ actionable: false, outcome: 'unavailable', code, retryable }` | `INVALID_INPUT`, `ENCODER_TIMEOUT`, `ENCODER_FAILURE`, `INVALID_EMBEDDING`, `INVALID_SCORE`, `ABORTED`, `CLOSED`, and in a split deployment `RELEASE_MISMATCH`. Never a negative label |

Timeouts must be in [1, 2147483647] ms (longer delays overflow timers). Load failures throw `RouterLoadError` with a stable `code` and `details`. `close()` refuses new
requests and resolves once in-flight ones and background work finish. Monitoring embeddings and
asynchronous observer deliveries run after `route()` returns, so neither delays a result; `flush()`
waits for them. In a function host (Lambda, Workers), await `flush()` before the invocation ends or
hand it to the platform (`ctx.waitUntil`), or that work may be frozen or dropped. `RouterSlot` swaps in a fully validated replacement
and keeps the old one if the reload fails. `slot.close()` is terminal: later or queued reloads are
refused, a replacement that finishes loading after close is closed rather than installed, and close
waits for retired routers to drain.

## Split deployment: a rules tier in front of the model

The deterministic rules can run in their own small function (a Lambda, or an edge Worker) and send
only the requests they can't settle to the model's function:

```ts
import { loadRouter, loadRulesTier } from '@liquidau/router';

// Rules function: loads the manifest, rules, policy and evidence - never the classifier - so it
// starts in milliseconds and needs no encoder.
const rules = await loadRulesTier({ source, manifestKey, expectedManifestSha256, mode: 'enforce', timeoutMs: 2000,
  monitoring: { sampleRate: 0.05, samplingSalt } });
const outcome = await rules.handle({ text, requestId });
if (outcome.answered) {
  respond(outcome.result);
  if (outcome.monitor) await sendToModelTier(outcome.monitor);   // telemetry only, after the answer
} else {
  respond(await sendToModelTier(outcome.forward));              // JSON; the model tier decides
}

// Model function: an ordinary router on the SAME manifest digest.
const router = await loadRouter({ source, manifestKey, expectedManifestSha256, mode: 'enforce', encoder, timeoutMs: 2000,
  monitoring: { sampleRate: 0.05, samplingSalt } });
const result = await router.routeForwarded(forward);            // null for a 'monitor' forward
await router.flush();
```

- **Exactly one process's decisions.** Both tiers run the same input checks, rules and settlement
  code; a `'decide'` forward is routed as `route()` would route it, and the two tiers' telemetry
  together is what one router emits. Use the same mode and monitoring settings in both.
- **One release.** Both tiers pin the same manifest digest. A forward carries it, and the model tier
  answers a forward from another release with `RELEASE_MISMATCH` (retryable: during a rollout, retry
  once both tiers serve the new release). The rules tier checks every file it reads against the
  manifest; the model tier validates the classifier and its pairing with the rules.
- **When it pays.** The rules tier answers only what no score could change: a rule firing the
  first-priority head that nothing can suppress, or every head dismissed. How much traffic that is
  depends on the release: measure it on your own traffic (the evaluation evidence reports
  `rulesSettlementRate`) before splitting.
- `createSettler(policy)` is the rules tier's settlement on its own, for offline checks.

## Serverless deployment

- **Load once per instance.** Call `loadRouter` (or `loadRulesTier`) at module scope, so a warm
  instance reuses the validated release; loading reads and hashes every release file and decodes any
  kNN reference, so it belongs in the cold start, not in a request.
- **The release.** Bundle the files with the function, or read them from object storage through a
  `ReleaseSource`; pass the manifest digest from deployment configuration (an environment variable read
  by the host, never by the router). A new release is a new digest: deploy it, then roll back by
  redeploying the previous digest. In a long-running process, `RouterSlot.reload` swaps releases
  without mixing them.
- **Finish background work.** Monitoring embeddings and observer deliveries run after `route()`
  returns. Await `router.flush()` before the invocation ends, or hand it to the platform
  (`ctx.waitUntil(router.flush())` on Workers); a frozen function may otherwise never run them.
- **Memory.** Size the model function for the encoder's model, the decoded kNN reference of any kNN or
  stack head (4 bytes per value: rows × dimensions) and the release documents. kNN and stack heads
  also compare each request with every reference row, so they cost far more per request than linear
  heads; measure both on your release before choosing a memory size.
- **Cold starts.** The rules tier has no model, so it starts fast enough for edge runtimes. The model
  tier's cold start is dominated by loading the encoder's model; keep instances warm (provisioned
  concurrency) where that latency matters.
- **Hosted embeddings.** An encoder adapter can call a hosted embedding API instead of a local model;
  its identity must still match the release exactly (a hosted model can't produce multi-layer features).

## What loading checks

1. The manifest's exact bytes against `expectedManifestSha256`, within size and time limits.
2. Manifest schema, safe relative keys, limits, encoder identity, recorded package versions.
3. Each file's SHA-256 before parsing; each file read once.
4. Classifier and rules with their owning libraries, including finite kNN reference values.
5. The policy: unique routes, exactly the classifier's heads, every route once in priority, every rule
   label a route, and no self-suppression or suppression cycles.
6. Pairing: the rule set's semantic hash against the manifest, the artifact's
   `training.router.ruleSetHash` and dismissal records, and the evidence's input digests. The manifest
   gates must equal the classifier gates combined with the evidence gates.
7. The encoder identity, every field and layer order, and the classifier's recorded embedding spec.
8. In `enforce` mode: classifier gates and complete-release gates passed, and a final-routing report
   with at least one item and no rules-path mismatches, recorded acceptance criteria with a
   performance target, and recorded gates equal to those criteria recomputed on the recorded reports. The report's structure (kind, counts that add
   up, items matching the datasets) is checked in both modes. Shadow mode tolerates failed performance gates but never structural, digest or pairing
   failures.
9. Freezing the parsed snapshot, compiling rules and preparing reference state.

## Decisions

Rules run on the original text. The first firing rule (rule-set order) names a head and beats
dismissal. The request is settled without embedding only when no valid score could change the
outcome; otherwise it is embedded, the vector is validated (row count, width, finite values, declared
normalisation), scored, and every probability the policy needs is checked to lie in [0, 1].
Suppression comes from unsuppressed evidence; the first eligible head in priority fires; review-band
evidence means review; otherwise `onAbstain` applies. `createDecisionEvaluator` exposes the same
evaluator over a threshold/review-floor projection for offline tests (findings that both fire and
dismiss one head can't come from the matcher and are rejected); its settled path is tested
exhaustively against an independently written oracle.

Monitoring embeds a deterministic sample of rules-settled requests, selected by a hash of release
ID, salt and request ID, never by text alone. It runs after the result is returned (bounded by
`monitoring.timeoutMs`, not the request's signal), never changes the decision, and a monitoring failure
is a `monitoring_error` event. Decision events carry `inclusionProbability`, so drift estimates can weight
the biased scored sample. With a sample rate of 0 there is no scored evidence for settled traffic.

## Building and evaluating releases

```ts
import { buildRelease, documentDigest, encodeJson, createDecisionEvaluator } from '@liquidau/router';
import { evaluateFinal, evaluatePerHead, checkAcceptance, buildEvidence, acceptanceExitCode } from '@liquidau/router/evaluation';

const artifact = buildArtifact(result, { version, embedding, router: { ruleSetHash: ruleSetHash(rules) } });
const docs = { classifier: encodeJson(artifact), rules: encodeJson(rules), policy: encodeJson(policy) };
const final = evaluateFinal(evaluator, heldOutItems);          // route/review/abstain after the policy
const perHead = evaluatePerHead(model, heldOutItems);          // each head's own decisions
const criteria = { minCoverage: 0.8, routes: { urgent: { minRecall: 0.95 } } }; // at least one performance target
const evidence = buildEvidence({ inputs: { classifierSha256: await documentDigest(docs.classifier), ... }, dataset, final, perHead, criteria });
const gates = evidence.gates;                                  // checkAcceptance(final, perHead, criteria), recorded with the criteria
const release = await buildRelease({ releaseId, documents: { ...docs, evidence: encodeJson(evidence) }, embedding: identity, serving, packages });
process.exitCode = acceptanceExitCode(gates);                  // non-zero on failure unless diagnostics-only
```

`buildRelease` runs the loader's validation and reports `enforcement` (why the release would be
shadow-only). Claims measured on other data (e.g. per-head guarantees on the calibration split) cite
their own dataset via `otherDatasets`. Every reported number is an empirical point estimate with a Wilson interval, not a
guarantee. Final-route ground truth lists the acceptable routes per item: routing to any one of them is
correct for that item, but per-route recall still counts the item against each acceptable route it
wasn't sent to.

## Document releases

A release whose classifier is a `liquidau-document-classifier/1` envelope (embedding-classifier's
`buildDocumentArtifact`) builds as `liquidau-router/2`, with `liquidau-router-evidence/2` binding the
feature identity and the serving input limit. Load it with the pipeline's exact `tokenizer` (otherwise
`CAPABILITY_MISSING`/`TOKENIZER_MISMATCH`). Rules still run first on the original text; an unresolved
request goes through the classifier's shared `scoreDocument` under ONE `timeoutMs` deadline for planning,
every embedding batch and scoring. New failure codes: `PREPROCESSING_FAILURE`, `INPUT_LIMIT_EXCEEDED`
(both not retryable) and `PROCESSING_TIMEOUT` (retryable). Decision events add chunk counts, tokens,
timings and the feature digest - never text, chunks or spans. Legacy `/1` releases are unchanged and need
no tokenizer.

## Adapters

Source and encoder adapters (filesystem, S3, local ONNX, Bedrock) belong in separate packages that
depend on this one. Never add them to this package, including as peer or optional dependencies.
`encoderContractProblems` and `releaseSourceContractProblems` form the shared contract suite those
adapters run. `memorySource` and `functionEncoder` are dependency-free fixtures.
`embeddingCacheKey` gives a cache key built from the full identity, the preprocessed input and a
tenant scope.

## Checks

```sh
npm test              # unit, equivalence, split-deployment, loading, routing, host-wrapper and packaging tests
npm run typecheck     # including a no-Node-types pass over src
npm run check:portable
```
