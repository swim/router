/**
 * Host independence: the same release bytes and encoder give the same semantic results through a
 * direct call, a Fetch-style HTTP wrapper and a Lambda-style event wrapper, with outbound networking
 * disabled and no credentials. The wrappers are minimal examples of host-owned translation; native
 * request objects never reach the core.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { encoderContractProblems, functionEncoder, loadRouter, memorySource, releaseSourceContractProblems, type Encoder, type Router, type RouteResult } from '../src/index.ts';
import { EVAL, fakeEncoder, IDENTITY, makeRelease, options } from './helpers.ts';

/** Fetch wrapper: parse, size-limit and map results to HTTP; the destination is only acted on when actionable. */
async function fetchHandler(router: Router, request: Request): Promise<Response> {
  const body = (await request.json()) as { text?: unknown; id?: unknown };
  const result = await router.route({ text: body.text as string, requestId: String(body.id ?? request.headers.get('x-request-id') ?? ''), signal: request.signal });
  const status = result.outcome === 'unavailable' ? (result.retryable ? 503 : 400) : 200;
  return new Response(JSON.stringify(result), { status, headers: { 'content-type': 'application/json' } });
}

/** Lambda-style wrapper over an API Gateway-like event. */
async function lambdaHandler(router: Router, event: { body: string; requestContext: { requestId: string } }) {
  const { text } = JSON.parse(event.body) as { text: string };
  const result = await router.route({ text, requestId: event.requestContext.requestId });
  return { statusCode: result.outcome === 'unavailable' ? 503 : 200, body: JSON.stringify(result) };
}

const semantic = (r: RouteResult) => (r.outcome === 'route' ? [r.outcome, r.routeId, r.mechanism] : r.outcome === 'unavailable' ? [r.outcome, r.code] : r.outcome === 'shadow' ? ['shadow', r.candidate] : [r.outcome, r.reason]);

test('direct, HTTP and Lambda wrappers agree, with networking disabled', async (t) => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('outbound networking is disabled in this test'); };
  t.after(() => { globalThis.fetch = realFetch; });

  const built = await makeRelease();
  const router = await loadRouter(options(built, { monitoring: { sampleRate: 1, samplingSalt: 'salt' } }));
  for (const [i, { text }] of [...EVAL, { text: 'x'.repeat(5000) }].entries()) {
    const id = `req-${i}`;
    const direct = await router.route({ text, requestId: id });
    const http = JSON.parse(await (await fetchHandler(router, new Request('http://local/route', { method: 'POST', body: JSON.stringify({ text, id }) }))).text()) as RouteResult;
    const lambda = JSON.parse((await lambdaHandler(router, { body: JSON.stringify({ text }), requestContext: { requestId: id } })).body) as RouteResult;
    assert.deepEqual(semantic(http), semantic(direct), text);
    assert.deepEqual(semantic(lambda), semantic(direct), text);
    assert.equal(http.requestId, id);
    assert.equal(lambda.releaseId, direct.releaseId);
  }
});

test('adapter contract suite: in-memory source and function encoder conform; broken ones are caught', async () => {
  const built = await makeRelease();
  assert.deepEqual(await releaseSourceContractProblems(memorySource(built.files), { present: Object.fromEntries(built.files) }), []);
  const leaky = { read: async () => new Uint8Array([1]) };
  assert.ok((await releaseSourceContractProblems(leaky, { present: {} })).some((p) => p.includes('../escape.json')));

  assert.deepEqual(await encoderContractProblems(fakeEncoder().encoder), []);
  const wrongWidth = functionEncoder(IDENTITY, () => [1, 2, 3]);
  assert.match((await encoderContractProblems(wrongWidth)).join(), /width 3/);
  const ignoresAbort: Encoder = { identity: IDENTITY, embed: async (texts) => texts.map(() => [0, 0, 0, 0]) };
  assert.match((await encoderContractProblems(ignoresAbort)).join(), /aborted signal/);
  let n = 0;
  const unstable = functionEncoder(IDENTITY, () => [n++, 0, 0, 0]);
  assert.match((await encoderContractProblems(unstable)).join(), /different vectors|differs/);
});
