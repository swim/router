import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { COMPATIBLE_PACKAGES } from '../src/index.ts';
// @ts-expect-error - a plain .mjs script without types
import { checkPortable } from '../scripts/check-portable.mjs';

const root = join(import.meta.dirname, '..');
const pkg = (dir: string) => JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')) as { name: string; version: string; dependencies?: Record<string, string>; peerDependencies?: Record<string, string>; optionalDependencies?: Record<string, string> };

test('every core export bundles for a neutral platform without Node, cloud or inference imports', async () => {
  assert.deepEqual(await checkPortable(), []);
});

test('the runtime dependency graph is only the Liquidau libraries; no optional or peer provider SDKs', () => {
  const allowed = new Set(['@liquidau/embedding-classifier', '@liquidau/rule-miner', '@liquidau/solvers', '@liquidau/text-preprocessing']);
  const seen = new Set<string>();
  const walk = (dir: string) => {
    const p = pkg(dir);
    assert.equal(p.peerDependencies, undefined, `${p.name} declares peer dependencies`);
    assert.equal(p.optionalDependencies, undefined, `${p.name} declares optional dependencies`);
    for (const name of Object.keys(p.dependencies ?? {})) {
      assert.ok(allowed.has(name), `${p.name} depends on ${name}`);
      if (seen.has(name)) continue;
      seen.add(name);
      walk(resolveDir(dir, name));
    }
  };
  walk(root);
  assert.ok(seen.has('@liquidau/solvers'));
});

function resolveDir(from: string, name: string): string {
  for (let dir = from; ; dir = join(dir, '..')) {
    try {
      readFileSync(join(dir, 'node_modules', name, 'package.json'));
      return join(dir, 'node_modules', name);
    } catch {
      if (dir === '/') throw new Error(`cannot resolve ${name} from ${from}`);
    }
  }
}

test('COMPATIBLE_PACKAGES matches package.json ranges and the router version', () => {
  const p = pkg(root);
  assert.equal(COMPATIBLE_PACKAGES['@liquidau/router'], p.version.split('.').slice(0, 2).join('.'));
  for (const name of ['@liquidau/embedding-classifier', '@liquidau/rule-miner', '@liquidau/text-preprocessing']) {
    const range = p.dependencies![name];
    assert.match(range, /^\^0\.\d+\.\d+$/);
    assert.equal(COMPATIBLE_PACKAGES[name], range.slice(1).split('.').slice(0, 2).join('.'));
  }
});
