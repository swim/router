// Bundles every core export for a platform-neutral runtime and fails on any Node built-in, cloud SDK,
// inference engine or hosting framework in the bundle's inputs. Used by `npm run check:portable` and
// the packaging test.
import { build } from 'esbuild';
import { join } from 'node:path';

const root = join(import.meta.dirname, '..');
const ENTRIES = { index: 'src/index.ts', evaluation: 'src/evaluation/index.ts' };
const FORBIDDEN = [/^node:/, /@aws-sdk\//, /aws-sdk/, /onnxruntime/, /@huggingface\//, /express/, /aws-lambda/, /@types\/node/];
const NODE_BUILTINS = ['fs', 'path', 'crypto', 'os', 'child_process', 'http', 'https', 'net', 'stream', 'url', 'buffer', 'process', 'worker_threads', 'zlib'];

export async function checkPortable() {
  const problems = [];
  for (const [name, entry] of Object.entries(ENTRIES)) {
    const result = await build({
      entryPoints: [join(root, entry)], bundle: true, write: false, metafile: true, platform: 'neutral', format: 'esm', mainFields: ['module', 'main'],
      logLevel: 'silent',
    }).catch((e) => ({ errors: e.errors ?? [{ text: String(e) }] }));
    if (result.errors?.length) { problems.push(...result.errors.map((e) => `${name}: ${e.text}`)); continue; }
    for (const input of Object.keys(result.metafile.inputs)) {
      for (const imp of result.metafile.inputs[input].imports) {
        if (FORBIDDEN.some((f) => f.test(imp.path)) || NODE_BUILTINS.includes(imp.path)) problems.push(`${name}: ${input} imports ${imp.path}`);
      }
    }
    if (result.outputFiles.some((f) => /\bprocess\.env\b|\brequire\(/.test(f.text))) problems.push(`${name}: bundle reads process.env or uses require()`);
  }
  return problems;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const problems = await checkPortable();
  if (problems.length) { console.error(problems.join('\n')); process.exit(1); }
  console.log('core exports bundle for a neutral platform with no Node built-ins, cloud SDKs or inference engines');
}
