// Development only: until @liquidau/embedding-classifier 0.8.0 and @liquidau/rule-miner 0.5.1 are
// published, point node_modules at the sibling checkouts. Never commit a lockfile produced this way.
import { mkdirSync, rmSync, symlinkSync } from 'node:fs';
import { join } from 'node:path';

const root = join(import.meta.dirname, '..');
mkdirSync(join(root, 'node_modules', '@liquidau'), { recursive: true });
for (const name of ['embedding-classifier', 'rule-miner']) {
  const link = join(root, 'node_modules', '@liquidau', name);
  rmSync(link, { recursive: true, force: true });
  symlinkSync(join('..', '..', '..', name), link, 'dir');
  console.log(`linked @liquidau/${name} -> ../${name}`);
}
