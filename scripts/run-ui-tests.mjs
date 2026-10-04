/**
 * Run the UI test suite.
 *
 * WHY THIS SCRIPT EXISTS, rather than `node --test tests/ui/`:
 *
 * Node 26 resolves a bare directory argument as a MODULE, not as a directory to
 * scan. `node --test tests/ui/` therefore dies with
 *
 *     Error: Cannot find module '.../tests/ui'   (MODULE_NOT_FOUND)
 *
 * in ~48 ms, reporting one failed "test" that names nothing and points at no
 * defect — while every file in the suite passes. Enumerating the files here and
 * passing them individually works on every Node version and cannot produce that
 * misleading result.
 *
 * Guarded by tests/ui/test_runner_invocation_is_documented_correctly.mjs.
 */

import { readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const testDir = join(root, 'tests', 'ui');
const files = readdirSync(testDir)
    .filter(name => /^test_.*\.mjs$/.test(name))
    .sort()
    .map(name => relative(root, join(testDir, name)));

if (files.length === 0) {
    throw new Error('No UI tests found under tests/ui');
}

// The build-script tests (scripts/__tests__/*.test.mjs) ran nowhere, so four of
// them sat red through three intended plugin changes (4.1.20 WP11). Run them
// with the UI suite so `npm test` covers every Node test in the repo.
const scriptTestDir = join(root, 'scripts', '__tests__');
const scriptTests = readdirSync(scriptTestDir)
    .filter(name => /\.test\.mjs$/.test(name))
    .sort()
    .map(name => relative(root, join(scriptTestDir, name)));
if (scriptTests.length === 0) {
    throw new Error('No build-script tests found under scripts/__tests__');
}
files.push(...scriptTests);

const result = spawnSync(process.execPath, ['--test', ...files], {
    cwd: root,
    stdio: 'inherit',
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
