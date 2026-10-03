#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * scripts/postinstall.js never tried `python3.14`, `python3.13` or
 * `python3.12` — only bare `python3`/`python` and a few fixed absolute
 * paths. On Debian 12, Ubuntu 22.04 with deadsnakes, and RHEL 9, the
 * system `python3` is older than 3.12 and the versioned interpreter sits
 * alongside it unused, so the install failed even though a supported
 * interpreter was present. `SLM_PYTHON`, documented in
 * docs/getting-started.md as the escape hatch for exactly this case, was
 * read by nothing.
 *
 * Run with: node --test tests/postinstall/test_postinstall_python_discovery.js
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SCRIPT = path.join(REPO_ROOT, 'scripts', 'postinstall.js');
const mod = require(SCRIPT);

// ---------------------------------------------------------------------------
// Candidate order — python3.14/3.13/3.12 must be tried before bare
// python3/python on every non-Windows platform.
// ---------------------------------------------------------------------------

test('Linux: versioned 3.14/3.13/3.12 interpreters come before bare python3/python', () => {
  const names = mod.pythonCandidates('linux').map((c) => c[0]);
  assert.deepEqual(
    names.slice(0, 5),
    ['python3.14', 'python3.13', 'python3.12', 'python3', 'python'],
  );
});

test('macOS: versioned 3.14/3.13/3.12 interpreters come before bare python3/python', () => {
  const names = mod.pythonCandidates('darwin').map((c) => c[0]);
  assert.deepEqual(
    names.slice(0, 5),
    ['python3.14', 'python3.13', 'python3.12', 'python3', 'python'],
  );
});

test('Linux/macOS: the fixed absolute-path fallbacks are still tried after the versioned names', () => {
  const names = mod.pythonCandidates('linux').map((c) => c[0]);
  assert.ok(names.includes('/opt/homebrew/bin/python3'));
  assert.ok(names.includes('/usr/local/bin/python3'));
  assert.ok(names.includes('/usr/bin/python3'));
  assert.ok(names.indexOf('/usr/bin/python3') > names.indexOf('python3.12'));
});

test('Windows candidate order is unchanged by this fix', () => {
  const names = mod.pythonCandidates('win32').map((c) => c.join(' '));
  assert.deepEqual(names, ['py -3.14', 'py -3.13', 'py -3.12', 'python3', 'python']);
});

// ---------------------------------------------------------------------------
// SLM_PYTHON override — honoured, but still version-checked.
// ---------------------------------------------------------------------------

function withEnv(vars, fn) {
  const prior = {};
  for (const key of Object.keys(vars)) prior[key] = process.env[key];
  Object.assign(process.env, vars);
  try {
    return fn();
  } finally {
    for (const key of Object.keys(vars)) {
      if (prior[key] === undefined) delete process.env[key];
      else process.env[key] = prior[key];
    }
  }
}

function withCapturedStderr(fn) {
  const lines = [];
  const original = console.error;
  console.error = (...args) => { lines.push(args.join(' ')); };
  try {
    const result = fn();
    return { result, stderr: lines.join('\n') };
  } finally {
    console.error = original;
  }
}

// A real interpreter the test host is guaranteed to have: the one running
// this very test's parent toolchain (resolved from PYTHON_FOR_TESTS, a
// supported-version interpreter path set by the harness) falls back to the
// first candidate that resolves on this machine when unset.
function firstWorkingCandidate() {
  const { spawnSync } = require('node:child_process');
  for (const candidate of mod.pythonCandidates()) {
    try {
      const r = spawnSync(candidate[0], [...candidate.slice(1), '--version'], { stdio: 'pipe', timeout: 5000 });
      const out = `${(r.stdout || '').toString()} ${(r.stderr || '').toString()}`;
      const version = mod.parsePythonVersion(out);
      if (r.status === 0 && mod.isSupportedPython(version)) return candidate[0];
    } catch {
      // keep looking
    }
  }
  return null;
}

test('SLM_PYTHON, when it points at a supported interpreter, is used instead of the candidate search', () => {
  const override = firstWorkingCandidate();
  if (!override) {
    // No supported Python on this CI host at all — nothing to override with.
    return;
  }
  withEnv({ SLM_PYTHON: override }, () => {
    const python = mod.findSupportedPython();
    assert.ok(python, 'SLM_PYTHON pointed at a supported interpreter but none was returned');
    assert.equal(python.command, override);
  });
});

test('SLM_PYTHON, when the interpreter it names is too old, fails loudly instead of silently falling back', () => {
  const tooOld = '/usr/bin/python3'; // macOS/CI system Python — historically 3.9.x
  if (!fs.existsSync(tooOld)) return; // not every host has this path
  const { spawnSync } = require('node:child_process');
  const probe = spawnSync(tooOld, ['--version'], { stdio: 'pipe', timeout: 5000 });
  const version = mod.parsePythonVersion(`${probe.stdout || ''} ${probe.stderr || ''}`);
  if (mod.isSupportedPython(version)) return; // this host's system python is new enough; skip

  withEnv({ SLM_PYTHON: tooOld }, () => {
    const { result: python, stderr } = withCapturedStderr(() => mod.findSupportedPython());
    assert.equal(python, null, 'an unsupported SLM_PYTHON must not be accepted');
    assert.match(stderr, /SLM_PYTHON/);
    assert.match(stderr, new RegExp(tooOld.replace(/[/.]/g, '\\$&')));
  });
});

test('SLM_PYTHON pointing at a non-existent path fails loudly, not with a crash', () => {
  const missing = path.join(os.tmpdir(), 'slm-test-no-such-python-binary-xyz');
  withEnv({ SLM_PYTHON: missing }, () => {
    const { result: python, stderr } = withCapturedStderr(() => mod.findSupportedPython());
    assert.equal(python, null);
    assert.match(stderr, /SLM_PYTHON/);
  });
});
