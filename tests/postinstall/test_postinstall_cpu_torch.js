#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * GB4 — scripts/postinstall.js must not let the npm installer pull PyPI's
 * CUDA-tagged `torch` wheel (torch + triton + ~18 nvidia-* packages, ~2.6
 * GiB) onto a CPU-only Linux box. These tests exercise the pure decision
 * functions only — no pip, no network, no real install.
 *
 * Run with: node --test tests/postinstall/test_postinstall_cpu_torch.js
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
// shouldForceCpuTorch
// ---------------------------------------------------------------------------

test('non-Linux never forces the CPU index', () => {
  assert.equal(mod.shouldForceCpuTorch({}, 'darwin', () => true), false);
  assert.equal(mod.shouldForceCpuTorch({}, 'darwin', () => false), false);
  assert.equal(mod.shouldForceCpuTorch({}, 'win32', () => false), false);
});

test('Linux with no GPU and no override forces the CPU index', () => {
  assert.equal(mod.shouldForceCpuTorch({}, 'linux', () => false), true);
});

test('Linux with a visible GPU does not force the CPU index', () => {
  assert.equal(mod.shouldForceCpuTorch({}, 'linux', () => true), false);
});

test('SLM_TORCH_BACKEND=cpu overrides GPU detection', () => {
  assert.equal(
    mod.shouldForceCpuTorch({ SLM_TORCH_BACKEND: 'cpu' }, 'linux', () => true),
    true,
  );
});

test('SLM_TORCH_BACKEND=cuda opts out even with no GPU detected', () => {
  assert.equal(
    mod.shouldForceCpuTorch({ SLM_TORCH_BACKEND: 'cuda' }, 'linux', () => false),
    false,
  );
});

test('an already-set PIP_INDEX_URL is never overridden', () => {
  assert.equal(
    mod.shouldForceCpuTorch({ PIP_INDEX_URL: 'https://example.invalid/simple' }, 'linux', () => false),
    false,
  );
});

test('an already-set PIP_EXTRA_INDEX_URL is never overridden', () => {
  assert.equal(
    mod.shouldForceCpuTorch({ PIP_EXTRA_INDEX_URL: 'https://example.invalid/simple' }, 'linux', () => false),
    false,
  );
});

test('the documented CPU wheel index is exactly the pytorch.org one', () => {
  // https://pytorch.org/get-started/locally/ (fetched 2026-10-06).
  assert.equal(mod.TORCH_CPU_INDEX_URL, 'https://download.pytorch.org/whl/cpu');
});

// ---------------------------------------------------------------------------
// hasNvidiaGpu — real function, but only the "definitely false" branches are
// safe to assert on a CI/dev box that may or may not have the fake files.
// ---------------------------------------------------------------------------

test('hasNvidiaGpu is always false off Linux', () => {
  assert.equal(mod.hasNvidiaGpu('darwin'), false);
  assert.equal(mod.hasNvidiaGpu('win32'), false);
});

// ---------------------------------------------------------------------------
// cpuTorchPin
// ---------------------------------------------------------------------------

function withTempPackageRoot(fn) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-cpu-torch-pin-'));
  try {
    fn(root);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test('cpuTorchPin reads the pin line out of plugin/requirements-cpu-torch.txt', () => {
  withTempPackageRoot((root) => {
    const pluginDir = path.join(root, 'plugin');
    fs.mkdirSync(pluginDir, { recursive: true });
    fs.writeFileSync(
      path.join(pluginDir, 'requirements-cpu-torch.txt'),
      '# comment\ntorch==2.13.0\n',
    );
    assert.equal(mod.cpuTorchPin(root), 'torch==2.13.0');
  });
});

test('cpuTorchPin fails open (returns null) when the file is missing', () => {
  withTempPackageRoot((root) => {
    assert.equal(mod.cpuTorchPin(root), null);
  });
});

test('cpuTorchPin fails open when the file has no torch== line', () => {
  withTempPackageRoot((root) => {
    const pluginDir = path.join(root, 'plugin');
    fs.mkdirSync(pluginDir, { recursive: true });
    fs.writeFileSync(path.join(pluginDir, 'requirements-cpu-torch.txt'), '# nothing here\n');
    assert.equal(mod.cpuTorchPin(root), null);
  });
});

test('the real shipped pin file parses', () => {
  const pin = mod.cpuTorchPin(REPO_ROOT);
  assert.ok(pin, 'expected plugin/requirements-cpu-torch.txt to exist and parse once built');
  assert.match(pin, /^torch==\d+\.\d+\.\d+$/);
});
