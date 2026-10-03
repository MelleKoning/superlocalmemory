#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * docs/hermes.md and README.md's Hermes section quote the pinned GitHub
 * release in prose and were hand-kept, so they stayed at 4.1.13 for
 * several releases after the real one moved on (L3-08). They are now
 * stamped from pyproject.toml by scripts/build-hermes-plugin.mjs, the same
 * way plugin.yaml and __init__.py already are.
 *
 * Run with: node --test tests/test_scripts/test_build_hermes_plugin_stamps_doc_version.js
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SCRIPT = path.join(REPO_ROOT, 'scripts', 'build-hermes-plugin.mjs');
const PYPROJECT = path.join(REPO_ROOT, 'pyproject.toml');
const DOCS_HERMES = path.join(REPO_ROOT, 'docs', 'hermes.md');
const README = path.join(REPO_ROOT, 'README.md');

function readVersion() {
  const match = fs.readFileSync(PYPROJECT, 'utf8').match(/^version = "(\d+\.\d+\.\d+)"$/m);
  assert.ok(match, 'pyproject.toml has no x.y.z version');
  return match[1];
}

test('build-hermes-plugin --check passes against the committed tree (docs already stamped)', () => {
  // If this fails, docs/hermes.md or README.md's Hermes section drifted
  // from pyproject.toml's version and nobody ran the build.
  execFileSync('node', [SCRIPT, '--check'], { cwd: REPO_ROOT });
});

test('docs/hermes.md states the current pyproject version, not a stale pin', () => {
  const version = readVersion();
  const text = fs.readFileSync(DOCS_HERMES, 'utf8');
  assert.match(text, new RegExp(`SuperLocalMemory ${version.replace(/\./g, '\\.')} ships`));
  assert.match(text, new RegExp(`superlocalmemory==${version.replace(/\./g, '\\.')}`));
  assert.match(text, new RegExp('`v' + version.replace(/\./g, '\\.') + '`'));
  assert.match(text, new RegExp(`download/v${version.replace(/\./g, '\\.')}/`));
});

test("README.md's Hermes section states the current pyproject version", () => {
  const version = readVersion();
  const text = fs.readFileSync(README, 'utf8');
  assert.match(text, new RegExp('reviewed pinned pack from the `v' + version.replace(/\./g, '\\.') + '` GitHub release'));
});
