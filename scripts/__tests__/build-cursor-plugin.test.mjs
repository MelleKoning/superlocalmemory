/**
 * build-cursor-plugin.test.mjs — the Cursor-format files (Grok Bot).
 *
 * Grok Bot spawns an MCP command verbatim and never replaces ${...} on its
 * plugin computer, so the Cursor definition must be a PATH command, pinned to
 * the release being built, with no placeholder anywhere.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  CURSOR_SERVER_ENV,
  cursorPlan,
  packagePin,
  renderCursorMarketplaceJson,
  renderCursorMcpJson,
  renderCursorPluginJson,
} from '../build-cursor-plugin.mjs';

const MANIFEST = {
  version: '9.8.7',
  pluginName: 'superlocalmemory',
  repository: 'https://github.com/qualixar/superlocalmemory',
  marketplace: { ownerEmail: 'owner@example.com', description: 'x' },
};

// Fields https://cursor.com/docs/reference/plugins documents for plugin.json.
const DOCUMENTED_PLUGIN_FIELDS = new Set([
  'name', 'description', 'version', 'author', 'homepage', 'repository', 'license',
  'keywords', 'logo', 'rules', 'agents', 'skills', 'commands', 'hooks', 'mcpServers',
  'variables',
]);

function allStrings(value, out = []) {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) value.forEach((v) => allStrings(v, out));
  else if (value && typeof value === 'object') Object.values(value).forEach((v) => allStrings(v, out));
  return out;
}

describe('Cursor MCP definition', () => {
  const server = JSON.parse(renderCursorMcpJson(MANIFEST)).mcpServers.superlocalmemory;

  test('is a PATH command pinned to the release being built', () => {
    assert.equal(server.command, 'uvx');
    assert.deepEqual(server.args, ['--from', 'superlocalmemory==9.8.7', 'slm', 'mcp']);
    assert.equal(packagePin(MANIFEST), 'superlocalmemory==9.8.7');
  });

  test('contains no ${...} placeholder in command, args or env', () => {
    for (const s of allStrings(server)) assert.ok(!s.includes('${'), s);
  });

  test('names the plugin format, the core profile, non-interactive and CPU torch', () => {
    assert.deepEqual(server.env, { ...CURSOR_SERVER_ENV });
    assert.equal(server.env.SLM_AGENT_ID, 'cursor_plugin');
    assert.equal(server.env.SLM_MCP_PROFILE, 'core');
    assert.equal(server.env.SLM_NON_INTERACTIVE, '1');
    assert.equal(server.env.UV_TORCH_BACKEND, 'cpu');
    assert.ok(!('SLM_DATA_DIR' in server.env), 'never re-point the store');
  });

  test('GB5: opts into the lite bot-host profile (reranker off, short idle, one embedding worker)', () => {
    assert.equal(server.env.SLM_RERANKER_ENABLED, 'false');
    assert.equal(server.env.SLM_RERANKER_IDLE_TIMEOUT, '120');
    assert.equal(server.env.SLM_MAX_EMBEDDING_WORKERS, '1');
  });
});

describe('Cursor plugin.json', () => {
  const pj = JSON.parse(renderCursorPluginJson(MANIFEST));

  test('uses only documented fields, a kebab-case name and the release version', () => {
    for (const k of Object.keys(pj)) assert.ok(DOCUMENTED_PLUGIN_FIELDS.has(k), `undocumented field ${k}`);
    assert.match(pj.name, /^[a-z0-9][a-z0-9.-]*$/);
    assert.equal(pj.version, '9.8.7');
  });

  test('every path is relative and stays inside the plugin', () => {
    for (const p of [pj.logo, pj.skills, pj.mcpServers]) {
      assert.ok(!path.isAbsolute(p) && !p.split('/').includes('..'), p);
    }
    assert.equal(pj.mcpServers, './mcp.cursor.json');
  });

  test('declares hooks inline-empty so the Claude Code hooks file is never run', () => {
    assert.deepEqual(pj.hooks, { hooks: {} });
  });

  test('contains no ${...} placeholder', () => {
    for (const s of allStrings(pj)) assert.ok(!s.includes('${'), s);
  });
});

describe('Cursor marketplace.json', () => {
  test('lists only the Cursor-ready tree, from ./plugin', () => {
    const mk = JSON.parse(renderCursorMarketplaceJson(MANIFEST));
    assert.equal(mk.plugins.length, 1);
    assert.equal(mk.plugins[0].name, 'superlocalmemory');
    assert.equal(mk.plugins[0].source, './plugin');
    assert.equal(mk.plugins[0].version, '9.8.7');
    assert.equal(mk.owner.name, 'Qualixar');
  });
});

describe('cursorPlan', () => {
  test('writes four files and fails loudly without the logo source', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-cursor-'));
    const pluginRoot = path.join(tmp, 'plugin');
    assert.throws(() => cursorPlan(tmp, MANIFEST, pluginRoot), /logo/);
    fs.mkdirSync(path.join(tmp, 'assets', 'branding'), { recursive: true });
    fs.writeFileSync(path.join(tmp, 'assets', 'branding', 'slm-mark.svg'), '<svg/>\n');
    const plan = cursorPlan(tmp, MANIFEST, pluginRoot);
    assert.deepEqual([...plan.keys()].sort(), [
      path.join(tmp, '.cursor-plugin', 'marketplace.json'),
      path.join(pluginRoot, '.cursor-plugin', 'plugin.json'),
      path.join(pluginRoot, 'assets', 'logo.svg'),
      path.join(pluginRoot, 'mcp.cursor.json'),
    ].sort());
    assert.equal(plan.get(path.join(pluginRoot, 'assets', 'logo.svg')), '<svg/>\n');
  });
});
