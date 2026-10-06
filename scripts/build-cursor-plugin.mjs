/**
 * build-cursor-plugin.mjs — the Cursor-format files of the plugin tree.
 *
 * Grok Bot installs plugins in Cursor's format. It spawns an MCP server's
 * `command` exactly as written and, on its plugin computer, does not replace
 * `${CLAUDE_PLUGIN_ROOT}`. The Claude Code `.mcp.json` names a script inside the
 * plugin through that variable, so on Grok Bot it can never start.
 *
 * So the Cursor manifest names its own MCP definition, and that definition is a
 * command found on PATH: `uvx`, pinned to this release. No `${...}` appears in
 * any file written here — Cursor treats every `${VAR}` other than its plugin
 * root as a user variable that the listing must declare.
 *
 * Written by build-plugin.mjs as part of the same plan, so `--check` covers
 * these files and the version pin moves with plugin-src/manifest.json.
 *
 * Reference: https://cursor.com/docs/reference/plugins (manifest fields,
 * component discovery, variable expansion, marketplace.json, submission list).
 */

import fs from 'node:fs';
import path from 'node:path';

/** Who is calling, for attribution, and what a shared bot computer needs. */
export const CURSOR_SERVER_ENV = Object.freeze({
  // Attribution: memories written from Grok Bot are told apart from other hosts.
  SLM_AGENT_ID: 'grok_bot',
  // The smallest tool set: a bot host lists every tool to the model.
  SLM_MCP_PROFILE: 'core',
  // Never wait on a prompt (setup_wizard.is_interactive honours it).
  SLM_NON_INTERACTIVE: '1',
  // CPU-only torch from uv: no CUDA wheel stack on a computer with no GPU.
  // The variable rather than `--torch-backend`, so an older uvx that does not
  // know the option ignores it instead of refusing to start.
  UV_TORCH_BACKEND: 'cpu',
  TOKENIZERS_PARALLELISM: 'false',
});

const DESCRIPTION =
  'Local-first long-term memory for your agents and bots: remember, recall and '
  + 'session context stored on your own computer, with reversible compression '
  + 'and a cache for large tool outputs.';

const KEYWORDS = ['memory', 'long-term-memory', 'local-first', 'mcp', 'agents'];
const LOGO_SRC = path.join('assets', 'branding', 'slm-mark.svg');
const LOGO_REL = 'assets/logo.svg';

function sortedJson(obj) {
  const sort = (val) => {
    if (Array.isArray(val)) return val.map(sort);
    if (val !== null && typeof val === 'object') {
      return Object.fromEntries(Object.keys(val).sort().map((k) => [k, sort(val[k])]));
    }
    return val;
  };
  return JSON.stringify(sort(obj), null, 2) + '\n';
}

/** The package spec uvx installs: always this release, never "latest". */
export function packagePin(manifest) {
  return `superlocalmemory==${manifest.version}`;
}

export function renderCursorMcpJson(manifest) {
  return sortedJson({
    mcpServers: {
      superlocalmemory: {
        command: 'uvx',
        args: ['--from', packagePin(manifest), 'slm', 'mcp'],
        env: { ...CURSOR_SERVER_ENV },
      },
    },
  });
}

/** plugin/.cursor-plugin/plugin.json — documented fields only. */
export function renderCursorPluginJson(manifest) {
  return sortedJson({
    author: { name: 'Qualixar' },
    description: DESCRIPTION,
    homepage: manifest.repository,
    // The Claude Code hooks file is not Cursor's format. Declaring hooks
    // replaces folder discovery, so Cursor never tries to run it.
    hooks: { hooks: {} },
    keywords: KEYWORDS,
    license: 'AGPL-3.0-or-later',
    logo: LOGO_REL,
    mcpServers: './mcp.cursor.json',
    name: manifest.pluginName,
    repository: manifest.repository,
    skills: './skills/',
    version: manifest.version,
  });
}

/** Repo-root .cursor-plugin/marketplace.json — the Cursor tree only. */
export function renderCursorMarketplaceJson(manifest) {
  const owner = { name: 'Qualixar' };
  if (manifest.marketplace && manifest.marketplace.ownerEmail) {
    owner.email = manifest.marketplace.ownerEmail;
  }
  return sortedJson({
    metadata: { description: 'SuperLocalMemory: local-first long-term memory for agents and bots.' },
    name: 'qualixar',
    owner,
    plugins: [
      {
        category: 'productivity',
        description: DESCRIPTION,
        homepage: manifest.repository,
        keywords: KEYWORDS,
        license: 'AGPL-3.0-or-later',
        name: manifest.pluginName,
        repository: manifest.repository,
        source: './plugin',
        version: manifest.version,
      },
    ],
  });
}

/**
 * Plan entries (absolute path -> content) for the Cursor-format files.
 * @param {string} root repository root
 * @param {object} manifest parsed plugin-src/manifest.json
 * @param {string} pluginRoot absolute plugin/ directory
 * @returns {Map<string, string>}
 */
export function cursorPlan(root, manifest, pluginRoot) {
  const logoPath = path.join(root, LOGO_SRC);
  let logo;
  try {
    logo = fs.readFileSync(logoPath, 'utf8');
  } catch (err) {
    throw new Error(`cursorPlan: cannot read logo ${logoPath}: ${err.message}`);
  }
  return new Map([
    [path.join(pluginRoot, '.cursor-plugin', 'plugin.json'), renderCursorPluginJson(manifest)],
    [path.join(pluginRoot, 'mcp.cursor.json'), renderCursorMcpJson(manifest)],
    [path.join(pluginRoot, ...LOGO_REL.split('/')), logo],
    [path.join(root, '.cursor-plugin', 'marketplace.json'), renderCursorMarketplaceJson(manifest)],
  ]);
}
