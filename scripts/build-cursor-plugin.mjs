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
  // Attribution. Grok Bot and the Cursor editor both read this manifest and
  // the definition cannot tell them apart, so it names the plugin format.
  SLM_AGENT_ID: 'cursor_plugin',
  // The 18-tool core set: Grok Bot lists every tool to the model on a shared,
  // memory-tight computer. The one documented exception to "a plugin never
  // narrows the tool set" (tests/test_packaging/test_no_plugin_hijacks_your_install.py).
  SLM_MCP_PROFILE: 'core',
  // Never wait on a prompt (setup_wizard.is_interactive honours it).
  SLM_NON_INTERACTIVE: '1',
  // CPU-only torch from uv: no CUDA wheel stack on a computer with no GPU.
  // The variable rather than `--torch-backend`, so an older uvx that does not
  // know the option ignores it instead of refusing to start.
  UV_TORCH_BACKEND: 'cpu',
  TOKENIZERS_PARALLELISM: 'false',
  // GB5 lite bot-host profile: the box is shared by every bot on it with
  // 1.8-3.5 GiB free RAM, and the cross-encoder reranker subprocess alone
  // measured ~200 MB resident once warm (plus its own PyTorch import). Turn
  // it off here — opt-in via env, nowhere else — so recall still works (BM25
  // + semantic + the other fusion channels), just without cross-encoder
  // re-ordering of the fused results. Quality cost: reranking measurably
  // improves top-of-list precision in SLM's own benchmarks (see
  // bench-v342-locomo.md referenced in core/config.py); turning it off trades
  // that precision for RAM headroom on a host where OOM would lose the
  // session entirely. Set SLM_RERANKER_ENABLED=true to opt back in once RAM
  // allows. SLM_RERANKER_IDLE_TIMEOUT is set too so a future re-enable (or a
  // host that flips this at the MCP env level) recycles the worker quickly
  // instead of holding it warm for the default 30 minutes.
  SLM_RERANKER_ENABLED: 'false',
  SLM_RERANKER_IDLE_TIMEOUT: '120',
  // Explicit, not just relying on the default: one embedding worker, never a
  // pool, on a computer this memory-tight.
  SLM_MAX_EMBEDDING_WORKERS: '1',
});

const DESCRIPTION =
  'Local-first long-term memory for your agents and bots: remember, recall and '
  + 'session context stored on your own computer, with reversible compression '
  + 'and a cache for large tool outputs.';

/**
 * GB7 — the 4-6 skills visible to Grok Bot / Cursor, independent of the full
 * ~12-skill set every other host gets (plugin/skills/). Chosen to match what
 * this manifest's `SLM_MCP_PROFILE=core` (18 tools) can actually do, plus the
 * two skills written specifically for a headless, hook-less bot host:
 *   - slm-getting-started-bot / slm-bot-memory: new, bot-host-specific.
 *   - slm-remember / slm-recall: the two tools every session uses.
 *   - slm-session: what session_init/close_session do (no hooks run them here).
 *   - slm-scope: the personal/shared/global model slm-bot-memory namespaces on.
 * Deliberately excluded: slm-graph/slm-mesh/slm-governance/slm-loop (their
 * tools are not in the core profile this manifest sets — listing a skill for
 * a tool that is not installed would just teach the model to call something
 * that does not exist) and slm-cache/slm-compress/slm-profile (useful, but
 * over the 4-6 budget; still shipped to every other host).
 */
export const CURSOR_SKILLS = Object.freeze([
  'slm-getting-started-bot',
  'slm-bot-memory',
  'slm-remember',
  'slm-recall',
  'slm-session',
  'slm-scope',
]);

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
    // GB7: a curated 4-6, not the full set every other host gets — see
    // CURSOR_SKILLS. A manifest field replaces folder discovery, so this
    // points Cursor at cursor-skills/ instead of the full skills/ directory.
    skills: './cursor-skills/',
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
 * @param {Map<string, string>} [fullSkillsPlan] the already-rendered
 *   plugin/skills/<name>/SKILL.md entries build-plugin.mjs's buildPlan() has
 *   by the time it calls this (skills are planned before cursorPlan runs).
 *   Reusing those entries means the Cursor copy gets the exact same
 *   version-stamped, attribution-normalized content as every other host,
 *   through one rendering path instead of two that could drift apart.
 *   Defaults to an empty Map so existing callers that don't need the
 *   cursor-skills/ subset (tests of the other three files) still work; a
 *   real build always passes the real plan.
 * @returns {Map<string, string>}
 */
export function cursorPlan(root, manifest, pluginRoot, fullSkillsPlan = new Map()) {
  const logoPath = path.join(root, LOGO_SRC);
  let logo;
  try {
    logo = fs.readFileSync(logoPath, 'utf8');
  } catch (err) {
    throw new Error(`cursorPlan: cannot read logo ${logoPath}: ${err.message}`);
  }
  const plan = new Map([
    [path.join(pluginRoot, '.cursor-plugin', 'plugin.json'), renderCursorPluginJson(manifest)],
    [path.join(pluginRoot, 'mcp.cursor.json'), renderCursorMcpJson(manifest)],
    [path.join(pluginRoot, ...LOGO_REL.split('/')), logo],
    [path.join(root, '.cursor-plugin', 'marketplace.json'), renderCursorMarketplaceJson(manifest)],
  ]);
  for (const name of CURSOR_SKILLS) {
    const srcKey = path.join(pluginRoot, 'skills', name, 'SKILL.md');
    const content = fullSkillsPlan.get(srcKey);
    if (content === undefined) {
      throw new Error(
        `cursorPlan: CURSOR_SKILLS names '${name}', but ${srcKey} is not in the rendered `
        + 'skills plan (check plugin-src/manifest.json lists it, and plugin-src/skills/'
        + `${name}/SKILL.md exists).`,
      );
    }
    plan.set(path.join(pluginRoot, 'cursor-skills', name, 'SKILL.md'), content);
  }
  return plan;
}
