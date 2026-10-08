/**
 * tests/ui/test_web_access_status.mjs — the owner is never surprised by Web access stopping.
 *
 * A completed connection reports access_state (renews_automatically, ending_soon,
 * ended, sign_in_required) and access_expires_at_ms. The "Web access" card turns
 * that into one calm, plain sentence and never shows internal names, tokens or paths.
 *
 * Runner: node scripts/run-ui-tests.mjs
 */

import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const source = (name) => readFileSync(join(UI, 'js', name), 'utf8');
const SCRIPTS = ['od-apps-ui.js', 'od-apps-list.js', 'od-connections.js', 'od-apps.js'];

const CID = 'c'.repeat(32);
const DAY = 86400e3;
const MCP_URL = 'https://mcp.superlocalmemory.com/mcp';

const settle = async (window, ticks = 12) => {
  for (let i = 0; i < ticks; i += 1) await new Promise((r) => window.setTimeout(r, 0));
};

function connection(extra) {
  return Object.assign({
    connection_id: CID, host: 'composio', state: 'ready_for_client', version: 4, verified: true,
    mcp_url: MCP_URL, intent_key: 'b'.repeat(32), cleanup_pending: false,
    access_state: 'renews_automatically', access_expires_at_ms: Date.now() + 29 * DAY,
  }, extra || {});
}

const open = [];
afterEach(() => { while (open.length) open.pop().window.close(); });

async function mount(initial) {
  const calls = [];
  const state = { connections: initial };
  const dom = new JSDOM('<!doctype html><html><body><div id="apps-pane"></div></body></html>', {
    runScripts: 'dangerously', url: 'http://localhost:8765/',
  });
  const { window } = dom;
  open.push(dom);
  await new Promise((resolve) => (window.document.readyState === 'complete' ? resolve() : window.addEventListener('load', resolve)));
  window.fetch = (path, init) => {
    calls.push({ path, method: (init && init.method) || 'GET' });
    let body = {};
    if (/\/status$/.test(path)) {
      body = { available: true, installation_id: 'inst-1', current_profile: 'default',
        hosts: ['muse', 'chatgpt', 'composio'], connections: state.connections };
    } else if (/\/apps$/.test(path)) body = { connection_id: CID, apps: [] };
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
  };
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  for (const name of SCRIPTS) {
    const el = window.document.createElement('script');
    el.textContent = source(name);
    window.document.head.appendChild(el);
  }
  const card = window.odCreateAiConnectionsCard();
  window.document.getElementById('apps-pane').appendChild(card);
  await settle(window);
  const line = card.querySelector('[data-access-line]');
  const pill = card.querySelector('[data-computer-status]');
  const refresh = async (next) => {
    state.connections = next;
    Array.from(card.querySelectorAll('button')).find((b) => b.textContent === 'Refresh status').click();
    await settle(window);
  };
  return { dom, window, card, calls, line, pill, refresh };
}

const LEAKS = /renews_automatically|ending_soon|sign_in_required|authorization_required|access_state|access_expires|\btoken\b|credential|\/Users\/|\.json|keyring|undefined|NaN|null/i;
const buttons = (card) => Array.from(card.querySelectorAll('button, a')).map((n) => n.textContent);
const longDate = (ms) => new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });

describe('Web access status: the four honest states', () => {
  it('renews_automatically: one quiet line, no warning tone', async () => {
    const { card, line, pill } = await mount([connection()]);
    assert.ok(line, 'access line exists');
    assert.equal(line.hidden, false);
    assert.equal(line.textContent, 'Renews automatically.');
    assert.ok(!line.classList.contains('is-warn'));
    assert.equal(pill.textContent, 'Ready');
    assert.doesNotMatch(card.textContent, LEAKS);
  });

  it('ending_soon: names the date, says it normally renews, asks to check the computer is online', async () => {
    const ends = Date.now() + 3 * DAY;
    const { card, line, pill } = await mount([connection({ access_state: 'ending_soon', access_expires_at_ms: ends })]);
    assert.equal(line.textContent,
      `Web access ends on ${longDate(ends)} unless this computer reconnects. It normally renews by itself; check that this computer is online.`);
    assert.ok(line.classList.contains('is-warn'));
    assert.equal(pill.textContent, 'Ending soon');
    assert.ok(pill.classList.contains('is-warn'));
    assert.doesNotMatch(card.textContent, LEAKS);
  });

  it('ended: says so and tells the owner how to carry on', async () => {
    const { card, line, pill } = await mount([connection({ access_state: 'ended', access_expires_at_ms: Date.now() - DAY })]);
    assert.equal(line.textContent, 'Web access has ended. Turn it on again to reconnect your apps.');
    assert.ok(line.classList.contains('is-warn'));
    assert.equal(pill.textContent, 'Needs attention');
    assert.doesNotMatch(card.textContent, /Apps you approve can reach your memory/);
    assert.doesNotMatch(card.textContent, LEAKS);
  });

  it('sign_in_required: asks for a new sign-in and reuses the existing Restart sign-in action', async () => {
    const { card, line, pill } = await mount([connection({
      state: 'pending', verified: false, mcp_url: undefined, transport_state: 'authorization_required',
      access_state: 'sign_in_required', access_expires_at_ms: Date.now() + 20 * DAY,
    })]);
    assert.equal(line.textContent, 'Sign in again to keep Web access working.');
    assert.ok(line.classList.contains('is-warn'));
    assert.equal(pill.textContent, 'Needs attention');
    assert.ok(buttons(card).includes('Restart sign-in'), 'existing restart action offered');
    assert.doesNotMatch(card.textContent, /Cancel this setup and link again/);
    assert.doesNotMatch(card.textContent, LEAKS);
  });
});

describe('Web access status: restraint', () => {
  it('shows nothing for a sign-in link row, a cancelled row, an unknown value or no connection', async () => {
    const cases = [
      [connection({ state: 'pending', verified: false, access_state: undefined, access_expires_at_ms: undefined, sign_in_state: 'required' })],
      [connection({ state: 'cancelled', verified: false })],
      [connection({ access_state: 'something_new' })],
      [],
    ];
    for (const rows of cases) {
      const { line } = await mount(rows);
      assert.ok(line && line.hidden, 'access line stays hidden');
      assert.equal(line.textContent, '');
    }
  });

  it('with several rows, shows the most urgent state only', async () => {
    const { line } = await mount([
      connection(),
      connection({ connection_id: 'd'.repeat(32), access_state: 'ended', access_expires_at_ms: Date.now() - DAY }),
    ]);
    assert.equal(line.textContent, 'Web access has ended. Turn it on again to reconnect your apps.');
  });

  it('writes with textContent only: a hostile value cannot become markup', async () => {
    const { card, line } = await mount([connection({ access_state: '<img src=x onerror=alert(1)>', access_expires_at_ms: '<b>x</b>' })]);
    assert.equal(card.querySelector('img'), null);
    assert.ok(line.hidden);
  });

  it('keeps the line honest when a refresh moves from ending soon to renewing', async () => {
    const m = await mount([connection({ access_state: 'ending_soon', access_expires_at_ms: Date.now() + 2 * DAY })]);
    assert.match(m.line.textContent, /^Web access ends on /);
    assert.equal(m.pill.textContent, 'Ending soon');
    await m.refresh([connection()]);
    assert.equal(m.line.textContent, 'Renews automatically.');
    assert.ok(!m.line.classList.contains('is-warn'));
    assert.equal(m.pill.textContent, 'Ready');
    await m.refresh([connection({ access_state: 'ended', access_expires_at_ms: Date.now() - DAY })]);
    assert.match(m.line.textContent, /^Web access has ended\./);
  });
});
