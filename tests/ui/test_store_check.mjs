/**
 * Health → Memory store: what the owner sees after the upgrade check, and that
 * Repair now only appears when there is something to repair and posts to the
 * backup-first route. Sidebar Health carries a "Repair" tag only while needed.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';

const SCRIPT = readFileSync(new URL('../../src/superlocalmemory/ui/js/od-store-check.js', import.meta.url), 'utf8');
const LABELS = { stale_vectors: 'search vectors that no longer match their memory', lance_orphans: 'LanceDB search entries for memories that are gone' };
const tick = () => new Promise((r) => setTimeout(r, 0));

async function setup(summary) {
  const dom = new JSDOM('<a class="nav-link" data-tab="health-pane"><span>Health</span></a><div id="store-check"></div>', { runScripts: 'outside-only' });
  const calls = [];
  dom.window.slmFetch = (path, init) => {
    calls.push({ path, method: (init && init.method) || 'GET' });
    const body = path === '/api/integrity/summary' ? { labels: LABELS, running: false, ...summary() } : { started: 'x' };
    return Promise.resolve(new dom.window.Response(JSON.stringify(body), { status: path === '/api/integrity/summary' ? 200 : 202 }));
  };
  dom.window.Response = Response;
  dom.window.setTimeout = () => 0;
  dom.window.eval(SCRIPT);
  for (let i = 0; i < 4; i += 1) await tick();
  return { dom, calls, mount: dom.window.document.getElementById('store-check'), nav: dom.window.document.querySelector('a.nav-link') };
}
const buttons = (mount) => [...mount.querySelectorAll('button')].map((b) => b.textContent);

test('before the first check the owner is told when it will happen', async () => {
  const f = await setup(() => ({}));
  assert.match(f.mount.textContent, /Not checked yet/);
  assert.deepEqual(buttons(f.mount), ['Check now']);
  assert.equal(f.nav.querySelector('[data-store-check-tag]'), null);
});

test('a clean store says so, with no repair button and no sidebar tag', async () => {
  const f = await setup(() => ({ status: 'checked', checked_at: '2026-10-08T10:00:00+00:00', to_repair: 0, findings: {} }));
  assert.match(f.mount.textContent, /everything is in order/);
  assert.deepEqual(buttons(f.mount), ['Check again']);
  assert.equal(f.nav.querySelector('[data-store-check-tag]'), null);
});

test('findings are listed in plain words and Repair now posts to the backup-first route', async () => {
  let state = { status: 'checked', checked_at: '2026-10-08T10:00:00+00:00', to_repair: 18, findings: { stale_vectors: 11, lance_orphans: 7 } };
  const f = await setup(() => state);
  assert.match(f.mount.textContent, /18 items can be repaired/);
  assert.match(f.mount.textContent, /11 search vectors that no longer match their memory/);
  assert.match(f.mount.textContent, /backup copy of your memory first/);
  assert.equal(f.nav.querySelector('[data-store-check-tag]').textContent, 'Repair');
  state = { ...state, status: 'repairing', running: true };
  [...f.mount.querySelectorAll('button')].find((b) => b.textContent === 'Repair now').click();
  for (let i = 0; i < 6; i += 1) await tick();
  assert.ok(f.calls.some((c) => c.path === '/api/integrity/repair-now' && c.method === 'POST'));
  assert.match(f.mount.textContent, /Saving a backup copy first, then repairing/);
  assert.deepEqual(buttons(f.mount), ['Check again']);
  assert.ok(f.mount.querySelector('button').disabled);
});

test('a failed repair shows why, and server text is never parsed as markup', async () => {
  const f = await setup(() => ({ status: 'repair_failed', error: '<img src=x onerror=alert(1)> The backup copy could not be made, so nothing was changed.' }));
  assert.match(f.mount.textContent, /nothing was changed/);
  assert.equal(f.mount.querySelector('img'), null);
});

test('the sidebar tag appears at page load, before the Health page is opened', async () => {
  const dom = new JSDOM('<a class="nav-link" data-tab="health-pane"><span>Health</span></a>', { runScripts: 'outside-only' });
  dom.window.slmFetch = () => Promise.resolve(new Response(JSON.stringify({ labels: LABELS, running: false, status: 'checked', to_repair: 2, findings: { stale_vectors: 2 } }), { status: 200 }));
  dom.window.setTimeout = () => 0;
  dom.window.eval(SCRIPT);
  for (let i = 0; i < 4; i += 1) await tick();
  assert.equal(dom.window.document.querySelector('[data-store-check-tag]').textContent, 'Repair');
});
