/**
 * tests/ui/test_connected_apps_replaced.mjs — an app's older approval is marked "replaced".
 *
 * When an app reconnects, the gateway keeps the old approval next to the new one.
 * The Connected apps list groups approvals of the same app (same connection, same
 * name, same sign-in host), shows the newest as current, and marks each older one
 * "Replaced by a newer approval on <date>" with a Remove access button that uses
 * the existing revoke route. Nothing is removed automatically.
 *
 * Runner: node scripts/run-ui-tests.mjs   (needs jsdom)
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const source = (name) => readFileSync(join(UI, 'js', name), 'utf8');

const CID = 'a'.repeat(32);
const DAY = 86400e3;
const HOSTILE = '<img src=x onerror=alert(1)>';

function makeApi(handlers) {
  const calls = [];
  function fetchImpl(path, init) {
    const method = (init && init.method) || 'GET';
    calls.push({ path, method, body: init && init.body ? JSON.parse(init.body) : null });
    for (const [m, pattern, handler] of handlers) {
      if (method === m && pattern.exec(path)) {
        return Promise.resolve(handler()).then((out) => ({
          ok: out.status >= 200 && out.status < 300, status: out.status, json: () => Promise.resolve(out.body),
        }));
      }
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) });
  }
  return { fetchImpl, calls };
}

const settle = async (window, ticks = 8) => {
  for (let i = 0; i < ticks; i += 1) await new Promise((r) => window.setTimeout(r, 0));
};

function app(overrides) {
  return Object.assign({
    authorization_id: 'auth-new', name: 'Cursor', client_host: 'cursor.com',
    permissions: { read: true, save: true, session: false }, version: 2,
    connected_at_ms: Date.now() - 1 * DAY, last_used_at_ms: Date.now() - 3600e3,
  }, overrides || {});
}

async function mount(appsBody, handlers) {
  const api = makeApi([
    ['GET', new RegExp(`/api/v3/connections/${CID}/apps$`), () => ({ status: 200, body: { connection_id: CID, apps: appsBody } })],
  ].concat(handlers || []));
  const dom = new JSDOM('<!doctype html><html><body></body></html>', { runScripts: 'dangerously', url: 'http://localhost:8765/' });
  const { window } = dom;
  await new Promise((resolve) => (window.document.readyState === 'complete' ? resolve() : window.addEventListener('load', resolve)));
  window.fetch = api.fetchImpl;
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  for (const name of ['od-apps-ui.js', 'od-apps-list.js']) {
    const el = window.document.createElement('script');
    el.textContent = source(name);
    window.document.head.appendChild(el);
  }
  const list = window.odCreateConnectedAppsList();
  window.document.body.appendChild(list);
  await list.odSetConnections([CID], 'default');
  await settle(window);
  return { dom, window, document: window.document, list, api };
}

const rows = (document) => Array.from(document.querySelectorAll('[data-apps-list] .apps-row'));
const chipsOf = (row) => Array.from(row.querySelectorAll('.apps-chip')).map((c) => c.textContent);

describe('Connected apps: an older approval of the same app', () => {
  it('shows the newest as current and the older one as replaced, with its own permissions', async () => {
    const olderMs = Date.now() - 20 * DAY;
    const newerMs = Date.now() - 1 * DAY;
    // The older approval is listed FIRST by the server on purpose: order must not decide.
    const { dom, document } = await mount([
      app({ authorization_id: 'auth-old', version: 1, permissions: { read: true, save: false, session: false },
        connected_at_ms: olderMs, last_used_at_ms: null }),
      app({ authorization_id: 'auth-new', connected_at_ms: newerMs }),
    ]);
    const found = rows(document);
    assert.equal(found.length, 2);
    const current = found.find((r) => r.dataset.authorizationId === 'auth-new');
    const old = found.find((r) => r.dataset.authorizationId === 'auth-old');

    assert.ok(!current.classList.contains('is-replaced'));
    assert.equal(current.querySelector('[data-replaced]'), null);
    assert.match(current.textContent, /Current/);
    assert.deepEqual(chipsOf(current), ['Read', 'Save']);

    assert.ok(old.classList.contains('is-replaced'));
    const expectedDate = new Date(newerMs).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    assert.equal(old.querySelector('[data-replaced]').textContent, `Replaced by a newer approval on ${expectedDate}`);
    assert.match(old.textContent, /still works until you remove it/);
    assert.deepEqual(chipsOf(old), ['Read'], 'the older approval keeps showing what it can do');
    assert.ok(found.indexOf(old) === found.indexOf(current) + 1, 'the replaced row sits right under the current one');
    dom.window.close();
  });

  it('Remove access on the replaced row calls the existing revoke route for that approval only', async () => {
    const { dom, window, document, api } = await mount([
      app({ authorization_id: 'auth-old', version: 1, connected_at_ms: Date.now() - 20 * DAY }),
      app(),
    ], [['POST', /\/apps\/auth-old\/revoke$/, () => ({ status: 200, body: { revoked: true } })]]);

    const old = rows(document).find((r) => r.dataset.authorizationId === 'auth-old');
    const button = old.querySelector('[data-remove-access]');
    assert.equal(button.textContent, 'Remove access');
    assert.equal(api.calls.filter((c) => c.method === 'POST').length, 0, 'nothing is removed automatically');

    button.click();
    const dialog = document.querySelector('[role="dialog"]');
    assert.match(dialog.textContent, /older approval/);
    assert.match(dialog.textContent, /newer approval for Cursor is not affected/);
    Array.from(dialog.querySelectorAll('button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);

    const posts = api.calls.filter((c) => c.method === 'POST');
    assert.equal(posts.length, 1);
    assert.equal(posts[0].path, `/api/v3/connections/${CID}/apps/auth-old/revoke`);
    assert.deepEqual(posts[0].body, { profile_id: 'default', expected_version: 1 });
    const left = rows(document);
    assert.equal(left.length, 1);
    assert.equal(left[0].dataset.authorizationId, 'auth-new');
    assert.equal(left[0].querySelector('[data-replaced]'), null);
    assert.ok(!/Current/.test(left[0].textContent), 'a lone approval is not labelled Current');
    dom.window.close();
  });

  it('keeps the newest as current and replaced rows in order when there are three approvals', async () => {
    const { dom, document } = await mount([
      app({ authorization_id: 'a-1', connected_at_ms: Date.now() - 30 * DAY }),
      app({ authorization_id: 'a-3', connected_at_ms: Date.now() - 1 * DAY }),
      app({ authorization_id: 'a-2', connected_at_ms: Date.now() - 10 * DAY }),
    ]);
    assert.deepEqual(rows(document).map((r) => r.dataset.authorizationId), ['a-3', 'a-2', 'a-1']);
    assert.deepEqual(rows(document).map((r) => Boolean(r.querySelector('[data-replaced]'))), [false, true, true]);
    dom.window.close();
  });

  it('removing the current approval promotes the next newest to current', async () => {
    const { dom, window, document } = await mount([
      app({ authorization_id: 'auth-old', version: 1, connected_at_ms: Date.now() - 20 * DAY }),
      app(),
    ], [['POST', /\/apps\/auth-new\/revoke$/, () => ({ status: 200, body: { revoked: true } })]]);
    const current = rows(document).find((r) => r.dataset.authorizationId === 'auth-new');
    current.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);
    const left = rows(document);
    assert.equal(left.length, 1);
    assert.equal(left[0].dataset.authorizationId, 'auth-old');
    assert.equal(left[0].querySelector('[data-replaced]'), null, 'the survivor is now the only approval, so it is not "replaced"');
    dom.window.close();
  });

  it('shows a hostile app name as plain text in the replaced row too', async () => {
    const { dom, window, document } = await mount([
      app({ authorization_id: 'h-old', name: HOSTILE, connected_at_ms: Date.now() - 9 * DAY }),
      app({ authorization_id: 'h-new', name: HOSTILE }),
    ]);
    let alerted = false; window.alert = () => { alerted = true; };
    assert.equal(document.querySelectorAll('img, [onerror]').length, 0);
    assert.equal(alerted, false);
    assert.equal(rows(document).length, 2);
    dom.window.close();
  });
});

describe('Connected apps: approvals that are NOT the same app', () => {
  it('does not group different apps, even from the same host', async () => {
    const { dom, document } = await mount([
      app({ authorization_id: 'x-1', name: 'Cursor', connected_at_ms: Date.now() - 20 * DAY }),
      app({ authorization_id: 'x-2', name: 'ChatGPT', client_host: 'cursor.com' }),
    ]);
    assert.equal(rows(document).length, 2);
    assert.equal(document.querySelectorAll('[data-replaced]').length, 0);
    assert.ok(!/Replaced/.test(document.body.textContent));
    dom.window.close();
  });

  it('does not group the same name signing in from a different host', async () => {
    const { dom, document } = await mount([
      app({ authorization_id: 'h-1', client_host: 'cursor.com', connected_at_ms: Date.now() - 20 * DAY }),
      app({ authorization_id: 'h-2', client_host: 'grok.example' }),
    ]);
    assert.equal(document.querySelectorAll('[data-replaced]').length, 0);
    dom.window.close();
  });

  it('does not label an approval replaced when the connect times are unknown or tied', async () => {
    const tied = Date.now() - 5 * DAY;
    const { dom, document } = await mount([
      app({ authorization_id: 'u-1', connected_at_ms: null }),
      app({ authorization_id: 'u-2', connected_at_ms: null }),
      app({ authorization_id: 'u-3', name: 'Tied', connected_at_ms: tied }),
      app({ authorization_id: 'u-4', name: 'Tied', connected_at_ms: tied }),
    ]);
    assert.equal(rows(document).length, 4);
    assert.equal(document.querySelectorAll('[data-replaced]').length, 0);
    dom.window.close();
  });
});

describe('Connected apps: a single approval', () => {
  it('shows no "replaced" text and no Current label', async () => {
    const { dom, document } = await mount([app()]);
    const [row] = rows(document);
    assert.ok(!/Replaced|older approval|Current/.test(row.textContent));
    assert.equal(document.querySelectorAll('[data-replaced], .apps-current').length, 0);
    assert.equal(row.querySelector('[data-remove-access]').getAttribute('aria-label'), 'Remove access for Cursor');
    dom.window.close();
  });
});
