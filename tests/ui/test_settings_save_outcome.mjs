/**
 * tests/ui/test_settings_save_outcome.mjs — every Settings save reports what
 * the daemon actually answered (4.1.22).
 *
 * Before: a rejected save showed a bare "Error" (or, for the auto-saving
 * switches, a toast with no reason and a switch left showing the value the
 * daemon refused). The status code and the server's own message were thrown
 * away, so a 409 "a re-index is already running (job 7 ...)" read the same as
 * a dead daemon. Now every save shows "Error <status>: <server message>".
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function reply(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

async function mount(rejection) {
  const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
  const toasts = [];
  const mutations = [];
  h.window.showToast = (msg) => toasts.push(String(msg));
  h.window.fetch = async function (url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    if (url === '/internal/token') return reply(200, { token: 'tok' });
    if (method === 'GET') return reply(200, {});
    mutations.push({ url, method });
    return reply(rejection.status, rejection.body);
  };
  evalModule(h.window, 'od-reindex.js');
  evalModule(h.window, 'od-settings.js');
  h.window.odRenderSettings(h.document.getElementById('mount'));
  await flushPromises();
  return { h, toasts, mutations };
}

const BUTTON_SAVES = [
  ['mode-save', 'mode'], ['emb-save', 'emb'], ['stor-save', 'stor'],
  ['forg-save', 'forg'], ['trust-save', 'trust'], ['rl-save', 'rl'],
  ['dmn-save', 'dmn'], ['evo-save', 'evo'], ['bk-save', 'bk'], ['bk-now', 'bk'],
];

const CONFLICT = {
  status: 409,
  body: { error: 'reindex_running', job_id: 7, state: 'running',
          detail: 'a re-index is already running (job 7: a::768 -> b::384)' },
};

describe('settings saves report the real outcome', function () {
  for (const [button, status] of BUTTON_SAVES) {
    it(`${button}: a 409 shows the status code and the server's message`, async function () {
      const { h, toasts, mutations } = await mount(CONFLICT);
      h.document.getElementById('od-s-' + button).click();
      await flushPromises();
      await flushPromises();
      assert.ok(mutations.length >= 1, 'the save never reached the daemon');
      const text = h.document.getElementById('od-s-' + status + '-st').textContent;
      assert.doesNotMatch(text, /^Saved|^Done/);
      assert.match(text, /409/, `status line hides the status code: "${text}"`);
      assert.match(text, /already running \(job 7/, `status line hides the reason: "${text}"`);
      assert.ok(toasts.some((t) => /already running \(job 7/.test(t)),
        `no toast carried the reason: ${JSON.stringify(toasts)}`);
    });
  }

  it('a 500 with only an error field still shows that error and the code', async function () {
    const { h } = await mount({ status: 500, body: { error: 'disk full' } });
    h.document.getElementById('od-s-trust-save').click();
    await flushPromises();
    await flushPromises();
    const text = h.document.getElementById('od-s-trust-st').textContent;
    assert.match(text, /500/);
    assert.match(text, /disk full/);
  });

  it('a 422 whose detail is {code, message} shows the message, not [object Object]', async function () {
    const { h } = await mount({ status: 422, body: { error: 'invalid_request',
      detail: { code: 'invalid_request', message: 'a model name is required' } } });
    h.document.getElementById('od-s-emb-save').click();
    await flushPromises();
    await flushPromises();
    const text = h.document.getElementById('od-s-emb-st').textContent;
    assert.match(text, /422/);
    assert.match(text, /a model name is required/);
    assert.doesNotMatch(text, /object Object/);
  });

  for (const sw of ['cap-en', 'rec-en', 'ai-en', 'mesh-en', 'scope-shared', 'rt-xenc']) {
    it(`${sw}: a refused auto-save toasts the code and reason`, async function () {
      const { h, toasts } = await mount({ status: 403, body: { detail: 'manage permission required' } });
      h.document.getElementById('od-s-' + sw).click();
      await flushPromises();
      await flushPromises();
      assert.ok(toasts.some((t) => /403/.test(t) && /manage permission required/.test(t)),
        `no toast carried the code and reason: ${JSON.stringify(toasts)}`);
    });
  }

  it('the provider test reports an HTTP failure instead of "Failed: "', async function () {
    const { h } = await mount({ status: 503, body: { detail: 'daemon is starting' } });
    h.document.getElementById('od-s-mode-test').click();
    await flushPromises();
    await flushPromises();
    const text = h.document.getElementById('od-s-mode-st').textContent;
    assert.match(text, /503/);
    assert.match(text, /daemon is starting/);
  });

  it('server text in a status line is shown as text, never HTML-escaped twice', async function () {
    const { h } = await mount({ status: 400, body: { error: 'model <b> & co' } });
    h.document.getElementById('od-s-stor-save').click();
    await flushPromises();
    await flushPromises();
    const el = h.document.getElementById('od-s-stor-st');
    assert.match(el.textContent, /model <b> & co/);
    assert.equal(el.querySelector('b'), null, 'server text was parsed as HTML');
  });

});
