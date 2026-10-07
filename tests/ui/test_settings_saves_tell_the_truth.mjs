/**
 * tests/ui/test_settings_saves_tell_the_truth.mjs — two Settings saves that
 * misreported what the daemon holds (found while fixing 4.1.22 save outcomes).
 *
 * 1. "Save all changes" clicked the backup Save button AND called the backup
 *    save directly, so every Save all sent the backup settings twice.
 * 2. The legacy auto-capture / auto-recall switches ignored a refused save
 *    (only console.log on a network error), leaving the switch showing a value
 *    the daemon never took.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function reply(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

async function mount() {
  const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
  const mutations = [];
  h.window.showToast = () => {};
  h.window.fetch = async function (url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    if (url === '/internal/token') return reply(200, { token: 'tok' });
    if (method !== 'GET') mutations.push({ url, method });
    return reply(200, { success: true });
  };
  for (const f of ['od-decay-preview.js', 'od-reindex.js']) {
    try { evalModule(h.window, f); } catch (e) { /* optional modules */ }
  }
  evalModule(h.window, 'od-settings.js');
  h.window.odRenderSettings(h.document.getElementById('mount'));
  await flushPromises();
  return { h, mutations };
}

describe('settings: Save all', function () {
  it('Save all sends the backup settings once, not twice', async function () {
    const { h, mutations } = await mount({ status: 200, body: { success: true } });
    const footer = Array.from(h.document.querySelectorAll('button'))
      .find((b) => b.textContent === 'Save all changes');
    footer.click();
    await flushPromises();
    await flushPromises();
    assert.equal(mutations.filter((m) => m.url === '/api/backup/configure').length, 1);
  });
});

describe('legacy settings: refused auto-capture / auto-recall saves', function () {
  for (const [id, url] of [['auto-capture-toggle', '/api/v3/auto-capture/config'],
                           ['auto-recall-toggle', '/api/v3/auto-recall/config']]) {
    it(`${id}: a refused save puts the switch back to the daemon's value`, async function () {
      const h = buildHarness([], { ok: true, status: 200, json: {} });
      h.document.body.innerHTML = `<input type="checkbox" id="${id}">`;
      h.window.console.warn = () => {};
      const puts = [];
      h.window.fetch = async (u, options = {}) => {
        if ((options.method || 'GET') === 'PUT') { puts.push(u); return reply(403, { detail: 'no' }); }
        return reply(200, { config: { enabled: false } });  // the daemon's truth: off
      };
      evalModule(h.window, 'auto-settings.js');
      const box = h.document.getElementById(id);
      box.checked = true;
      box.dispatchEvent(new h.window.Event('change', { bubbles: true }));
      await flushPromises();
      await flushPromises();
      assert.equal(box.checked, false, 'switch still shows the refused value');
      assert.deepEqual(puts, [url]);
    });
  }
});
