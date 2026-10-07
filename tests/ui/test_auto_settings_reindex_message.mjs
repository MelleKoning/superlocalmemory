/**
 * tests/ui/test_auto_settings_reindex_message.mjs — the legacy Settings save
 * tells the truth about an embedding change (4.1.22).
 *
 * It used to say "Embeddings will be re-indexed on next use" — never true
 * since 4.1.22, where a model change is a background job in the daemon and
 * recall keeps using the old model until the switch completes. And because
 * the 202 body carries no needs_reindex flag, a started job read as a plain
 * "Configuration saved!". A rejected save showed "Save failed" with no code
 * and no reason.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function reply(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

async function saveWith(status, body) {
  const h = buildHarness(['settings-save-status', 'settings-save-all'],
    { ok: true, status: 200, json: {} });
  h.window.setTimeout = () => 0;
  h.window.fetch = async (url, options = {}) =>
    (url === '/api/v3/mode/set' ? reply(status, body) : reply(200, {}));
  evalModule(h.window, 'auto-settings.js');
  await h.window.saveAllSettings();
  await flushPromises();
  return h.document.getElementById('settings-save-status').textContent;
}

describe('legacy settings save: embedding change outcome', function () {
  it('a 202 shows the server\'s background re-index message', async function () {
    const detail = 're-indexing 10 memories with b::384 in the background; recall keeps ' +
                   'using a::768 until it is done. Progress: slm embedder status';
    const text = await saveWith(202, { success: true, accepted: true,
      job: { job_id: 4, state: 'queued' }, detail });
    assert.match(text, /in the background/);
    assert.match(text, /recall keeps using a::768/);
    assert.doesNotMatch(text, /next use|next recall/);
  });

  it('a 200 that defers the re-index to the daemon shows the server\'s message', async function () {
    const text = await saveWith(200, { success: true, needs_reindex: true,
      message: 'the SLM daemon re-indexes your memories with b in the background' });
    assert.match(text, /re-indexes your memories with b in the background/);
    assert.doesNotMatch(text, /next use|next recall/);
  });

  it('a rejected save shows the status code and the server\'s reason', async function () {
    const text = await saveWith(409, { error: 'reindex_running', job_id: 3,
      detail: 'a re-index is already running (job 3: a -> b)' });
    assert.match(text, /409/);
    assert.match(text, /already running \(job 3/);
  });

  it('no UI text promises a re-index "on next use"', async function () {
    const { readFileSync } = await import('node:fs');
    const src = readFileSync(new URL('../../src/superlocalmemory/ui/js/auto-settings.js',
      import.meta.url), 'utf8');
    assert.doesNotMatch(src, /re-indexed on next use/);
  });
});
