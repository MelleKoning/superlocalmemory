/**
 * tests/ui/test_settings_reindex_progress.mjs — Settings shows a background
 * embedding re-index (4.1.22).
 *
 * Since 4.1.22 changing the embedding model answers 202 with a job and the
 * daemon re-indexes in the background; recall keeps using the old model until
 * the switch completes. The dashboard used to answer that 202 with "Saved"
 * and show nothing else, so a user could not tell a 40-minute re-index was
 * running, how far it was, or that it had failed. These tests drive the real
 * od-settings.js + od-reindex.js against a stubbed daemon and a controlled
 * clock: polling timers only run when the test fires them.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function reply(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function job(over = {}) {
  return Object.assign({ job_id: 7, kind: 'switch', state: 'queued',
    from: 'nomic::768', to: 'stub-new::384', done: 0, total: 10, eta_seconds: null,
    error: null }, over);
}

/**
 * daemon: { initial: [status, body] for the GET made when Settings opens,
 *           save: [status, body], cancel: [status, body],
 *           status: [[status, body], ...] one per later poll, the last repeats }
 */
async function mount(daemon) {
  const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
  const timers = [];
  const calls = [];
  h.window.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
  h.window.clearTimeout = () => {};
  h.window.showToast = () => {};
  const statuses = (daemon.status || []).slice();
  let opened = false;
  h.window.fetch = async function (url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    calls.push({ url, method, headers: options.headers || {} });
    if (url === '/internal/token') return reply(200, { token: 'tok' });
    if (url === '/api/v3/embedding/reindex' && method === 'GET') {
      if (!opened) return reply(...(daemon.initial || [200, { job: null }]));
      const next = statuses.length > 1 ? statuses.shift() : statuses[0];
      return next ? reply(next[0], next[1]) : reply(200, { job: null });
    }
    if (url === '/api/v3/embedding/reindex/cancel') return reply(...daemon.cancel);
    if (url === '/api/v3/mode/set') return reply(...daemon.save);
    return reply(200, {});
  };
  evalModule(h.window, 'od-reindex.js');
  evalModule(h.window, 'od-settings.js');
  h.window.odRenderSettings(h.document.getElementById('mount'));
  await flushPromises();
  opened = true;
  const panel = () => {
    const p = h.document.getElementById('od-reindex');
    assert.ok(p, 'no re-index panel in the Embeddings group');
    return p;
  };
  const polls = () => calls.filter((c) => c.url === '/api/v3/embedding/reindex' && c.method === 'GET').length;
  async function tick() {
    const due = timers.splice(0);
    due.forEach((t) => t.fn());
    await flushPromises();
    await flushPromises();
  }
  async function save() {
    h.document.getElementById('od-s-emb-save').click();
    await flushPromises();
    await flushPromises();
  }
  return { h, calls, timers, panel, polls, tick, save };
}

const ACCEPTED = [202, { success: true, accepted: true, job: job(),
  detail: 're-indexing 10 memories with stub-new::384 in the background; recall keeps ' +
          'using nomic::768 until it is done.' }];

describe('settings: background embedding re-index', function () {
  it('a 202 save is shown as started, not as Saved, with the job state and progress',
    async function () {
      const t = await mount({ save: ACCEPTED, status: [[200, { job: job({ state: 'queued' }) }]] });
      await t.save();
      const st = t.h.document.getElementById('od-s-emb-st').textContent;
      assert.doesNotMatch(st, /^Saved/);
      assert.match(st, /background/);
      const p = t.panel();
      assert.ok(p, 'no re-index panel in the Embeddings group');
      assert.notEqual(p.style.display, 'none');
      assert.match(p.textContent, /Queued/i);
      assert.match(p.textContent, /0\s*\/\s*10/);
      assert.match(p.textContent, /nomic::768/, 'does not say recall keeps the old model');
    });

  it('polls while active, shows progress, and stops once the switch is done', async function () {
    const t = await mount({ save: ACCEPTED, status: [
      [200, { job: job({ state: 'running', done: 4, eta_seconds: 12 }) }],
      [200, { job: job({ state: 'catching_up', done: 10 }) }],
      [200, { job: job({ state: 'activated', done: 10 }) }],
    ] });
    const before = t.polls();
    await t.save();
    await t.tick();
    assert.match(t.panel().textContent, /4\s*\/\s*10/);
    assert.match(t.panel().textContent, /Re-indexing/i);
    await t.tick();
    assert.match(t.panel().textContent, /Catching up/i);
    await t.tick();
    assert.match(t.panel().textContent, /Done/i);
    assert.match(t.panel().textContent, /stub-new::384/);
    const settled = t.polls();
    assert.ok(settled - before >= 3, `expected >= 3 polls, saw ${settled - before}`);
    await t.tick();
    await t.tick();
    assert.equal(t.polls(), settled, 'kept polling after the job finished');
    assert.equal(t.h.document.getElementById('od-reindex-cancel').style.display, 'none');
  });

  it('a failed job is shown as failed, with its error, and the old model staying', async function () {
    const t = await mount({ save: ACCEPTED, status: [
      [200, { job: job({ state: 'failed', error: 'the model stub-new could not be started' }) }],
    ] });
    await t.save();
    await t.tick();
    const text = t.panel().textContent;
    assert.match(text, /Failed/);
    assert.match(text, /could not be started/);
    assert.match(text, /nomic::768/);
    const settled = t.polls();
    await t.tick();
    assert.equal(t.polls(), settled, 'kept polling after the job failed');
  });

  it('a 409 save names the job already running and starts showing it', async function () {
    const t = await mount({
      save: [409, { error: 'reindex_running', job_id: 3, state: 'running',
        detail: 'a re-index is already running (job 3: a::768 -> b::384). Wait for it' }],
      status: [[200, { job: job({ job_id: 3, state: 'running', done: 2 }) }]],
    });
    await t.save();
    const st = t.h.document.getElementById('od-s-emb-st').textContent;
    assert.match(st, /409/);
    assert.match(st, /job 3/);
    await t.tick();
    assert.match(t.panel().textContent, /job 3/);
    assert.match(t.panel().textContent, /2\s*\/\s*10/);
  });

  it('cancel posts to the cancel route with the install token and shows the result',
    async function () {
      const t = await mount({ save: ACCEPTED,
        cancel: [200, { success: true, job: job({ state: 'cancelled' }) }],
        status: [[200, { job: job({ state: 'running', done: 1 }) }],
                 [200, { job: job({ state: 'cancelled', error: 'cancelled' }) }]] });
      await t.save();
      await t.tick();
      const btn = t.h.document.getElementById('od-reindex-cancel');
      assert.notEqual(btn.style.display, 'none', 'no cancel action while running');
      btn.click();
      await flushPromises();
      await flushPromises();
      const post = t.calls.find((c) => c.url === '/api/v3/embedding/reindex/cancel');
      assert.ok(post, 'cancel never reached the daemon');
      assert.equal(post.method, 'POST');
      assert.equal(post.headers['X-Install-Token'], 'tok');
      assert.match(t.panel().textContent, /Cancelled/);
      assert.match(t.panel().textContent, /nomic::768/);
    });

  it('a refused cancel shows the status code and reason', async function () {
    const t = await mount({ save: ACCEPTED,
      cancel: [409, { error: 'refused', detail: 'no re-index is running' }],
      status: [[200, { job: job({ state: 'running', done: 1 }) }]] });
    await t.save();
    await t.tick();
    t.h.document.getElementById('od-reindex-cancel').click();
    await flushPromises();
    await flushPromises();
    assert.match(t.panel().textContent, /409/);
    assert.match(t.panel().textContent, /no re-index is running/);
  });

  it('opening Settings while a job runs shows it without any save', async function () {
    const t = await mount({ initial: [200, { job: job({ state: 'running', done: 6 }) }] });
    assert.match(t.panel().textContent, /6\s*\/\s*10/);
    assert.ok(t.timers.length > 0, 'no poll scheduled for the running job');
  });

  it('with no job, or no daemon runner, the panel stays hidden and nothing polls', async function () {
    for (const initial of [[200, { job: null }], [409, { error: 'daemon_required', detail: 'x' }]]) {
      const t = await mount({ initial });
      assert.equal(t.panel().style.display, 'none');
      const polled = t.polls();
      await t.tick();
      assert.equal(t.polls(), polled);
    }
  });

  it('server strings are rendered as text, never as HTML', async function () {
    const t = await mount({ initial: [200, { job: job({ state: 'failed',
      error: '<img src=x onerror=alert(1)>' }) }] });
    // a terminal job seen on load is history: hidden, never injected
    assert.equal(t.panel().querySelector('img'), null);
    const s = await mount({ save: ACCEPTED, status: [[200, { job: job({ state: 'failed',
      error: '<img src=x onerror=alert(1)>' }) }]] });
    await s.save();
    await s.tick();
    assert.equal(s.panel().querySelector('img'), null);
    assert.match(s.panel().textContent, /<img src=x/);
  });
});
