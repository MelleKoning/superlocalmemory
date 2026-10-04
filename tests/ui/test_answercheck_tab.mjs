/**
 * tests/ui/test_answercheck_tab.mjs — the Answer Check tab (4.1.20).
 * Runner: npm test   (scripts/run-ui-tests.mjs). Requires jsdom.
 *
 * Every state of the status card, the stats and the feed is rendered against
 * a routed fetch stub; the item renderer must ignore fields it does not know
 * (a server bug adding `query` or `content` renders nothing extra); every
 * outcome has words, not only a colour; polling stops when the pane hides.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

const OUTCOMES = ['answered', 'abstained', 'nothing_found', 'not_checked_off', 'not_checked_time',
  'not_checked_shared', 'not_checked_busy', 'not_checked_loading', 'not_checked_unavailable'];

function status(over = {}) {
  return Object.assign({ mode: 'laya', active: 'laya', apple_silicon: true,
    laya: { state: 'ready' }, jev: { provider: 'typesafe', has_key: false, key_problem: '',
    consent: false, rerank: { enabled: false, active: false, k: 20 } } }, over);
}

function item(id, over = {}) {
  return Object.assign({ id: id.padEnd(32, '0'), at: '2026-10-04T09:12:44.120Z', outcome: 'answered',
    status: 'judged', detail: '', judge: 'laya', origin: 'other', abstained: false,
    abstention_reason: null, answer_confidence: 0.81, threshold: 0.5, reordered: false,
    result_count: 7, query_type: 'factual', retrieval_ms: 812.3, judge_ms: 201, total_ms: 1013.3,
    over_ceiling: false }, over);
}

function summary(over = {}) {
  return Object.assign({ enabled: true, profile: 'default', window: '7d', include_dashboard: false,
    counts: { total: 40, answered: 25, abstained: 5, nothing_found: 4,
              not_checked: { off: 1, time: 1, shared: 1, busy: 1, loading: 1, unavailable: 1 } },
    by_judge: { laya: 30, jev: 0 },
    abstention: { k: 5, n: 30, rate: 0.1667, ci95: [0.0734, 0.3365], min_sample: 20, enough: true },
    latency: { ceiling_ms: 3000, over_ceiling: 2, total: { n: 40, p50: 900, p95: 2600, max: 3400 },
               judge: { n: 30, p50: 200, p95: 450 }, recent_total_ms: [800, 900, 1200] },
    recording: { boot_id: 'b'.repeat(32), writer_running: true, last_saved_at: null, retention_days: 30,
                 max_rows: 10000, unsaved: 0,
                 since_start: { recorded: 40, saved: 40, dropped_before_save: 0, save_failures: 0 } } }, over);
}

/** Build the page and route fetches by path. routes: {path: {status, body} | fn(url, init)} */
function page(routes) {
  const h = buildHarness(['answercheck-pane'], { ok: true, status: 200, json: {} });
  const w = h.window;
  const calls = [];
  w.fetch = function (url, init) {
    calls.push({ url: String(url), method: (init && init.method) || 'GET' });
    const path = String(url).split('?')[0];
    let r = routes[path];
    if (typeof r === 'function') r = r(String(url), init);
    r = r || { status: 404, body: {} };
    return Promise.resolve({ ok: r.status >= 200 && r.status < 300, status: r.status,
                             json: () => Promise.resolve(r.body) });
  };
  const timers = [];
  w.setInterval = function (fn) { timers.push({ fn, cleared: false }); return timers.length; };
  w.clearInterval = function (id) { if (timers[id - 1]) timers[id - 1].cleared = true; };
  evalModule(w, 'od-answercheck-tryit.js');
  evalModule(w, 'od-answercheck.js');
  const pane = w.document.getElementById('answercheck-pane');
  pane.className = 'tab-pane fade show active';
  return { w, d: w.document, pane, calls, timers };
}

function defaults(over = {}) {
  return Object.assign({
    '/api/v3/answer-check': { status: 200, body: status() },
    '/api/v3/answer-check/history/summary': { status: 200, body: summary() },
    '/api/v3/answer-check/history': { status: 200, body: { enabled: true, items: [item('a1'), item('a2',
      { outcome: 'abstained', abstained: true, origin: 'dashboard' })], next_cursor: null } },
    '/api/v3/answer-check/history/live': { status: 200, body: { enabled: true, boot_id: 'b'.repeat(32),
      last_seq: 2, items: [] } },
  }, over);
}

async function render(over) {
  const p = page(defaults(over));
  p.w.odRenderAnswerCheck(p.pane);
  await flushPromises(); await flushPromises();
  return p;
}

const text = (p, sel) => (p.d.querySelector(sel) || {}).textContent || '';

describe('Answer Check tab — sections and data', function () {
  it('renders every section from live data', async function () {
    const p = await render();
    for (const id of ['ac-status', 'ac-tryit', 'ac-stats', 'ac-latency', 'ac-feed']) {
      assert.ok(p.d.getElementById(id), id);
    }
    assert.match(text(p, '#ac-status'), /On this Mac — memory text never leaves it/);
    assert.match(text(p, '#ac-stats'), /17%/);
    assert.match(text(p, '#ac-stats'), /of 30 checked answers/);
    assert.match(text(p, '#ac-stats'), /95% range 7\.3%–34%/);
    assert.match(text(p, '#ac-latency'), /Over the limit: 2/);
    assert.equal(p.d.querySelectorAll('#ac-feed tbody tr').length, 2);
    assert.match(text(p, '#ac-feed'), /Dashboard test/);
    assert.match(text(p, '#ac-feed'), /Kept on this machine for 30 days/);
    assert.ok(!/not saved/.test(text(p, '#ac-feed')), 'no warning when nothing was lost');
  });

  it('withholds the rate below the minimum sample', async function () {
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 200, body: summary({
      abstention: { k: 3, n: 12, rate: null, ci95: null, min_sample: 20, enough: false } }) } });
    assert.match(text(p, '#ac-stats'), /Not enough checks yet \(12 of 20 needed\)/);
    assert.ok(!/25%/.test(text(p, '#ac-stats')));
  });

  it('warns when checks were not saved', async function () {
    const s = summary();
    s.recording.since_start.dropped_before_save = 3;
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 200, body: s } });
    assert.match(text(p, '#ac-feed'), /3 checks were not saved \(the history was busy\)\. Your recalls were not affected\./);
  });

  it('says how many checks an erasure removed before they were saved', async function () {
    const s = summary();
    s.recording.since_start.erased_unsaved = 2;
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 200, body: s } });
    assert.match(text(p, '#ac-feed'), /2 checks were not saved because their history was erased first\./);
    assert.ok(!/history was busy/.test(text(p, '#ac-feed')), 'an erasure is not a busy history');
  });

  it('item renderer ignores fields it does not know', async function () {
    const leaky = item('z1', { query: 'SECRET-Q', content: 'SECRET-M', fact_id: 'FACT-9' });
    const p = await render({ '/api/v3/answer-check/history': { status: 200,
      body: { enabled: true, items: [leaky], next_cursor: null } } });
    const html = p.pane.innerHTML;
    assert.ok(!html.includes('SECRET-Q') && !html.includes('SECRET-M') && !html.includes('FACT-9'));
  });

  it('every outcome has words, not only a colour', async function () {
    const items = OUTCOMES.map((o, i) => item('o' + i, { outcome: o }));
    const p = await render({ '/api/v3/answer-check/history': { status: 200,
      body: { enabled: true, items, next_cursor: null } } });
    const cells = [...p.d.querySelectorAll('#ac-feed tbody tr td:nth-child(2) .badge:first-child')];
    assert.equal(cells.length, OUTCOMES.length);
    for (const c of cells) assert.ok(c.textContent.trim().length > 5, c.textContent);
    assert.equal(new Set(cells.map(c => c.textContent)).size, OUTCOMES.length);
  });

  it('has live regions for the verdict and for new rows', async function () {
    const p = await render();
    assert.ok(p.d.querySelector('.ac-live[aria-live="polite"]'));
    assert.ok(p.d.querySelector('.ac-verdict[aria-live="polite"]'));
  });

  it('announces new checks from the live feed', async function () {
    const p = await render();
    p.w.fetch = ((orig) => function (url, init) {
      if (String(url).includes('/history/live')) {
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ enabled: true,
          boot_id: 'b'.repeat(32), last_seq: 4, items: [item('n1'), item('n2')] }) });
      }
      return orig(url, init);
    })(p.w.fetch);
    p.timers[0].fn();
    await flushPromises();
    assert.equal(text(p, '.ac-live'), '2 new checks');
    assert.equal(p.d.querySelectorAll('#ac-feed tbody tr').length, 4);
  });

  it('loads older checks with the cursor', async function () {
    const cursor = '1000:' + 'c'.repeat(32);
    const p = await render({ '/api/v3/answer-check/history': (url) => url.includes('cursor=')
      ? { status: 200, body: { enabled: true, items: [item('old1')], next_cursor: null } }
      : { status: 200, body: { enabled: true, items: [item('a1')], next_cursor: cursor } } });
    p.d.querySelector('.ac-older').click();
    await flushPromises();
    assert.ok(p.calls.some(c => c.url.includes('cursor=' + encodeURIComponent(cursor))));
    assert.equal(p.d.querySelectorAll('#ac-feed tbody tr').length, 2);
    assert.equal(p.d.querySelector('.ac-older'), null);
  });
});

describe('Answer Check tab — states', function () {
  const cases = [
    ['never set up', status({ mode: 'auto', active: 'off', laya: { state: 'not_installed' } }), /Answer check is not set up/],
    ['judge off', status({ mode: 'off', active: 'off', laya: { state: 'ready' } }), /Off/],
    ['Jev consent missing', status({ mode: 'jev', active: 'off', jev: { provider: 'typesafe', has_key: true, consent: false } }),
      /consent was not given, so nothing is checked and nothing is sent/],
    ['Jev key missing', status({ mode: 'jev', active: 'off', jev: { provider: 'typesafe', has_key: false, consent: true } }),
      /no key is saved yet/],
    ['key problem', status({ mode: 'jev', active: 'off', jev: { provider: 'typesafe', has_key: false, consent: true,
      key_problem: 'The saved key could not be read.' } }), /The saved key could not be read\./],
    ['Intel Mac', status({ mode: 'laya', active: 'off', apple_silicon: false, laya: { state: 'unsupported' } }),
      /needs Apple Silicon/],
    ['installing', status({ mode: 'laya', active: 'off', laya: { state: 'installing' } }), /Setting up the on-device model/],
    ['Jev running', status({ mode: 'jev', active: 'jev', jev: { provider: 'openrouter', has_key: true, consent: true,
      rerank: { active: true } } }), /Online via OpenRouter — the question and the top memories are sent there/],
  ];
  for (const [name, s, re] of cases) {
    it(name, async function () {
      const p = await render({ '/api/v3/answer-check': { status: 200, body: s } });
      assert.match(text(p, '#ac-status'), re);
      assert.ok(p.d.querySelector('#ac-status .ac-settings-link'), 'always links to Settings');
    });
  }

  it('no access keeps the status card and hides the activity', async function () {
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 403, body: {} },
                             '/api/v3/answer-check/history': { status: 403, body: {} } });
    assert.match(text(p, '.ac-root'), /Your role cannot view answer-check activity in this workspace\./);
    assert.ok(p.d.getElementById('ac-status'));
    assert.equal(p.d.getElementById('ac-feed'), null);
    assert.ok(p.timers.every(t => t.cleared), 'no polling without access');
  });

  it('history turned off', async function () {
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 200, body: { enabled: false } },
                             '/api/v3/answer-check/history': { status: 200, body: { enabled: false, items: [] } } });
    assert.match(text(p, '#ac-stats'), /History is turned off on this machine\./);
    assert.match(text(p, '#ac-feed'), /History is turned off on this machine\./);
  });

  it('empty history', async function () {
    const empty = summary({ counts: { total: 0, answered: 0, abstained: 0, nothing_found: 0, not_checked: {} } });
    const p = await render({ '/api/v3/answer-check/history/summary': { status: 200, body: empty },
                             '/api/v3/answer-check/history': { status: 200, body: { enabled: true, items: [], next_cursor: null } } });
    assert.match(text(p, '#ac-stats .ac-empty'), /No checks yet/);
  });

  it('daemon error uses the pane-error convention with Retry', async function () {
    let n = 0;
    const p = await render({ '/api/v3/answer-check/history/summary': () => (n++ === 0
      ? { status: 500, body: {} } : { status: 200, body: summary() }) });
    assert.ok(p.d.querySelector('#ac-stats .pane-error[role="alert"]'));
    p.d.querySelector('#ac-stats .pane-error-retry').click();
    await flushPromises();
    assert.equal(p.d.querySelector('#ac-stats .pane-error'), null);
    assert.match(text(p, '#ac-stats'), /17%/);
  });

  it('clear denied hides the button and says why', async function () {
    const p = await render({ '/api/v3/answer-check/history': (url, init) => (init && init.method === 'DELETE')
      ? { status: 403, body: {} } : { status: 200, body: { enabled: true, items: [item('a1')], next_cursor: null } } });
    p.w.confirmDestructive = () => Promise.resolve(true);
    p.d.querySelector('.ac-clear').click();
    await flushPromises(); await flushPromises();
    assert.equal(p.d.querySelector('.ac-clear'), null);
    assert.match(text(p, '#ac-feed'), /Not allowed for your role/);
  });
});

describe('Answer Check tab — behaviour', function () {
  it('polling stops when the pane loses .active', async function () {
    const p = await render();
    assert.ok(p.timers.length >= 2 && p.timers.every(t => !t.cleared));
    p.pane.classList.remove('active');
    const before = p.calls.length;
    p.timers.forEach(t => t.fn());
    await flushPromises();
    assert.equal(p.calls.length, before, 'no request while hidden');
    assert.ok(p.timers.every(t => t.cleared));
  });

  it('polling restarts when the pane becomes visible again', async function () {
    const p = await render();
    p.pane.classList.remove('active');
    p.timers.forEach(t => t.fn());
    const n = p.timers.length;
    p.pane.classList.add('active');
    await flushPromises();
    assert.ok(p.timers.length > n, 'new timers after re-activation');
  });

  it('"Change in Settings" opens Settings and focuses the answer-check group', async function () {
    const p = await render();
    const link = p.d.createElement('a');
    link.className = 'nav-link';
    link.dataset.tab = 'settings-pane';
    let clicked = 0;
    link.addEventListener('click', () => {
      clicked += 1;
      const grp = p.d.createElement('div');
      grp.id = 'settings-answer-check';
      grp.innerHTML = '<h3>Answer check</h3>';
      p.d.body.appendChild(grp);
    });
    p.d.body.appendChild(link);
    p.d.querySelector('.ac-settings-link').click();
    await flushPromises();
    assert.equal(clicked, 1);
    assert.equal(p.d.activeElement && p.d.activeElement.textContent, 'Answer check');
  });

  it('a re-render tears down the previous controller', async function () {
    const p = await render();
    const first = p.w.SLMAnswerCheckTab.controller();
    p.w.odRenderAnswerCheck(p.pane);
    await flushPromises();
    assert.notEqual(p.w.SLMAnswerCheckTab.controller(), first);
    assert.equal(first.stopped, true);
  });
});
