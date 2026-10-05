/**
 * tests/ui/test_saved_views_and_summaries_panes.mjs — issue #113 panes.
 * Runner: npm test   (scripts/run-ui-tests.mjs)
 *
 * Saved views (od-views.js): a non-technical user sees what a view is and what
 * to do when there are none, can save one from a form, run it, and every result
 * shows the id of the memory it came from — openable in place.
 *
 * Summaries (od-summaries.js): "Today" is the browser's today and travels with
 * the browser's offset; the session picker is filled; the memories a summary
 * came from are listed by id.
 *
 * Every value from the API is set as text: a memory containing markup must
 * render as text, never as HTML.
 */

// Must be set before any Date is created in this process.
process.env.TZ = 'Asia/Kolkata';

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

const EVIL = '<img src=x onerror="window.__pwned=1">';

function harness(routes) {
    const h = buildHarness(['root'], { ok: true, status: 200, json: {} });
    const calls = [];
    h.window.fetch = function (url, init) {
        const method = (init && init.method) || 'GET';
        calls.push({ url: String(url), method, body: init && init.body });
        const key = Object.keys(routes).find(k => String(url).startsWith(k.split(' ').pop())
            && (k.indexOf(' ') < 0 || k.startsWith(method + ' ')));
        const answer = key ? routes[key] : { status: 404, json: { detail: 'no route' } };
        const json = typeof answer.json === 'function' ? answer.json(init) : answer.json;
        return Promise.resolve({ ok: (answer.status || 200) < 400, status: answer.status || 200,
                                 json: () => Promise.resolve(json) });
    };
    evalModule(h.window, 'od-summaries.js');
    evalModule(h.window, 'od-views.js');
    return { ...h, calls };
}

async function settle() { for (let i = 0; i < 4; i++) await flushPromises(); }

const LIMITS = { max_name_chars: 80, max_query_chars: 1000,
                 kinds: [{ value: 'decision', label: 'Decision' }] };

describe('Saved views pane', function () {
    it('says what to do when there are no views', async function () {
        const h = harness({ '/api/v3/views': { json: { views: [], limits: LIMITS } } });
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        const empty = h.document.querySelector('[data-views-empty]');
        assert.ok(empty, 'no empty state');
        assert.match(empty.textContent, /no saved views yet/i);
        assert.match(empty.textContent, /Save view/);
        assert.ok(h.document.getElementById('m-views-name'), 'create form missing');
        const kinds = [...h.document.querySelectorAll('#m-views-kind option')].map(o => o.value);
        assert.deepEqual(kinds, ['', 'decision'], 'kinds come from the server');
    });

    it('saves a view from the form with its filters', async function () {
        let posted = null;
        const h = harness({
            'POST /api/v3/views': { json: (init) => { posted = JSON.parse(init.body);
                                                      return { message: 'Saved the view.' }; } },
            'GET /api/v3/views': { json: { views: [], limits: LIMITS } },
        });
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        h.document.getElementById('m-views-name').value = 'Work log';
        h.document.getElementById('m-views-query').value = 'what did I ship';
        h.document.getElementById('m-views-window').value = '7d';
        h.document.getElementById('m-views-kind').value = 'decision';
        h.document.querySelector('[data-views-act="create"]').click();
        await settle();
        assert.deepEqual(posted, { name: 'Work log', query: 'what did I ship',
                                   filters: { window: '7d', kind: 'decision' } });
        assert.equal(h.document.getElementById('m-views-msg').textContent, 'Saved the view.');
    });

    it('refuses an empty form without calling the server', async function () {
        const h = harness({ 'GET /api/v3/views': { json: { views: [], limits: LIMITS } } });
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        h.document.querySelector('[data-views-act="create"]').click();
        await settle();
        assert.ok(!h.calls.some(c => c.method === 'POST'));
        assert.match(h.document.getElementById('m-views-msg').textContent, /name and a question/);
    });

    it('runs a view and shows every result with its memory id, as text', async function () {
        const view = { name: EVIL, query: 'q', filters: { window: '7d' }, limit: 5 };
        const h = harness({
            '/api/v3/views/run': { json: { view, count: 2, no_confident_match: false, results: [
                { rank: 1, fact_id: 'fact-b', content: EVIL, score: 0.9, age_label: '2d ago' },
                { rank: 2, fact_id: 'fact-a', content: 'second', score: 0.4 }] } },
            '/api/v3/views': { json: { views: [view], limits: LIMITS } },
        });
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        const run = h.document.querySelector('[data-views-act="run"]');
        assert.equal(run.dataset.name, EVIL);
        run.click();
        await settle();
        const runCall = h.calls.find(c => c.url.startsWith('/api/v3/views/run'));
        assert.equal(runCall.url, '/api/v3/views/run?name=' + encodeURIComponent(EVIL));
        const items = [...h.document.querySelectorAll('#m-views-out .fact-result-item')];
        assert.deepEqual(items.map(i => i.getAttribute('data-fact-id')), ['fact-b', 'fact-a']);
        assert.match(items[0].textContent, /Memory ID fact-b/);
        assert.equal(h.document.querySelector('#m-views-out img'), null, 'markup was rendered');
        assert.equal(h.window.__pwned, undefined);
    });

    it('shows the server refusal in plain words', async function () {
        const h = harness({
            'POST /api/v3/views': { status: 409, json: { detail: 'A view called "W" already exists.' } },
            'GET /api/v3/views': { json: { views: [], limits: LIMITS } },
        });
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        h.document.getElementById('m-views-name').value = 'W';
        h.document.getElementById('m-views-query').value = 'q';
        h.document.querySelector('[data-views-act="create"]').click();
        await settle();
        assert.match(h.document.getElementById('m-views-msg').textContent, /already exists/);
    });

    it('deletes only after the shared confirmation', async function () {
        const view = { name: 'W', query: 'q', filters: {}, limit: 10 };
        const h = harness({
            'POST /api/v3/views/delete': { json: { message: 'Deleted.' } },
            'GET /api/v3/views': { json: { views: [view], limits: LIMITS } },
        });
        let asked = null;
        h.window.confirmDestructive = (opts) => { asked = opts; return Promise.resolve(false); };
        h.document.getElementById('root').innerHTML = h.window.ODViews.pane('m');
        h.window.ODViews.onShow('m');
        await settle();
        h.document.querySelector('[data-views-act="delete"]').click();
        await settle();
        assert.equal(asked.target, 'W');
        assert.ok(!h.calls.some(c => c.url === '/api/v3/views/delete'), 'deleted on cancel');
        h.window.confirmDestructive = () => Promise.resolve(true);
        h.document.querySelector('[data-views-act="delete"]').click();
        await settle();
        const del = h.calls.find(c => c.url === '/api/v3/views/delete');
        assert.deepEqual(JSON.parse(del.body), { name: 'W' });
    });
});

describe('Summaries pane', function () {
    it('asks for the browser\'s own today with its offset', async function () {
        const h = harness({
            '/api/summary/projects': { json: { projects: [] } },
            '/api/summary/sessions': { json: { sessions: [] } },
            '/api/summary?': { json: { summary: 's', coverage: 'full', source_count: 0,
                                       source_fact_ids: [], metadata: {} } },
        });
        h.document.getElementById('root').innerHTML = '<div id="pane">' +
            h.window.ODSummaries.controls('m') + '</div>';
        h.window.ODSummaries.onShow('m');
        await settle();
        h.document.querySelector('[data-sum-act="day"][data-target="today"]').click();
        await settle();
        const call = h.calls.find(c => c.url.startsWith('/api/summary?'));
        const now = new Date();
        const local = now.getFullYear() + '-' + String(now.getMonth() + 1).padStart(2, '0') +
            '-' + String(now.getDate()).padStart(2, '0');
        assert.match(call.url, /tz_offset_minutes=330/);
        assert.match(call.url, new RegExp('target=' + local));
    });

    it('fills the session picker and summarises the chosen session', async function () {
        const h = harness({
            '/api/summary/projects': { json: { projects: [] } },
            '/api/summary/sessions': { json: { sessions: [
                { session_id: 'sess-1', memory_count: 3 }] } },
            '/api/summary?': { json: { summary: EVIL, coverage: 'partial', source_count: 2,
                                       source_fact_ids: ['f-1', 'f-2'], metadata: {} } },
        });
        h.document.getElementById('root').innerHTML = '<div id="pane">' +
            h.window.ODSummaries.controls('m') + '</div>';
        h.window.ODSummaries.onShow('m');
        await settle();
        const sel = h.document.getElementById('m-sum-sess');
        assert.deepEqual([...sel.options].map(o => o.value), ['', 'sess-1']);
        assert.equal(h.document.getElementById('m-sum-proj').disabled, true);
        sel.value = 'sess-1';
        sel.dispatchEvent(new h.window.Event('change', { bubbles: true }));
        await settle();
        assert.ok(h.calls.some(c => /kind=session/.test(c.url) && /target=sess-1/.test(c.url)));
        const out = h.document.getElementById('m-sum-out');
        assert.match(out.textContent, /Partial view/);
        const ids = [...out.querySelectorAll('.fact-result-item')].map(
            i => i.getAttribute('data-fact-id'));
        assert.deepEqual(ids, ['f-1', 'f-2']);
        assert.equal(out.querySelector('img'), null, 'summary text rendered as markup');
    });

    it('says when a picker could not load instead of showing an empty list', async function () {
        const h = harness({
            '/api/summary/projects': { status: 500, json: {} },
            '/api/summary/sessions': { json: { sessions: [] } },
        });
        h.document.getElementById('root').innerHTML = '<div id="pane">' +
            h.window.ODSummaries.controls('m') + '</div>';
        h.window.ODSummaries.onShow('m');
        await settle();
        assert.match(h.document.getElementById('m-sum-proj').textContent, /Could not load/);
        assert.match(h.document.getElementById('m-sum-sess').textContent,
                     /No sessions with saved memories/);
    });
});
