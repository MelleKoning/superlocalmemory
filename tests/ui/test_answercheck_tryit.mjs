/**
 * tests/ui/test_answercheck_tryit.mjs — the Answer Check tab's "Try it" panel.
 * Runner: npm test   (scripts/run-ui-tests.mjs). Requires jsdom.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function block(over = {}) {
  return Object.assign({ status: 'judged', detail: '', judge: 'laya', abstained: false,
    abstention_reason: null, answer_confidence: 0.81, threshold: 0.5, reordered: false,
    retrieval_ms: 812.3, judge_ms: 201, total_ms: 1013.3, ceiling_ms: 3000 }, over);
}

function results(n) {
  return Array.from({ length: n }, (_, i) => ({ content: '<b>Project Kestrel</b> memory ' + i,
    score: 0.9, relevance_score: 0.8 - i * 0.1, memory_kind_label: 'Fact' }));
}

function mount(reply, statusObj) {
  const h = buildHarness(['tryit'], { ok: true, status: 200, json: {} });
  const w = h.window;
  const sent = [];
  w.fetch = function (url, init) {
    sent.push({ url: String(url), init });
    return Promise.resolve({ ok: reply.status === 200, status: reply.status,
                             json: () => Promise.resolve(reply.body) });
  };
  evalModule(w, 'od-answercheck-tryit.js');
  const root = w.document.getElementById('tryit');
  const api = w.SLMAnswerCheckTryIt.mount(root, () => statusObj || { active: 'laya' });
  return { w, d: w.document, root, api, sent };
}

async function ask(p, q = 'When does Project Kestrel launch?') {
  p.d.getElementById('ac-tryit-q').value = q;
  p.d.querySelector('.ac-tryit-form').dispatchEvent(new p.w.Event('submit', { cancelable: true }));
  await flushPromises(); await flushPromises();
}

const label = (p) => p.d.querySelector('.ac-verdict-label');

describe('Try it', function () {
  it('abstained: the product saying "I don\'t have that", in violet, with the numbers', async function () {
    const p = mount({ status: 200, body: { result_count: 7, results: results(7),
      answer_check: block({ abstained: true, answer_confidence: 0.18, abstention_reason: 'judged_insufficient' }) } });
    await ask(p, 'What is Project Kestrel\'s marketing budget?');
    assert.equal(label(p).dataset.outcome, 'abstained');
    assert.match(label(p).textContent, /I don’t have that/);
    assert.match(label(p).getAttribute('style'), /var\(--violet\)/);
    assert.match(p.root.textContent, /Confidence 0\.18 — needs 0\.50 to count as an answer/);
    assert.equal(p.d.querySelectorAll('.ac-found li').length, 3);
    assert.match(p.root.textContent, /\+4 more/);
    assert.match(p.d.querySelector('.ac-stack').getAttribute('aria-label'), /total 1,013 ms of the 3,000 ms limit/);
  });

  it('answered', async function () {
    const p = mount({ status: 200, body: { result_count: 2, results: results(2), answer_check: block() } });
    await ask(p);
    assert.equal(label(p).dataset.outcome, 'answered');
    assert.match(p.root.textContent, /Judge: Laya, on this Mac/);
  });

  it('not checked', async function () {
    const p = mount({ status: 200, body: { result_count: 2, results: results(2),
      answer_check: block({ status: 'off', judge: '', answer_confidence: null, threshold: null }) } });
    await ask(p);
    assert.match(label(p).textContent, /Not checked — answer check is off/);
  });

  it('zero results', async function () {
    const p = mount({ status: 200, body: { result_count: 0, results: [],
      answer_check: block({ status: 'skipped', detail: 'no_results', abstained: true, answer_confidence: null }) } });
    await ask(p);
    assert.match(p.root.textContent, /No memories matched\./);
    assert.equal(label(p).dataset.outcome, 'nothing_found');
  });

  it('memory text is text, never markup', async function () {
    const p = mount({ status: 200, body: { result_count: 1, results: results(1), answer_check: block() } });
    await ask(p);
    assert.equal(p.d.querySelector('.ac-found b'), null);
    assert.match(p.d.querySelector('.ac-snippet').textContent, /<b>Project Kestrel<\/b>/);
  });

  it('posts to the Recall Lab route without invalidating cached panes', async function () {
    const p = mount({ status: 200, body: { result_count: 0, results: [], answer_check: block() } });
    await ask(p);
    assert.equal(p.sent[0].url, '/api/v3/recall/trace');
    assert.equal(p.sent[0].init.method, 'POST');
    assert.deepEqual(JSON.parse(p.sent[0].init.body), { query: 'When does Project Kestrel launch?', limit: 10 });
    assert.equal(p.sent[0].init.slmInvalidatesCache, false);
  });

  it('the Jev notice appears only when Jev runs', function () {
    const laya = mount({ status: 200, body: {} }, { active: 'laya' });
    assert.equal(laya.d.querySelector('.ac-jev-notice').hidden, true);
    const jev = mount({ status: 200, body: {} }, { active: 'jev', jev: { provider: 'openrouter' } });
    const n = jev.d.querySelector('.ac-jev-notice');
    assert.equal(n.hidden, false);
    assert.equal(n.textContent, 'Asking sends this question and the top memories to OpenRouter.');
  });

  it('focus lands on the verdict heading', async function () {
    const p = mount({ status: 200, body: { result_count: 1, results: results(1), answer_check: block() } });
    await ask(p);
    assert.equal(p.d.activeElement, label(p));
    assert.equal(p.d.querySelector('.ac-verdict').getAttribute('aria-live'), 'polite');
  });

  it('403 says the role cannot search', async function () {
    const p = mount({ status: 403, body: { detail: 'no' } });
    await ask(p);
    assert.match(p.root.textContent, /Your role cannot search this workspace\./);
  });

  it('500 asks to try again and re-enables the button', async function () {
    const p = mount({ status: 500, body: { error: 'x' } });
    await ask(p);
    assert.match(p.root.textContent, /Try again in a moment/);
    assert.equal(p.d.querySelector('.ac-tryit-form button').disabled, false);
  });

  it('outcome mapping matches the server table', function () {
    const p = mount({ status: 200, body: {} });
    const k = p.w.SLMAnswerCheckTryIt.outcomeKey;
    assert.equal(k('judged', '', true, 3), 'abstained');
    assert.equal(k('judged', '', false, 3), 'answered');
    assert.equal(k('skipped', 'no_results', true, 0), 'nothing_found');
    assert.equal(k('skipped', 'budget', false, 3), 'not_checked_time');
    assert.equal(k('skipped', 'other_profile_memory', false, 3), 'not_checked_shared');
    assert.equal(k('busy', '', false, 3), 'not_checked_busy');
    assert.equal(k('warming', '', false, 3), 'not_checked_loading');
    assert.equal(k('unavailable', '', false, 3), 'not_checked_unavailable');
    assert.equal(k('off', '', false, 3), 'not_checked_off');
  });
});
