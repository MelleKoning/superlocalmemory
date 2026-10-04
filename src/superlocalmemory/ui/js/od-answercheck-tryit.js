// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
// Part of SuperLocalMemory | https://qualixar.com
//
// Answer Check tab — "Try it" panel (4.1.20).
//
// Ask one question and see, side by side, what retrieval found and what the
// answer check said about it. Uses POST /api/v3/recall/trace (the Recall Lab
// route), whose `answer_check` block carries the verdict and the timings.
// The daemon tags these recalls as dashboard tests, so trying the tab never
// moves the abstention rate the tab reports.
//
// DOM is built with createElement + textContent only: memory text is user
// data and is never parsed as HTML.

(function () {
  'use strict';

  var CEILING_MS = 3000;
  var SNIPPET = 160;

  var OUTCOME_TEXT = {
    answered: 'Answered',
    abstained: 'Said: “I don’t have that”',
    nothing_found: 'Nothing found',
    not_checked_off: 'Not checked — answer check is off',
    not_checked_time: 'Not checked — out of time',
    not_checked_shared: 'Not checked — shared memory stays on this machine',
    not_checked_busy: 'Not checked — judge busy',
    not_checked_loading: 'Not checked — judge loading',
    not_checked_unavailable: 'Not checked — judge unavailable',
  };
  var OUTCOME_TONE = { answered: 'ok', abstained: 'violet', nothing_found: 'neutral' };
  var SKIP_DETAIL = { no_results: 'nothing_found', budget: 'not_checked_time',
                      other_profile_memory: 'not_checked_shared' };
  var STATUS_OUTCOME = { off: 'not_checked_off', busy: 'not_checked_busy',
                         warming: 'not_checked_loading', unavailable: 'not_checked_unavailable' };

  // The same mapping as core/answer_check_history_stats.outcome_key.
  function outcomeKey(status, detail, abstained, resultCount) {
    if (status === 'judged') return abstained ? 'abstained' : 'answered';
    if (status === 'skipped') {
      if (SKIP_DETAIL[detail]) return SKIP_DETAIL[detail];
      return Number(resultCount || 0) === 0 ? 'nothing_found' : 'not_checked_unavailable';
    }
    return STATUS_OUTCOME[status] || 'not_checked_unavailable';
  }

  function el(tag, props, children) {
    var node = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      var v = props[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') node.textContent = String(v);
      else if (k === 'class') node.className = v;
      else if (k === 'style') node.setAttribute('style', v);
      else node.setAttribute(k, v === true ? '' : String(v));
    });
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function ms(v) {
    return (typeof v === 'number' && isFinite(v)) ? Math.round(v).toLocaleString('en-US') + ' ms' : '—';
  }

  function judgeName(judge) {
    if (judge === 'laya') return 'Laya, on this Mac';
    if (judge === 'jev') return 'Jev, online';
    return 'No judge';
  }

  function providerName(status) {
    var p = status && status.jev && status.jev.provider;
    return p === 'openrouter' ? 'OpenRouter' : 'TypeSafe';
  }

  function latencyBar(block) {
    var r = typeof block.retrieval_ms === 'number' ? block.retrieval_ms : 0;
    var j = typeof block.judge_ms === 'number' ? block.judge_ms : 0;
    var total = typeof block.total_ms === 'number' ? block.total_ms : r + j;
    var scale = Math.max(CEILING_MS, total);
    var label = 'Retrieval ' + ms(r) + ', answer check ' + ms(j) + ', total ' + ms(total) +
      ' of the ' + CEILING_MS.toLocaleString('en-US') + ' ms limit';
    var bar = el('div', { class: 'ac-stack', role: 'img', 'aria-label': label,
      style: 'position:relative;height:12px;border-radius:6px;background:var(--card-2);' +
             'border:1px solid var(--border);overflow:hidden;margin:8px 0 4px' }, [
      el('span', { style: 'position:absolute;left:0;top:0;bottom:0;background:var(--violet-line);' +
                          'width:' + (100 * r / scale).toFixed(2) + '%' }),
      el('span', { style: 'position:absolute;top:0;bottom:0;background:var(--violet);' +
                          'left:' + (100 * r / scale).toFixed(2) + '%;width:' + (100 * j / scale).toFixed(2) + '%' }),
      el('span', { 'aria-hidden': 'true', style: 'position:absolute;top:-2px;bottom:-2px;width:2px;' +
                          'background:var(--danger);left:' + (100 * CEILING_MS / scale).toFixed(2) + '%' }),
    ]);
    return el('div', {}, [bar, el('div', { class: 'ac-stack-legend', style: 'font-size:12px;color:var(--fg-2)',
      text: label })]);
  }

  function renderFound(col, body) {
    col.textContent = '';
    col.appendChild(el('h4', { style: 'font-size:13px;margin:0 0 8px', text: 'What retrieval found' }));
    var results = Array.isArray(body.results) ? body.results : [];
    if (!results.length) {
      col.appendChild(el('p', { class: 'ac-empty', style: 'font-size:13px;color:var(--fg-2)',
                                text: 'No memories matched.' }));
      return;
    }
    var list = el('ol', { style: 'padding-left:18px;margin:0;font-size:13px;line-height:1.5' });
    results.slice(0, 3).forEach(function (r) {
      var rel = typeof r.relevance_score === 'number' ? r.relevance_score : r.score;
      var score = typeof rel === 'number' ? Math.round(rel * 100) + '% relevant' : '';
      var text = String(r.content || '');
      if (text.length > SNIPPET) text = text.slice(0, SNIPPET) + '…';
      list.appendChild(el('li', { style: 'margin-bottom:6px' }, [
        el('span', { class: 'badge neutral', style: 'margin-right:6px', text: score || 'match' }),
        (r.memory_kind_label || r.memory_kind) ? el('span', { class: 'badge cyan', style: 'margin-right:6px',
          text: String(r.memory_kind_label || r.memory_kind) }) : null,
        el('span', { class: 'ac-snippet', text: text }),
      ]));
    });
    col.appendChild(list);
    if (results.length > 3) {
      col.appendChild(el('p', { style: 'font-size:12px;color:var(--fg-2);margin:4px 0 0',
                                text: '+' + (results.length - 3) + ' more' }));
    }
  }

  function renderVerdict(col, body) {
    var block = body.answer_check || {};
    var key = outcomeKey(block.status, block.detail, block.abstained, body.result_count);
    col.textContent = '';
    col.appendChild(el('h4', { style: 'font-size:13px;margin:0 0 8px', text: 'What the answer check said' }));
    var tone = OUTCOME_TONE[key] || 'warn';
    var heading = el('p', { class: 'ac-verdict-label', tabindex: '-1', 'data-outcome': key,
      style: 'font-size:20px;font-weight:700;margin:0 0 6px;color:var(--' +
             (tone === 'neutral' ? 'fg' : tone) + ')', text: OUTCOME_TEXT[key] });
    col.appendChild(heading);
    // Why it was not checked, in the server's own words (answer_check_note):
    // a recall that was not checked must never read like one that was.
    if (typeof body.answer_check_note === 'string' && body.answer_check_note) {
      col.appendChild(el('p', { class: 'ac-note', style: 'font-size:13px;margin:0 0 6px',
                                text: body.answer_check_note }));
    }
    if (block.detail === 'reused') {
      col.appendChild(el('p', { class: 'ac-reused', style: 'font-size:12.5px;color:var(--fg-2);margin:0 0 4px',
        text: 'Same question, same memories: the verdict from the earlier check was reused.' }));
    }
    if (typeof block.answer_confidence === 'number') {
      var line = 'Confidence ' + block.answer_confidence.toFixed(2);
      if (typeof block.threshold === 'number') {
        line += ' — needs ' + block.threshold.toFixed(2) + ' to count as an answer';
      }
      col.appendChild(el('p', { class: 'ac-confidence', style: 'font-size:13px;margin:0 0 4px', text: line }));
    }
    col.appendChild(el('p', { style: 'font-size:12.5px;color:var(--fg-2);margin:0 0 4px',
                              text: 'Judge: ' + judgeName(block.judge) }));
    if (block.reordered) {
      col.appendChild(el('p', { style: 'font-size:12.5px;margin:0 0 4px', text: 'Reordered by Jev' }));
    }
    col.appendChild(latencyBar(block));
    if (typeof body.retrieval_time_ms === 'number') {
      col.appendChild(el('p', { style: 'font-size:11.5px;color:var(--fg-3);margin:2px 0 0',
        text: 'Round trip including the request: ' + ms(body.retrieval_time_ms) }));
    }
    heading.focus();
  }

  function message(col, text) {
    col.textContent = '';
    col.appendChild(el('p', { class: 'ac-tryit-msg', style: 'font-size:13px', text: text }));
  }

  function mount(root, getStatus) {
    if (!root) return null;
    root.textContent = '';
    var input = el('input', { id: 'ac-tryit-q', type: 'text', maxlength: '500', autocomplete: 'off',
      class: 'form-control', style: 'flex:1;min-width:220px;height:38px;padding:0 12px;' +
      'border-radius:var(--r-md);border:1px solid var(--border);background:var(--card);color:var(--fg)' });
    var button = el('button', { type: 'submit', class: 'btn primary', text: 'Ask' });
    var notice = el('p', { class: 'ac-jev-notice', role: 'note', hidden: true,
                           style: 'font-size:12.5px;color:var(--warn);margin:6px 0 0' });
    var form = el('form', { class: 'ac-tryit-form', novalidate: true }, [
      el('label', { for: 'ac-tryit-q', style: 'display:block;font-size:13px;font-weight:600;margin-bottom:6px',
                    text: 'Ask your memory a question' }),
      el('div', { style: 'display:flex;gap:8px;flex-wrap:wrap' }, [input, button]),
      notice,
    ]);
    var found = el('div', { class: 'ac-found' });
    var verdict = el('div', { class: 'ac-verdict', role: 'status', 'aria-live': 'polite' });
    var cols = el('div', { class: 'ac-tryit-cols', hidden: true,
      style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:16px;margin-top:14px' },
      [found, verdict]);
    root.appendChild(form);
    root.appendChild(cols);

    function refreshNotice() {
      var status = typeof getStatus === 'function' ? getStatus() : null;
      if (status && status.active === 'jev') {
        notice.textContent = 'Asking sends this question and the top memories to ' + providerName(status) + '.';
        notice.hidden = false;
      } else {
        notice.textContent = '';
        notice.hidden = true;
      }
    }
    refreshNotice();

    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var query = String(input.value || '').trim();
      if (!query) { input.focus(); return; }
      refreshNotice();
      button.disabled = true;
      root.setAttribute('aria-busy', 'true');
      cols.hidden = false;
      message(found, 'Searching…');
      message(verdict, 'Checking…');
      fetch('/api/v3/recall/trace', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, limit: 10 }),
        slmInvalidatesCache: false,
      }).then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          return { status: res.status, ok: res.ok, body: body };
        });
      }).then(function (r) {
        if (r.status === 401 || r.status === 403) {
          message(found, 'Your role cannot search this workspace.');
          verdict.textContent = '';
          return;
        }
        if (!r.ok || !r.body || r.body.error) {
          message(found, 'That did not work. Try again in a moment.');
          verdict.textContent = '';
          return;
        }
        renderFound(found, r.body);
        renderVerdict(verdict, r.body);
        if (typeof root.onAsked === 'function') root.onAsked();
      }).catch(function () {
        message(found, 'That did not work. Try again in a moment.');
        verdict.textContent = '';
      }).then(function () {
        button.disabled = false;
        root.removeAttribute('aria-busy');
      });
    });
    return { refreshNotice: refreshNotice };
  }

  window.SLMAnswerCheckTryIt = {
    mount: mount,
    outcomeKey: outcomeKey,
    OUTCOME_TEXT: OUTCOME_TEXT,
    OUTCOME_TONE: OUTCOME_TONE,
  };
}());
