// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
// Part of SuperLocalMemory | https://qualixar.com
//
// Answer Check tab (4.1.20) — Jev and Laya, in one place.
//
// What runs (a status card that LINKS to Settings; no settings are duplicated
// here), a try-it panel (od-answercheck-tryit.js), how often SLM said "I don't
// have that" (only over checks that were actually judged, withheld below 20,
// with a 95% range), time against the 3-second recall limit, and a live feed of
// recent checks. The history holds outcomes and timings only — never questions
// or memories — so this pane has no text of anyone's to show.
//
// Reads: GET /api/v3/answer-check, /api/v3/answer-check/history/{live,summary},
// /api/v3/answer-check/history. Clear: DELETE /api/v3/answer-check/history.
// Polls the feed every 5 s and the summary every 30 s, only while the pane is
// visible. DOM via createElement + textContent only.

(function () {
  'use strict';

  var PANE_ID = 'answercheck-pane';
  var LIVE_MS = 5000;
  var SUMMARY_MS = 30000;
  var FEED_ROWS = 25;
  var MAX_FEED = 200;   // rows kept on screen while the live feed grows
  var CEILING_MS = 3000;
  var API = '/api/v3/answer-check';
  // The only item fields this pane ever reads. A server change that adds a
  // field (or, by mistake, text) to an item renders nothing extra.
  var ITEM_KEYS = ['id', 'at', 'outcome', 'judge', 'origin', 'answer_confidence',
                   'result_count', 'total_ms', 'over_ceiling'];

  var ctl = null; // the one live controller; replaced on every render

  function tryit() { return window.SLMAnswerCheckTryIt || {}; }
  function outcomeText(key) {
    var t = tryit().OUTCOME_TEXT || {};
    return t[key] || 'Not checked';
  }
  function outcomeTone(key) {
    var t = tryit().OUTCOME_TONE || {};
    return t[key] || 'warn';
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

  function card(id, title, sub) {
    return el('section', { class: 'card', id: id, style: 'margin-bottom:16px', 'aria-labelledby': id + '-h' }, [
      el('div', { class: 'card-head' }, [
        el('h3', { id: id + '-h', text: title }),
        sub ? el('span', { class: 'sub', text: sub }) : null,
      ]),
      el('div', { class: 'card-pad', 'data-body': '', id: id + '-body' }),
    ]);
  }
  function body(section) { return section.querySelector('[data-body]'); }
  function fill(node, children) {
    node.textContent = '';
    children.forEach(function (c) { if (c) node.appendChild(c); });
  }
  function note(text, extra) {
    return el('p', { class: 'ac-note', style: 'font-size:13px;color:var(--fg-2);margin:0' + (extra || ''), text: text });
  }
  function fmtMs(v) {
    return (typeof v === 'number' && isFinite(v)) ? Math.round(v).toLocaleString('en-US') + ' ms' : '—';
  }
  function pct(v) { return (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%'; }

  function getJSON(url, init) {
    return fetch(url, Object.assign({ credentials: 'same-origin' }, init || {})).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        return { status: res.status, ok: res.ok, body: data };
      });
    });
  }

  // ── status card ──────────────────────────────────────────────────────────
  function statusLines(s) {
    var jev = s.jev || {};
    var laya = s.laya || {};
    var provider = jev.provider === 'openrouter' ? 'OpenRouter' : 'TypeSafe';
    var summary = typeof window.answerCheckSummaryText === 'function' ? window.answerCheckSummaryText(s) : null;
    var lines = [];
    if (s.active === 'laya') lines.push('On this Mac — memory text never leaves it.');
    if (s.active === 'jev') lines.push('Online via ' + provider + ' — the question and the top memories are sent there.');
    if (s.mode === 'jev' && !jev.consent) {
      lines.push('Jev is chosen, but consent was not given, so nothing is checked and nothing is sent.');
    } else if (s.mode === 'jev' && jev.key_problem) {
      lines.push(String(jev.key_problem));
    } else if (s.mode === 'jev' && !jev.has_key) {
      lines.push('Jev is chosen, but no key is saved yet, so nothing is checked.');
    }
    if (s.apple_silicon === false && s.mode !== 'jev') {
      lines.push('On-device checking needs Apple Silicon. Online checking with Jev is available in Settings.');
    }
    if (laya.state === 'installing') lines.push('Setting up the on-device model…');
    if (laya.state === 'failed') {
      lines.push('The on-device check needs attention: ' + String(laya.error || laya.step || 'it is not ready') +
        ' Fix it in Settings.');
    }
    if (s.jev && s.jev.rerank && s.jev.rerank.enabled && s.active !== 'jev') {
      lines.push('Reordering with Jev is off while Jev isn’t chosen; it comes back when you choose Jev.');
    }
    if (s.mode === 'jev') lines.push('Consent: ' + (jev.consent ? 'given' : 'not given'));
    if (s.active === 'jev') lines.push('Reordering: ' + (jev.rerank && jev.rerank.active ? 'on' : 'off'));
    var headline = summary || (s.active === 'off' || !s.active ? 'Off' : String(s.active));
    if (s.active === 'off' && laya.state === 'not_installed' && !jev.has_key) {
      headline = 'Answer check is not set up';
    }
    return { headline: headline, lines: lines };
  }

  function goToSettings() {
    var link = document.querySelector('.nav-link[data-tab="settings-pane"]');
    if (link) link.click();
    var started = Date.now();
    (function look() {
      var target = document.getElementById('settings-answer-check');
      if (target) {
        if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
        var head = target.querySelector('h3, h4, [role="heading"]') || target;
        if (!head.hasAttribute('tabindex')) head.setAttribute('tabindex', '-1');
        head.focus();
        return;
      }
      if (Date.now() - started < 2000) setTimeout(look, 100);
    }());
  }

  function renderStatus(c, s) {
    var b = body(c.sections.status);
    if (!s || !s.laya) {
      fill(b, [note('Could not read which answer check runs.'), settingsButton()]);
      return;
    }
    var info = statusLines(s);
    fill(b, [
      el('p', { class: 'ac-status-headline', style: 'font-size:16px;font-weight:650;margin:0 0 6px', text: info.headline }),
      info.lines.length ? el('ul', { class: 'ac-status-lines', style: 'margin:0 0 12px;padding-left:18px;font-size:13px' },
        info.lines.map(function (t) { return el('li', { text: t }); })) : null,
      settingsButton(),
    ]);
  }
  function settingsButton() {
    var btn = el('button', { type: 'button', class: 'btn sm ac-settings-link', text: 'Change in Settings' });
    btn.addEventListener('click', goToSettings);
    return btn;
  }

  // ── stats ──────────────────────────────────────────────────────────────────
  function tile(label, value, sub, extra) {
    return el('div', { class: 'kpi ac-tile', style: 'position:relative' }, [
      el('div', { class: 'label', text: label }),
      el('div', { class: 'value', text: value }),
      sub ? el('div', { style: 'font-size:12px;color:var(--fg-2);margin-top:4px', text: sub }) : null,
      extra || null,
    ]);
  }

  function controls(c) {
    var seg = el('div', { class: 'seg ac-window', role: 'group', 'aria-label': 'Time window' });
    ['24h', '7d', '30d'].forEach(function (w) {
      var b = el('button', { type: 'button', 'data-window': w, 'aria-pressed': String(c.window === w),
                             class: c.window === w ? 'active' : '', text: w });
      b.addEventListener('click', function () { c.window = w; loadSummary(c); });
      seg.appendChild(b);
    });
    var box = el('input', { type: 'checkbox', id: 'ac-include-dashboard', class: 'ac-include-dashboard' });
    box.checked = c.includeDashboard;
    box.addEventListener('change', function () { c.includeDashboard = box.checked; loadSummary(c); });
    return el('div', { style: 'display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:14px' }, [
      seg, el('label', { for: 'ac-include-dashboard', style: 'font-size:12.5px;display:flex;gap:6px;align-items:center' },
        [box, 'Include dashboard tests'])]);
  }

  function renderStats(c, s) {
    var b = body(c.sections.stats);
    var counts = s.counts || {};
    var ab = s.abstention || {};
    if (!counts.total) {
      var empty = note('No checks yet. Ask a question above, or let your agent recall something.');
      empty.classList.add('ac-empty');
      fill(b, [controls(c), empty]);
      return;
    }
    var rateValue = ab.enough && typeof ab.rate === 'number' ? pct(ab.rate) : '—';
    var rateSub = ab.enough
      ? 'of ' + ab.n + ' checked answers' + (ab.ci95 ? ' · 95% range ' + pct(ab.ci95[0]) + '–' + pct(ab.ci95[1]) : '')
      : 'Not enough checks yet (' + (ab.n || 0) + ' of ' + (ab.min_sample || 20) + ' needed)';
    var nc = counts.not_checked || {};
    var ncTotal = Object.keys(nc).reduce(function (a, k) { return a + (nc[k] || 0); }, 0);
    var NC_LABEL = { off: 'answer check off', time: 'out of time', shared: 'shared memory', busy: 'judge busy',
                     loading: 'judge loading', unavailable: 'judge unavailable' };
    var breakdown = el('details', { style: 'font-size:12px;margin-top:6px' }, [
      el('summary', { text: 'Why' }),
      el('ul', { style: 'margin:4px 0 0;padding-left:16px' }, Object.keys(NC_LABEL).map(function (k) {
        return el('li', { text: NC_LABEL[k] + ': ' + (nc[k] || 0) });
      })),
    ]);
    fill(b, [
      controls(c),
      el('div', { class: 'kpi-strip ac-tiles', style: 'grid-template-columns:repeat(auto-fit,minmax(150px,1fr))' }, [
        tile('Checks', String(counts.total)),
        tile('Answered', String(counts.answered || 0)),
        tile('Said “I don’t have that”', rateValue, rateSub),
        tile('Nothing found', String(counts.nothing_found || 0)),
        tile('Not checked', String(ncTotal), null, breakdown),
      ]),
    ]);
  }

  function bar(label, value) {
    var v = typeof value === 'number' ? value : null;
    var width = v === null ? 0 : Math.min(100, 100 * v / CEILING_MS);
    var aria = label + ': ' + (v === null ? 'no data' : fmtMs(v)) + ' of the 3,000 ms limit';
    return el('div', { class: 'ac-bar-row', style: 'display:grid;grid-template-columns:120px 1fr 80px;gap:10px;align-items:center;margin:6px 0' }, [
      el('span', { style: 'font-size:12.5px', text: label }),
      el('div', { role: 'img', 'aria-label': aria, class: 'ac-bar',
        style: 'position:relative;height:10px;border-radius:5px;background:var(--card-2);border:1px solid var(--border)' }, [
        el('span', { style: 'position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:' +
          (v !== null && v > CEILING_MS ? 'var(--danger)' : 'var(--violet)') + ';width:' + width.toFixed(1) + '%' }),
        el('span', { 'aria-hidden': 'true', style: 'position:absolute;right:0;top:-3px;bottom:-3px;width:2px;background:var(--danger)' }),
      ]),
      el('span', { style: 'font-size:12.5px;font-variant-numeric:tabular-nums;text-align:right', text: fmtMs(v) }),
    ]);
  }

  function renderLatency(c, s) {
    var b = body(c.sections.latency);
    var lat = s.latency || {};
    var total = lat.total || {};
    var judge = lat.judge || {};
    var spark = null;
    var recent = lat.recent_total_ms || [];
    if (recent.length >= 2 && typeof window.slmSpark === 'function') {
      spark = el('div', { class: 'ac-spark', 'aria-hidden': 'true', style: 'margin-top:8px' });
      spark.innerHTML = window.slmSpark(recent.map(Number), { w: 320, h: 48 }); // numbers only
    }
    fill(b, [
      note('Time inside the recall pipeline — what the 3-second limit measures.', ';margin-bottom:8px'),
      bar('Total, typical (p50)', total.p50),
      bar('Total, slow (p95)', total.p95),
      bar('Check, typical (p50)', judge.p50),
      bar('Check, slow (p95)', judge.p95),
      el('p', { class: 'ac-over', style: 'font-size:12.5px;margin:8px 0 0', text: 'Over the limit: ' + (lat.over_ceiling || 0) }),
      spark,
    ]);
  }

  // ── feed ─────────────────────────────────────────────────────────────────
  function pick(item) {
    var out = {};
    ITEM_KEYS.forEach(function (k) { out[k] = item[k]; });
    return out;
  }

  function row(item) {
    var it = pick(item);
    var when = it.at ? new Date(it.at) : null;
    var tone = outcomeTone(it.outcome);
    return el('tr', { 'data-id': it.id }, [
      el('td', { text: when && !isNaN(when) ? when.toLocaleString() : '—' }),
      el('td', {}, [
        el('span', { class: 'badge ' + tone, text: outcomeText(it.outcome) }),
        it.origin === 'dashboard' ? el('span', { class: 'badge neutral', style: 'margin-left:6px', text: 'Dashboard test' }) : null,
      ]),
      el('td', { text: it.judge === 'laya' ? 'Laya' : it.judge === 'jev' ? 'Jev' : '—' }),
      el('td', { text: typeof it.answer_confidence === 'number' ? it.answer_confidence.toFixed(2) : '—' }),
      el('td', { text: String(it.result_count || 0) }),
      el('td', { text: fmtMs(it.total_ms) + (it.over_ceiling ? ' (over the limit)' : '') }),
    ]);
  }

  function renderFeed(c) {
    var b = body(c.sections.feed);
    var table = el('table', { class: 'table ac-feed', style: 'width:100%;font-size:12.5px;border-collapse:collapse' }, [
      el('caption', { style: 'text-align:left;font-size:12px;color:var(--fg-2);caption-side:top', text: 'Recent checks, newest first' }),
      el('thead', {}, [el('tr', {}, ['When', 'Outcome', 'Judge', 'Confidence', 'Results', 'Time'].map(function (h) {
        return el('th', { scope: 'col', style: 'text-align:left;padding:4px 6px', text: h });
      }))]),
      el('tbody', {}, c.feed.map(row)),
    ]);
    var older = null;
    if (c.nextCursor) {
      older = el('button', { type: 'button', class: 'btn sm ghost ac-older', text: 'Load older' });
      older.addEventListener('click', function () { loadOlder(c); });
    }
    fill(b, [
      c.feed.length ? table : note('No checks yet. Ask a question above, or let your agent recall something.'),
      older,
      el('div', { class: 'ac-footnote', style: 'margin-top:10px' }),
      clearControl(c),
    ]);
    renderFootnote(c);
  }

  function renderFootnote(c) {
    var foot = c.sections.feed.querySelector('.ac-footnote');
    var rec = c.recording;
    if (!foot || !rec) return;
    var since = rec.since_start || {};
    var lost = (since.dropped_before_save || 0);
    fill(foot, [
      note('Kept on this machine for ' + rec.retention_days + ' days. Saved ' + (since.saved || 0) +
           ' since the daemon started. Questions and memories are never kept here.'),
      (lost > 0 || since.save_failures > 0)
        ? el('p', { class: 'ac-not-saved', style: 'font-size:12.5px;color:var(--warn);margin:4px 0 0',
            text: (lost || since.save_failures) + ' checks were not saved (the history was busy). Your recalls were not affected.' })
        : null,
      (since.erased_unsaved > 0)
        ? el('p', { class: 'ac-erased', style: 'font-size:12.5px;color:var(--fg-2);margin:4px 0 0',
            text: since.erased_unsaved + ' checks were not saved because their history was erased first.' })
        : null,
    ]);
  }

  function clearControl(c) {
    if (c.clearDenied) {
      return el('p', { class: 'ac-clear-denied', style: 'font-size:12.5px;color:var(--fg-2);margin:8px 0 0', text: 'Not allowed for your role' });
    }
    var btn = el('button', { type: 'button', class: 'btn sm ac-clear', style: 'margin-top:8px', text: 'Clear history' });
    btn.addEventListener('click', function () { clearHistory(c); });
    return btn;
  }

  function clearHistory(c) {
    var ask = typeof window.confirmDestructive === 'function'
      ? window.confirmDestructive({ title: 'Clear answer-check history', target: 'This workspace’s answer-check history',
          consequence: 'Removes the saved outcomes and timings of recent checks. Your memories are not touched.',
          confirmationText: 'CLEAR', confirmLabel: 'Clear history' })
      : Promise.resolve(window.confirm('Clear this workspace’s answer-check history?'));
    return ask.then(function (yes) {
      if (!yes) return;
      return getJSON(API + '/history', { method: 'DELETE' }).then(function (r) {
        if (r.status === 401 || r.status === 403) { c.clearDenied = true; renderFeed(c); return; }
        c.feed = []; c.nextCursor = null; loadAll(c);
      });
    });
  }

  function announce(c, n) {
    if (n > 0) c.live.textContent = n === 1 ? '1 new check' : n + ' new checks';
  }

  function mergeFeed(c, items, prepend) {
    var seen = {};
    c.feed.forEach(function (it) { seen[it.id] = true; });
    var fresh = items.filter(function (it) { return it && it.id && !seen[it.id]; });
    c.feed = prepend ? fresh.concat(c.feed) : c.feed.concat(fresh);
    if (prepend && c.feed.length > MAX_FEED) c.feed = c.feed.slice(0, MAX_FEED);
    return fresh.length;
  }

  // ── states ────────────────────────────────────────────────────────────────
  function noAccess(c) {
    c.stopped = true;
    stopTimers(c);
    fill(c.lower, [el('div', { class: 'card card-pad ac-no-access', role: 'status' }, [
      note('Your role cannot view answer-check activity in this workspace.')])]);
  }
  function disabled(c) {
    ['stats', 'latency', 'feed'].forEach(function (k) {
      fill(body(c.sections[k]), [note('History is turned off on this machine.')]);
    });
  }
  // The dashboard's shared pane-error convention (core.js showPaneError).
  function failed(c, key, retry, status) {
    var target = body(c.sections[key]);
    if (typeof window.showPaneError === 'function' && typeof window.paneErrorMessage === 'function') {
      target.textContent = '';
      window.showPaneError(target.id, window.paneErrorMessage(status || 0), retry);
      return;
    }
    var btn = el('button', { type: 'button', class: 'btn sm pane-error-retry', text: 'Retry' });
    btn.addEventListener('click', retry);
    fill(target, [el('div', { class: 'pane-error', role: 'alert' }, [
      note('Could not load this part of the tab.', ';margin-bottom:8px'), btn])]);
  }
  function denied(r) { return r.status === 401 || r.status === 403; }

  // ── loaders ─────────────────────────────────────────────────────────────────
  function current(c) { return ctl === c && !c.stopped; }

  function loadStatus(c) {
    return getJSON(API).then(function (r) {
      if (!current(c)) return;
      c.status = r.ok ? r.body : null;
      renderStatus(c, c.status);
      if (c.tryit && c.tryit.refreshNotice) c.tryit.refreshNotice();
    }).catch(function () { if (current(c)) renderStatus(c, null); });
  }

  function loadSummary(c) {
    var q = '?window=' + encodeURIComponent(c.window) + '&include_dashboard=' + (c.includeDashboard ? 'true' : 'false');
    return getJSON(API + '/history/summary' + q).then(function (r) {
      if (!current(c)) return;
      if (denied(r)) return noAccess(c);
      if (r.status === 429) return;
      if (!r.ok) {
        failed(c, 'latency', function () { loadSummary(c); }, r.status);
        return failed(c, 'stats', function () { loadSummary(c); }, r.status);
      }
      if (r.body.enabled === false) { c.disabled = true; return disabled(c); }
      c.recording = r.body.recording;
      renderStats(c, r.body);
      renderLatency(c, r.body);
      renderFootnote(c);
    }).catch(function () { if (current(c)) failed(c, 'stats', function () { loadSummary(c); }); });
  }

  function loadFeed(c) {
    return getJSON(API + '/history?limit=' + FEED_ROWS + '&window=all').then(function (r) {
      if (!current(c)) return;
      if (denied(r)) return noAccess(c);
      if (r.status === 429) return;
      if (!r.ok) return failed(c, 'feed', function () { loadFeed(c); }, r.status);
      if (r.body.enabled === false) { c.disabled = true; return disabled(c); }
      c.feed = (r.body.items || []).slice();
      c.nextCursor = r.body.next_cursor || null;
      renderFeed(c);
      return pollLive(c, true);
    }).catch(function () { if (current(c)) failed(c, 'feed', function () { loadFeed(c); }); });
  }

  function loadOlder(c) {
    if (!c.nextCursor) return Promise.resolve();
    return getJSON(API + '/history?limit=' + FEED_ROWS + '&window=all&cursor=' + encodeURIComponent(c.nextCursor))
      .then(function (r) {
        if (!current(c) || !r.ok) return;
        mergeFeed(c, r.body.items || [], false);
        c.nextCursor = r.body.next_cursor || null;
        renderFeed(c);
      });
  }

  function pollLive(c, quiet) {
    return getJSON(API + '/history/live?after_seq=' + c.lastSeq + '&limit=50').then(function (r) {
      if (!current(c) || !r.ok || r.body.enabled === false) return;
      if (c.bootId && r.body.boot_id !== c.bootId) { c.bootId = r.body.boot_id; c.lastSeq = 0; return loadFeed(c); }
      c.bootId = r.body.boot_id;
      c.lastSeq = r.body.last_seq || c.lastSeq;
      var added = mergeFeed(c, r.body.items || [], true);
      if (added) { c.feed.sort(function (a, b) { return String(b.at).localeCompare(String(a.at)); }); renderFeed(c); }
      if (!quiet) announce(c, added);
    }).catch(function () {});
  }

  function loadAll(c) {
    return Promise.all([loadStatus(c), loadSummary(c), loadFeed(c)]);
  }

  // ── polling, only while visible ────────────────────────────────────────────
  function visible(c) {
    return c.pane.isConnected !== false && c.pane.classList.contains('active');
  }
  function stopTimers(c) {
    if (c.liveTimer) clearInterval(c.liveTimer);
    if (c.summaryTimer) clearInterval(c.summaryTimer);
    c.liveTimer = c.summaryTimer = null;
  }
  function startTimers(c) {
    stopTimers(c);
    c.liveTimer = setInterval(function () {
      if (!current(c) || !visible(c)) return stopTimers(c);
      pollLive(c, false);
    }, LIVE_MS);
    c.summaryTimer = setInterval(function () {
      if (!current(c) || !visible(c)) return stopTimers(c);
      loadSummary(c);
    }, SUMMARY_MS);
  }
  function watch(c) {
    if (typeof MutationObserver !== 'function') return;
    c.observer = new MutationObserver(function () {
      if (!current(c)) return c.observer.disconnect();
      if (visible(c) && !c.liveTimer) { loadSummary(c); pollLive(c, false); startTimers(c); }
    });
    c.observer.observe(c.pane, { attributes: true, attributeFilter: ['class'] });
  }

  function teardown() {
    if (!ctl) return;
    stopTimers(ctl);
    if (ctl.observer) ctl.observer.disconnect();
    ctl.stopped = true;
    ctl = null;
  }

  window.odRenderAnswerCheck = function (pane) {
    if (!pane) return;
    teardown();
    var c = { pane: pane, window: '7d', includeDashboard: false, feed: [], nextCursor: null,
              lastSeq: 0, bootId: null, status: null, recording: null, clearDenied: false,
              stopped: false, sections: {} };
    ctl = c;
    pane.textContent = '';
    c.sections.status = card('ac-status', 'Which check runs');
    c.sections.tryit = card('ac-tryit', 'Try it', 'a dashboard test — not counted below');
    c.sections.stats = card('ac-stats', 'How often it said “I don’t have that”');
    c.sections.latency = card('ac-latency', 'Time against the 3-second limit');
    c.sections.feed = card('ac-feed', 'Recent checks');
    c.live = el('div', { class: 'visually-hidden ac-live', role: 'status', 'aria-live': 'polite',
      style: 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)' });
    c.lower = el('div', { class: 'ac-lower' }, [c.sections.tryit, c.sections.stats, c.sections.latency, c.sections.feed]);
    ['stats', 'latency', 'feed'].forEach(function (k) {
      var b = body(c.sections[k]);
      b.setAttribute('aria-busy', 'true');
      fill(b, [note('Loading…')]);
    });
    fill(body(c.sections.status), [note('Loading…')]);
    var root = el('div', { class: 'ac-root', id: 'od-answercheck-root' }, [
      el('div', { class: 'page-head', style: 'margin-bottom:16px' }, [
        el('h2', { style: 'font-size:20px;margin-bottom:6px', text: 'Answer Check' }),
        el('p', { style: 'font-size:13.5px;max-width:760px', text: 'Before your agent uses what SLM found, a second ' +
          'model checks whether it actually answers the question. When it does not, SLM says so instead of ' +
          'handing over a confident wrong answer.' }),
      ]),
      c.sections.status, c.lower, c.live,
    ]);
    pane.appendChild(root);
    if (typeof tryit().mount === 'function') {
      var mountEl = body(c.sections.tryit);
      mountEl.onAsked = function () { setTimeout(function () { if (current(c)) pollLive(c, false); }, 300); };
      c.tryit = tryit().mount(mountEl, function () { return c.status; });
    }
    loadAll(c).then(function () {
      ['stats', 'latency', 'feed'].forEach(function (k) { body(c.sections[k]).removeAttribute('aria-busy'); });
    });
    watch(c);
    if (visible(c)) startTimers(c);
  };

  window.SLMAnswerCheckTab = {
    statusLines: statusLines,
    ITEM_KEYS: ITEM_KEYS,
    controller: function () { return ctl; },
    teardown: teardown,
  };
}());
