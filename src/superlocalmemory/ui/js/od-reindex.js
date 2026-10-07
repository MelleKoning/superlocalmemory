/* od-reindex.js — the background embedding re-index, shown in Settings (4.1.22).
 *
 * Changing the embedding model answers 202 with a job: the daemon re-embeds
 * every memory in the background and recall keeps using the old model until
 * the switch completes. This panel shows that job — state, done/total, the
 * cancel action — polls GET /api/v3/embedding/reindex while it is active, and
 * stops polling once it is activated, failed, cancelled or rolled back.
 *
 * Exposes window.odReindex:
 *   panel(post)    element for the Embeddings group; post(url, body) is the
 *                  caller's authenticated POST, rejecting on non-2xx
 *   refresh()      read the job once (Settings opened); polls if it is active
 *   track(body)    a save answered 202: show body.job and poll
 *   conflict(body) a save answered 409 reindex_running: show that job and poll
 *   onFinish(fn)   fn(job) once a job this page watched reaches a final state
 * Every server string is set with textContent, never parsed as HTML.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var API = '/api/v3/embedding/reindex';
  var POLL_MS = 2000;
  var MAX_POLL_FAILURES = 3;
  var ACTIVE = { queued: 'Queued', running: 'Re-indexing', catching_up: 'Catching up with new memories',
                 ready: 'Switching over' };

  var _post = null, _root = null, _parts = {}, _job = null;
  var _timer = 0, _gen = 0, _failures = 0, _onFinish = [];
  var _watching = false;  // a poll loop owns the panel; refresh() must not cut it short

  function part(tag, id, css) {
    var e = document.createElement(tag);
    if (id) e.id = 'od-reindex-' + id;
    if (css) Object.assign(e.style, css);
    return e;
  }

  function panel(post) {
    _post = post;
    _root = part('div', null, { display: 'none', fontSize: '12px', color: 'var(--fg-2)' });
    _root.id = 'od-reindex';
    _parts.state = part('div', 'state', { fontWeight: '600', color: 'var(--fg)' });
    _parts.progress = part('div', 'progress');
    _parts.note = part('div', 'note');
    _parts.error = part('div', 'error', { color: 'var(--danger)' });
    _parts.cancel = part('button', 'cancel', { display: 'none', marginTop: '6px' });
    _parts.cancel.type = 'button';
    _parts.cancel.className = 'btn sm ghost';
    _parts.cancel.textContent = 'Cancel re-index';
    _parts.cancel.addEventListener('click', cancel);
    ['state', 'progress', 'note', 'error', 'cancel'].forEach(function (k) { _root.appendChild(_parts[k]); });
    return _root;
  }

  function isActive(job) { return !!(job && ACTIVE[job.state]); }

  function progressText(job) {
    var text = (job.done || 0) + ' / ' + (job.total || 0) + ' memories';
    if (job.eta_seconds != null) text += ' · about ' + Math.max(1, Math.round(job.eta_seconds)) + ' s left';
    return text;
  }

  function finalText(job) {
    if (job.state === 'activated') return 'Done: recall now uses ' + job.to;
    if (job.state === 'failed') return 'Failed — ' + job.from + ' stays the embedding model';
    if (job.state === 'cancelled') return 'Cancelled — ' + job.from + ' stays the embedding model';
    if (job.state === 'rolled_back') return 'Rolled back — superseded by a later switch';
    return 'Re-index ' + job.state;
  }

  function render(job) {
    if (!_root) return;
    _job = job;
    _root.style.display = job ? '' : 'none';
    if (!job) return;
    var head = 'Re-index job ' + job.job_id + ' (' + job.from + ' → ' + job.to + '): ';
    _parts.state.textContent = head + (isActive(job) ? ACTIVE[job.state] : finalText(job));
    _parts.progress.textContent = progressText(job);
    _parts.note.textContent = isActive(job)
      ? 'Recall keeps using ' + job.from + ' until the switch completes.' : '';
    _parts.error.textContent = job.state === 'failed' && job.error ? 'Error: ' + job.error : '';
    _parts.cancel.style.display = isActive(job) ? '' : 'none';
    _parts.cancel.disabled = false;
  }

  function showProblem(text) {
    if (!_root) return;
    _root.style.display = '';
    _parts.error.textContent = text;
  }

  function describe(status, data, fallback) {
    var d = data || {};
    var detail = typeof d.detail === 'string' ? d.detail : (d.detail && d.detail.message);
    return 'Error ' + status + ': ' + (detail || d.error || d.message || fallback);
  }

  function schedule() {
    _watching = true;
    var gen = _gen;
    _timer = window.setTimeout(function () { if (gen === _gen) poll(); }, POLL_MS);
  }

  function stop() {
    _gen += 1;
    if (_timer) window.clearTimeout(_timer);
    _timer = 0;
  }

  function settle(job, watched) {
    render(job);
    if (isActive(job)) { schedule(); return; }
    _watching = false;
    if (watched) _onFinish.forEach(function (fn) { try { fn(job); } catch (e) { /* keep others */ } });
  }

  function poll() {
    stop();
    var gen = _gen;
    return fetch(API).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (gen !== _gen) return;
        if (!r.ok) throw { status: r.status, data: data };
        _failures = 0;
        if (data.notice && !data.job) showProblem(String(data.notice));
        settle(data.job || null, true);
      });
    }).catch(function (e) {
      if (gen !== _gen) return;
      _failures += 1;
      if (_failures < MAX_POLL_FAILURES) { schedule(); return; }
      _watching = false;
      showProblem('Progress unavailable — ' + describe(e && e.status || 'network', e && e.data,
        (e && e.message) || 'the daemon did not answer'));
    });
  }

  /** Settings opened: show a job only while it runs; finished ones are history. */
  function refresh() {
    if (_watching) return Promise.resolve();
    stop();
    var gen = _gen;
    return fetch(API).then(function (r) { return r.ok ? r.json() : null; }).then(function (data) {
      if (gen !== _gen || !data) return;
      if (isActive(data.job)) settle(data.job, false);
      else if (data.notice) showProblem(String(data.notice));
    }).catch(function () { /* no runner, or daemon down: nothing to show */ });
  }

  function track(body) {
    stop();
    _failures = 0;
    if (body && body.job) render(body.job);
    schedule();
  }

  function conflict(body) {
    stop();
    _failures = 0;
    showProblem(describe(409, body, 'a re-index is already running'));
    schedule();
  }

  function cancel() {
    if (!_post || !_job) return;
    _parts.cancel.disabled = true;
    stop();
    _post(API + '/cancel', {}).then(function (r) { return r.json(); }).then(function (data) {
      settle((data && data.job) || _job, true);
    }).catch(function (e) {
      _parts.cancel.disabled = false;
      showProblem('Cancel failed — Error ' + (e && e.status ? e.status : 'network') + ': ' +
        ((e && e.message) || 'request failed'));
      if (isActive(_job)) schedule();
    });
  }

  window.odReindex = {
    panel: panel, refresh: refresh, track: track, conflict: conflict,
    onFinish: function (fn) { _onFinish.push(fn); },
    POLL_MS: POLL_MS
  };
}());
