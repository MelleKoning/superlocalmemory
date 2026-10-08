// SuperLocalMemory — Health → Memory store (4.1.23).
// Shows the store check SLM runs once after each upgrade (read-only), and lets the
// owner check again or repair. Repair now takes a backup copy first; the server
// changes nothing if that copy fails. While something can be repaired, the Health
// item in the sidebar carries a "Repair" tag so the owner notices.
// Endpoints: GET /api/integrity/summary, POST /api/integrity/check,
// POST /api/integrity/repair-now. Text only via textContent.
(function () {
  'use strict';
  var POLL_RUNNING_MS = 4000;
  var POLL_IDLE_MS = 300000;
  var timer = null;

  function call(path, method) {
    var fetcher = typeof window.slmFetch === 'function' ? window.slmFetch : window.fetch;
    return fetcher(path, { method: method || 'GET', credentials: 'same-origin' })
      .then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (body) {
          return { ok: response.ok, status: response.status, body: body };
        });
      });
  }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function when(iso) {
    if (!iso) return '';
    var date = new Date(iso);
    return isNaN(date.getTime()) ? '' : date.toLocaleString();
  }
  function button(label, cls, onClick) {
    var b = el('button', 'btn btn-sm ' + cls, label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  }

  function tagHealth(count) {
    var link = document.querySelector('a.nav-link[data-tab="health-pane"]');
    if (!link) return;
    var tag = link.querySelector('[data-store-check-tag]');
    if (count > 0 && !tag) {
      tag = el('span', 'tag', 'Repair');
      tag.setAttribute('data-store-check-tag', '');
      link.appendChild(tag);
    } else if (count <= 0 && tag) {
      tag.remove();
    }
  }

  function describe(state) {
    var running = state.running === true;
    if (running && state.status === 'repairing') {
      return 'Saving a backup copy first, then repairing. Recall keeps working; this can take a few minutes.';
    }
    if (running) return 'Checking your memory store. Recall keeps working; this can take a few minutes.';
    if (state.status === 'repair_failed') return 'The repair did not finish: ' + (state.error || 'unknown error');
    if (state.status === 'check_failed') return 'The check did not finish: ' + (state.error || 'unknown error');
    if (!state.checked_at) return 'Not checked yet. SLM checks the store once after each upgrade, a few minutes after it starts.';
    var fixed = state.status === 'repaired' ? 'Repaired ' + when(state.repaired_at) + '. ' : '';
    if (!state.to_repair) return fixed + 'Last checked ' + when(state.checked_at) + ': everything is in order.';
    return fixed + 'Last checked ' + when(state.checked_at) + ': ' + state.to_repair +
      (state.to_repair === 1 ? ' item can' : ' items can') + ' be repaired.';
  }

  function render(mount, state) {
    mount.textContent = '';
    var running = state.running === true;
    mount.appendChild(el('p', 'mb-2', describe(state)));
    var findings = state.findings || {};
    var labels = state.labels || {};
    if (!running && state.to_repair > 0) {
      var list = el('ul', 'small mb-2');
      Object.keys(labels).forEach(function (key) {
        if (findings[key] > 0) list.appendChild(el('li', '', findings[key] + ' ' + labels[key]));
      });
      mount.appendChild(list);
      mount.appendChild(el('p', 'small text-muted mb-2',
        'Repair now saves a backup copy of your memory first and changes nothing if that copy fails. ' +
        'It removes no memory you have: it fixes search indexes and clears what deleted memories left behind. The same repair runs from a terminal with "slm db repair".'));
    }
    if (state.backup && state.status === 'repaired') {
      mount.appendChild(el('p', 'small text-muted mb-2', 'Backup copy: ' + state.backup));
    }
    var actions = el('div', 'd-flex gap-2');
    if (!running && state.to_repair > 0) {
      actions.appendChild(button('Repair now', 'btn-primary', function () { start(mount, '/api/integrity/repair-now'); }));
    }
    var again = button(state.checked_at ? 'Check again' : 'Check now', 'btn-outline-secondary',
      function () { start(mount, '/api/integrity/check'); });
    again.disabled = running;
    actions.appendChild(again);
    mount.appendChild(actions);
    tagHealth(running ? 0 : state.to_repair || 0);
  }

  function schedule(running) {
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(load, running ? POLL_RUNNING_MS : POLL_IDLE_MS);
  }

  // The Health page draws its cards when opened; the sidebar tag must show before
  // that, so the summary is read at page load whether or not the card exists.
  function current() { return document.getElementById('store-check'); }

  function load() {
    return call('/api/integrity/summary').then(function (result) {
      if (!result.ok) throw new Error('summary unavailable');
      var mount = current();
      if (mount) render(mount, result.body);
      else tagHealth(result.body.running === true ? 0 : result.body.to_repair || 0);
      schedule(result.body.running === true);
    }).catch(function () {
      var mount = current();
      if (mount) {
        mount.textContent = '';
        mount.appendChild(el('p', 'text-muted mb-0', 'The memory store check is not available right now.'));
      }
      schedule(false);
    });
  }

  function start(mount, path) {
    return call(path, 'POST').then(function (result) {
      if (!result.ok && result.status !== 409) {
        mount.appendChild(el('p', 'small text-danger mt-2', 'That did not start. Try again.'));
        return;
      }
      return load();
    });
  }

  function init() { return load(); }
  window.odStoreCheck = { init: init, load: load, describe: describe };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
