// od-views.js — the Saved views pane of the Memories page (issue #113)
// Public: window.ODViews = { pane(id), onShow(id) }
// CSP-safe: own event delegation on the pane. XSS-safe: every API value is set
// with textContent, never as HTML.
//
// A saved view is a question someone asks their memory often, saved under a
// name. Running it is running recall — the same search an AI assistant's recall runs — so the
// answer is the same each time on an unchanged store, and every result shows
// the id of the memory it came from (click a result to open that memory).
//
// Endpoints: GET /api/v3/views   GET /api/v3/views/run?name=
//            POST /api/v3/views  POST /api/v3/views/rename  POST /api/v3/views/delete
(function () {
  'use strict';

  var _INPUT = 'padding:8px 12px;border:1px solid var(--border);border-radius:6px;' +
    'font-size:13px;background:var(--bg-2);color:var(--fg)';
  //: The design system's own button classes (css/design-system.css).
  var _BTN = 'btn sm';
  var _PRIMARY = 'btn sm primary';

  //: Time ranges in recall's own words ("7d"); the server validates them again.
  var _WINDOWS = [['', 'Any time'], ['24h', 'Last 24 hours'], ['7d', 'Last 7 days'],
                  ['30d', 'Last 30 days'], ['1y', 'Last year']];

  function _el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'style') e.style.cssText = attrs[k];
      else e.setAttribute(k, attrs[k]);
    });
    if (text != null) e.textContent = String(text);
    return e;
  }

  /* The pane's static frame. Rendered by od-memories.js. */
  function pane(id) {
    return (
      '<div id="' + id + '-views" style="padding-top:12px">' +
        '<p style="font-size:13px;color:var(--fg-2);margin-bottom:12px">' +
          'A saved view is a question you ask your memory often. Save it once and run ' +
          'it any time: it runs the same search your AI assistant uses, so it gives the same ' +
          'answer, and every result shows the memory it came from.' +
        '</p>' +
        '<div id="' + id + '-views-list" style="margin-bottom:18px">Loading your views…</div>' +
        '<div id="' + id + '-views-form"></div>' +
        '<div id="' + id + '-views-msg" role="status" style="font-size:12px;margin:8px 0;' +
          'color:var(--fg-2)"></div>' +
        '<div id="' + id + '-views-out"></div>' +
      '</div>'
    );
  }

  function onShow(id) {
    var root = document.getElementById(id + '-views');
    if (!root) return;
    if (root.dataset.wired !== '1') { root.dataset.wired = '1'; _wire(id, root); }
    _refresh(id);
  }

  function _api(method, url, body) {
    var init = { method: method };
    if (body) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (d) {
        if (r.ok) return d;
        var detail = d && d.detail;
        throw new Error((detail && detail.message) || (typeof detail === 'string' && detail) ||
                        ('The request failed (HTTP ' + r.status + ').'));
      });
    });
  }

  function _say(id, text, bad) {
    var msg = document.getElementById(id + '-views-msg');
    if (!msg) return;
    msg.textContent = text || '';
    msg.style.color = bad ? 'var(--danger, #c0392b)' : 'var(--fg-2)';
  }

  function _refresh(id) {
    _api('GET', '/api/v3/views')
      .then(function (d) { _renderList(id, d.views || []); _renderForm(id, d.limits || {}); })
      .catch(function (e) {
        var list = document.getElementById(id + '-views-list');
        if (list) list.textContent = 'Could not load your saved views: ' + e.message;
      });
  }

  function _describe(v) {
    var bits = ['“' + v.query + '”'];
    var f = v.filters || {};
    if (f.window) bits.push('time: ' + f.window);
    if (f.kind) bits.push('kind: ' + f.kind);
    if (f.as_of) bits.push('as of ' + f.as_of);
    bits.push('up to ' + v.limit + ' results');
    return bits.join(' · ');
  }

  function _renderList(id, views) {
    var list = document.getElementById(id + '-views-list');
    if (!list) return;
    list.textContent = '';
    if (!views.length) {
      list.appendChild(_el('div', { 'data-views-empty': '1', style: 'padding:14px;' +
        'border:1px dashed var(--border);border-radius:var(--r-md);font-size:13px;' +
        'color:var(--fg-2);line-height:1.6' },
        'You have no saved views yet. Give one a name and a question below — for ' +
        'example "Work log" with "what did I ship" over the last 7 days — then press ' +
        'Save view. It appears here with a Run button.'));
      return;
    }
    views.forEach(function (v) {
      var row = _el('div', { 'data-view-row': v.name, style: 'display:flex;gap:10px;' +
        'align-items:center;padding:9px 0;border-bottom:1px solid var(--border)' });
      var text = _el('div', { style: 'flex:1;min-width:0' });
      text.appendChild(_el('div', { style: 'font-weight:600;color:var(--fg)' }, v.name));
      text.appendChild(_el('div', { style: 'font-size:12px;color:var(--fg-3)' }, _describe(v)));
      row.appendChild(text);
      [['run', 'Run', _PRIMARY], ['rename', 'Rename', _BTN], ['delete', 'Delete', _BTN]]
        .forEach(function (b) {
          var btn = _el('button', { 'data-views-act': b[0], 'data-name': v.name,
                                    'class': b[2] }, b[1]);
          row.appendChild(btn);
        });
      list.appendChild(row);
    });
  }

  function _renderForm(id, limits) {
    var form = document.getElementById(id + '-views-form');
    if (!form || form.dataset.built === '1') return;
    form.dataset.built = '1';
    form.appendChild(_el('div', { style: 'font-weight:600;margin-bottom:8px' }, 'New saved view'));
    var row = _el('div', { style: 'display:flex;gap:8px;flex-wrap:wrap;align-items:center' });
    row.appendChild(_el('input', { id: id + '-views-name', placeholder: 'Name, e.g. Work log',
      maxlength: String(limits.max_name_chars || 80), style: _INPUT + ';width:180px' }));
    row.appendChild(_el('input', { id: id + '-views-query',
      placeholder: 'What should it look for? e.g. what did I ship',
      maxlength: String(limits.max_query_chars || 1000), style: _INPUT + ';flex:1;min-width:220px' }));
    var win = _el('select', { id: id + '-views-window', style: _INPUT, title: 'Time range' });
    _WINDOWS.forEach(function (w) { win.appendChild(_option(w[0], w[1])); });
    row.appendChild(win);
    var kind = _el('select', { id: id + '-views-kind', style: _INPUT, title: 'Kind of memory' });
    kind.appendChild(_option('', 'Any kind'));
    (limits.kinds || []).forEach(function (k) { kind.appendChild(_option(k.value, k.label)); });
    row.appendChild(kind);
    row.appendChild(_el('button', { 'data-views-act': 'create', 'class': _PRIMARY },
                        'Save view'));
    form.appendChild(row);
  }

  function _option(value, text) {
    var o = document.createElement('option');
    o.value = value;
    o.textContent = text;
    return o;
  }

  function _value(id, suffix) {
    var el = document.getElementById(id + '-views-' + suffix);
    return el ? String(el.value || '').trim() : '';
  }

  function _create(id) {
    var name = _value(id, 'name'), query = _value(id, 'query');
    if (!name || !query) { _say(id, 'Give the view a name and a question first.', true); return; }
    var filters = {};
    if (_value(id, 'window')) filters.window = _value(id, 'window');
    if (_value(id, 'kind')) filters.kind = _value(id, 'kind');
    _api('POST', '/api/v3/views', { name: name, query: query, filters: filters })
      .then(function (d) {
        _say(id, d.message || 'Saved.');
        ['name', 'query'].forEach(function (s) {
          var el = document.getElementById(id + '-views-' + s);
          if (el) el.value = '';
        });
        _refresh(id);
      })
      .catch(function (e) { _say(id, e.message, true); });
  }

  function _run(id, name) {
    var out = document.getElementById(id + '-views-out');
    if (!out) return;
    out.textContent = 'Running “' + name + '”…';
    _api('GET', '/api/v3/views/run?name=' + encodeURIComponent(name))
      .then(function (d) { _renderRun(out, d); })
      .catch(function (e) { out.textContent = 'Could not run the view: ' + e.message; });
  }

  function _renderRun(out, d) {
    out.textContent = '';
    out.appendChild(_el('div', { style: 'font-weight:600;margin:6px 0 8px' },
      d.view.name + ' — ' + d.count + (d.count === 1 ? ' result' : ' results')));
    if (!d.results.length) {
      out.appendChild(_el('div', { style: 'color:var(--fg-2);font-size:13px' },
        'Nothing in your memory matches this view right now.'));
      return;
    }
    if (d.no_confident_match) {
      out.appendChild(_el('div', { style: 'color:var(--fg-2);font-size:12px;margin-bottom:6px' },
        'None of these is a confident match; treat them as leads.'));
    }
    d.results.forEach(function (r) {
      // fact-detail.js opens the memory when a .fact-result-item is clicked.
      var item = _el('div', { 'class': 'fact-result-item', 'data-fact-id': r.fact_id,
        style: 'padding:8px 0;border-bottom:1px solid var(--border);cursor:pointer' });
      item.appendChild(_el('div', { style: 'font-size:13px;color:var(--fg)' },
        r.rank + '. ' + (r.content || '')));
      item.appendChild(_el('div', { style: 'font-size:11px;color:var(--fg-3);' +
        'font-family:var(--mono, monospace)' },
        'Memory ID ' + r.fact_id + ' · score ' + Number(r.score || 0).toFixed(2) +
        (r.age_label ? ' · ' + r.age_label : '')));
      out.appendChild(item);
    });
  }

  function _rename(id, name) {
    var next = window.prompt('New name for “' + name + '”:', name);
    if (next == null || !next.trim() || next.trim() === name) return;
    _api('POST', '/api/v3/views/rename', { name: name, new_name: next.trim() })
      .then(function (d) { _say(id, d.message || 'Renamed.'); _refresh(id); })
      .catch(function (e) { _say(id, e.message, true); });
  }

  function _delete(id, name) {
    // The shared modal every destructive dashboard action goes through.
    var ask = typeof window.confirmDestructive === 'function'
      ? window.confirmDestructive({ title: 'Delete saved view', target: name,
          consequence: 'Removes this saved question. Your memories are not touched.',
          confirmationText: 'DELETE', confirmLabel: 'Delete view' })
      : Promise.resolve(window.confirm('Delete the saved view “' + name +
          '”? Your memories are not touched.'));
    return ask.then(function (yes) {
      if (!yes) return;
      return _api('POST', '/api/v3/views/delete', { name: name })
        .then(function (d) { _say(id, d.message || 'Deleted.'); _refresh(id); })
        .catch(function (e) { _say(id, e.message, true); });
    });
  }

  function _wire(id, root) {
    root.addEventListener('click', function (e) {
      var el = e.target.closest('[data-views-act]');
      if (!el) return;
      var act = el.dataset.viewsAct, name = el.dataset.name || '';
      if (act === 'create') _create(id);
      else if (act === 'run') _run(id, name);
      else if (act === 'rename') _rename(id, name);
      else if (act === 'delete') _delete(id, name);
    });
  }

  window.ODViews = { pane: pane, onShow: onShow };
}());
