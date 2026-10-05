// od-summaries.js — the Summaries pane of the Memories page (issue #113)
// Public: window.ODSummaries = { controls(id), onShow(id), load(id, kind, target),
//                                provenance(d) }
// CSP-safe: own event delegation on the pane. XSS-safe: every API value is set
// with textContent, never as HTML.
//
// Moved out of od-memories.js in 4.1.21 (that file is far over the size limit)
// when three things were added:
//   * a session picker — a session summary could not be asked for here at all;
//   * the browser's own day: "Today" sends the local date and the browser's
//     offset, so early-morning memories east of UTC land on the right day;
//   * the memories a summary came from, by id, each one openable — a summary
//     that cannot be traced back is the opaque summary #113 warned against.
//
// Endpoints: GET /api/summary?kind&target&tz_offset_minutes
//            GET /api/summary/projects   GET /api/summary/sessions
(function () {
  'use strict';

  var _SELECT_STYLE = 'padding:7px 10px;border:1px solid var(--border);' +
    'border-radius:var(--r-md);background:var(--card-2);color:var(--fg);' +
    'font-size:13px;max-width:340px';

  /* Offset east of UTC in minutes, as /api/summary takes it (330 for India). */
  function _offsetMinutes() { return -new Date().getTimezoneOffset(); }

  /* YYYY-MM-DD of a Date in the browser's own calendar, never toISOString (UTC). */
  function _localDate(d) {
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') +
      '-' + String(d.getDate()).padStart(2, '0');
  }

  function _dayTarget(word) {
    var d = new Date();
    if (word === 'yesterday') d.setDate(d.getDate() - 1);
    return _localDate(d);
  }

  /* The pane's controls. Rendered by od-memories.js inside the Summaries pane. */
  function controls(id) {
    return (
      '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px">' +
        '<button class="tab" data-sum-act="day" data-target="today">Today</button>' +
        '<button class="tab" data-sum-act="day" data-target="yesterday">Yesterday</button>' +
        // SLM runs as one global daemon and this is a browser tab: there is no
        // working directory to mean "this project", so the server lists the
        // projects (and sessions) it has actually seen and the user picks one.
        '<select id="' + id + '-sum-proj" data-sum-act="project" ' +
          'title="Projects SuperLocalMemory has recorded activity in" style="' + _SELECT_STYLE + '">' +
          '<option value="">Loading projects…</option>' +
        '</select>' +
        '<select id="' + id + '-sum-sess" data-sum-act="session" ' +
          'title="Sessions with saved memories" style="' + _SELECT_STYLE + '">' +
          '<option value="">Loading sessions…</option>' +
        '</select>' +
      '</div>' +
      '<div id="' + id + '-sum-out" style="font-size:13px;color:var(--fg-2)">Pick a summary above.</div>'
    );
  }

  function onShow(id) {
    _wire(id);
    _loadOptions(id, 'proj', '/api/summary/projects', 'projects', function (p) {
      return { value: p.path, title: p.path, text: p.label + ' (' + p.events + ')' };
    }, 'Summarise a project…', 'No projects recorded yet');
    _loadOptions(id, 'sess', '/api/summary/sessions', 'sessions', function (s) {
      var n = s.memory_count;
      return { value: s.session_id, title: s.session_id,
               text: s.session_id.slice(0, 24) + ' (' + n + (n === 1 ? ' memory' : ' memories') + ')' };
    }, 'Summarise a session…', 'No sessions with saved memories');
  }

  function _wire(id) {
    var out = document.getElementById(id + '-sum-out');
    var pane = out && out.parentNode;
    if (!pane || pane.dataset.sumWired === '1') return;
    pane.dataset.sumWired = '1';
    pane.addEventListener('click', function (e) {
      var el = e.target.closest('[data-sum-act="day"]');
      if (el) load(id, 'day', _dayTarget(el.dataset.target));
    });
    pane.addEventListener('change', function (e) {
      var act = e.target.dataset && e.target.dataset.sumAct;
      if ((act === 'project' || act === 'session') && e.target.value) {
        load(id, act, e.target.value);
      }
    });
  }

  /* Fill a picker once per rendered <select>. The guard lives on the element:
   * a module flag would outlive the DOM it described and suppress the fetch
   * after the pane re-renders. */
  function _loadOptions(id, suffix, url, key, toOption, prompt, emptyText) {
    var sel = document.getElementById(id + '-sum-' + suffix);
    if (!sel || sel.dataset.loaded === '1' || sel.dataset.loading === '1') return;
    sel.dataset.loading = '1';
    fetch(url)
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (d) {
        var list = (d && d[key]) || [];
        sel.dataset.loaded = '1';
        delete sel.dataset.loading;
        sel.textContent = '';
        if (!list.length) {
          sel.appendChild(_option('', emptyText));
          sel.disabled = true;
          return;
        }
        sel.appendChild(_option('', prompt));
        list.forEach(function (item) {
          var o = toOption(item);
          var opt = _option(o.value, o.text);
          opt.title = o.title;
          sel.appendChild(opt);
        });
      })
      .catch(function () {
        // A silent empty list reads as "you have none", a different and wrong
        // statement. Clearing the in-flight marker lets the next visit retry.
        delete sel.dataset.loading;
        sel.textContent = '';
        sel.appendChild(_option('', 'Could not load — retry'));
      });
  }

  function _option(value, text) {
    var o = document.createElement('option');
    o.value = value;
    o.textContent = text;
    return o;
  }

  function load(id, kind, target) {
    var out = document.getElementById(id + '-sum-out');
    if (!out) return;
    out.textContent = 'Building summary…';
    var q = '/api/summary?kind=' + encodeURIComponent(kind) +
      '&tz_offset_minutes=' + encodeURIComponent(String(_offsetMinutes()));
    if (target) q += '&target=' + encodeURIComponent(target);
    fetch(q)
      .then(function (r) {
        if (!r.ok) return r.json().then(function (e) {
          var detail = e && e.detail;
          throw new Error((detail && detail.message) || detail || ('HTTP ' + r.status));
        });
        return r.json();
      })
      .then(function (d) { _render(out, d); })
      .catch(function (e) {
        out.textContent = 'Could not build summary: ' + (e && e.message ? e.message : 'error');
      });
  }

  function _render(out, d) {
    out.textContent = '';
    var body = document.createElement('pre');
    body.style.cssText = 'white-space:pre-wrap;font-family:inherit;font-size:13px;' +
      'line-height:1.65;color:var(--fg);background:var(--card-2);border:1px solid var(--border);' +
      'border-radius:var(--r-md);padding:14px;margin:0';
    body.textContent = d.summary || '(nothing recorded)';
    out.appendChild(body);

    // Coverage on every result, never only on bad ones.
    var meta = document.createElement('div');
    meta.style.cssText = 'margin-top:10px;font-size:12px;color:var(--fg-3);line-height:1.6';
    meta.textContent = provenance(d);
    out.appendChild(meta);

    // Why it came out this way: a plain summary in a mode with no language
    // model is by design, and the user should learn that, not suspect a bug.
    var cap = d.capability;
    if (cap && cap.message) {
      var hint = document.createElement('div');
      hint.style.cssText = 'margin-top:10px;font-size:12px;line-height:1.6;' +
        'color:var(--fg-2);background:var(--card-2);border:1px solid var(--border);' +
        'border-left:3px solid var(--accent, var(--border));' +
        'border-radius:var(--r-md);padding:10px 12px';
      hint.textContent = cap.message;
      out.appendChild(hint);
    }
    var sources = _sources(d.source_fact_ids || []);
    if (sources) out.appendChild(sources);
  }

  /* The memories the summary came from, each openable in place (fact-detail.js
   * expands a `.fact-result-item[data-fact-id]` on click). */
  function _sources(ids) {
    if (!ids.length) return null;
    var box = document.createElement('details');
    box.style.cssText = 'margin-top:12px;font-size:12px;color:var(--fg-2)';
    var head = document.createElement('summary');
    head.style.cursor = 'pointer';
    head.textContent = 'The ' + ids.length + (ids.length === 1 ? ' memory' : ' memories') +
      ' this came from — click one to open it';
    box.appendChild(head);
    ids.forEach(function (fid) {
      var row = document.createElement('div');
      row.className = 'fact-result-item';
      row.setAttribute('data-fact-id', fid);
      row.style.cssText = 'padding:4px 0;cursor:pointer;font-family:var(--mono, monospace)';
      row.textContent = 'Memory ' + fid;
      box.appendChild(row);
    });
    return box;
  }

  /* Plain-English provenance line: how many memories, how much was covered,
   * and how it was written — never internal words like "llm_b". */
  var _COVERAGE_TEXT = {
    full:         'Covers everything recorded for this period.',
    partial:      'Partial view — some of what was recorded is not reflected here.',
    insufficient: 'Too little was recorded to summarise properly.',
    no_session:   'That session has no memories attached to it.',
    unavailable:  'The underlying data could not be read.',
  };

  var _METHOD_TEXT = {
    extractive: 'Assembled directly from your own notes — no AI involved.',
    llm_b:      'Written by the AI model running locally on this machine.',
    llm_c:      'Written by your configured cloud AI model.',
  };

  function provenance(d) {
    var n = d.source_count || 0;
    var md = d.metadata || {};
    var parts = [];
    if (n > 0) {
      parts.push('Based on ' + n + ' memor' + (n === 1 ? 'y' : 'ies') + '.');
    } else if (md.event_count) {
      // Activity with no stored facts is a description, not an error.
      parts.push('Based on ' + md.event_count + ' recorded actions; no facts were ' +
                 'saved for this project.');
    } else {
      parts.push('No stored memories matched.');
    }
    parts.push(_COVERAGE_TEXT[d.coverage] || 'Coverage unknown.');
    if (d.generated_by && _METHOD_TEXT[d.generated_by]) parts.push(_METHOD_TEXT[d.generated_by]);
    return parts.join(' ');
  }

  window.ODSummaries = { controls: controls, onShow: onShow, load: load,
                         provenance: provenance };
}());
