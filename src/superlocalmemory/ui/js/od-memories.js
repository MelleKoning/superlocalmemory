// od-memories.js — Memories · Timeline · Knowledge Clusters (OD design port)
// Render: #memories-pane  |  Public: window.odRenderMemories(container)
// CSP-safe: data-od-act event delegation. XSS-safe: _esc() on all API values.
// nosemgrep: innerHTML — all dynamic values escaped via _esc()
//
// Endpoints (live daemon 127.0.0.1:8765):
//   GET /api/memories?limit=N&offset=N[&kind=X][&category=X]
//       → { memories[{id,memory_id,content,category,memory_kind,
//                      memory_kind_label,memory_kind_state,importance(0-1),
//                      access_count,created_at,project_name}], total }
//       `kind` filters on the DISPLAYED kind (nine memory_kind values,
//       e.g. "decision"), the same precedence `slm list`/recall use
//       (core/kind_query.py). `category` still filters on the raw legacy
//       fact_type for callers that want that instead.
//   GET /api/memories/kind-counts[?scope=X] → { counts:{<kind>:n,...}, truncated }
//   GET /api/memories/{id}/facts → { facts[{fact_type,content}] }
//   GET /api/v3/timeline/?range=Xd&group_by=category&limit=N
//       → { events[{id,timestamp,category}] }
//   GET /api/clusters → { clusters[{cluster_id,member_count,categories,summary}] }
//   GET /api/clusters/{id}?limit=N → { members[], summary }
// TODO: /api/memories/{id}/score-breakdown — Semantic/BM25/Entity/Temporal not yet exposed.
(function () {
  'use strict';

  // ── Utilities ───────────────────────────────────────────────────────────────

  function _esc(s) {
    if (typeof escapeHtml === 'function') return escapeHtml(s);
    var d = document.createElement('div');
    d.textContent = String(s == null ? '' : s);
    return d.innerHTML;
  }

  function _fmt(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso) : d.toLocaleDateString();
  }

  function _ago(iso) {
    if (!iso) return '';
    var s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 0) s = 0;
    if (s < 60)    return Math.floor(s) + 's ago';
    if (s < 3600)  return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
  }

  // importance arrives as 0–1 float; display as X/10 integer
  function _imp(raw) { return Math.round((parseFloat(raw) || 0) * 10); }

  // importance (0–1) → badge class for Score column
  function _scoreCls(raw) {
    var v = parseFloat(raw) || 0;
    return v >= 0.7 ? 'ok' : v >= 0.4 ? 'warn' : 'danger';
  }

  // memory_kind → badge class. Only 6 .badge modifier colors exist
  // (design-system.css .badge.ok/.warn/.danger/.violet/.cyan/.neutral) for
  // the nine kinds, so two pairs share a color; the label text (not the
  // color) is what distinguishes them.
  function _catCls(kind) {
    var MAP = { semantic: 'violet', episodic: 'cyan', status: 'neutral',
                opinion: 'warn', rule: 'violet', decision: 'ok',
                procedure: 'cyan', prospective: 'ok', correction: 'danger' };
    return MAP[String(kind).toLowerCase()] || 'neutral';
  }

  // Normalize /api/search result → memory shape (fact_id→id, score→importance)
  function _normSearch(r) { return {id:r.fact_id||r.memory_id||'',content:r.content||'',category:r.category||'semantic',project_name:r.project_name||'',importance:r.score||r.confidence||0,created_at:r.created_at||'',access_count:r.access_count||0}; }
  // shared_with is stored as a JSON array string; show it comma-separated.
  function _sharedToStr(v) {
    if (!v) return '';
    if (Array.isArray(v)) return v.join(', ');
    try { var a = JSON.parse(v); return Array.isArray(a) ? a.join(', ') : ''; }
    catch (e) { return ''; }
  }

  // Write-auth token: reuses dashboard.js closure when available, else fails gracefully
  function _getMutToken() { return typeof window.dashboardInstallToken==='function'?window.dashboardInstallToken():Promise.resolve(''); }

  // ── CSS (injected once) ─────────────────────────────────────────────────────

  var _cssInjected = false;
  function _injectCSS() {
    if (_cssInjected) return;
    _cssInjected = true;
    var s = document.createElement('style');
    s.dataset.odModule = 'memories';
    s.textContent =
      '.od-drawer{position:fixed;top:0;right:0;height:100vh;width:420px;max-width:92vw;' +
      'background:var(--card);border-left:1px solid var(--border);box-shadow:var(--sh-lg);' +
      'transform:translateX(100%);transition:transform .28s cubic-bezier(.22,1,.36,1);' +
      'z-index:60;overflow-y:auto;padding:24px;}' +
      '.od-drawer.open{transform:none;}' +
      '.od-drawer-scrim{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:59;' +
      'opacity:0;pointer-events:none;transition:opacity .25s;}' +
      '.od-drawer-scrim.on{opacity:1;pointer-events:auto;}' +
      '.od-prov{padding:14px;border-radius:var(--r-md);background:var(--card-2);' +
      'border:1px solid var(--border);margin-top:12px;}';
    document.head.appendChild(s);
  }

  // ── Module state ────────────────────────────────────────────────────────────

  var _st = {
    rootId:    null,
    // Must match the pane carrying `active` in _scaffold(). Recall Lab leads.
    activeTab: 'recall',
    category:  null,   // null = all; string = filtered category
    sort:      'created',
    page:      0,
    pageSize:  50,
    total:     0,
    memories:  [],
    catCounts: {},     // pre-fetched per-category totals
    tlRange:   '30d',
    cluLoaded: false,  // reset in render() to avoid stale state on re-entry
    searchQ:   null,
    requestSeq: 0,
  };

  // The nine memory kinds (storage/memory_kinds.py MemoryKind) and their
  // human-facing labels (storage/memory_kinds.py LABELS). Kept in this
  // fixed order (same as the Python enum) rather than sorted by count, so
  // the chip bar does not reshuffle every time counts change.
  var KNOWN_KINDS = [
    { kind: 'semantic',    label: 'Fact' },
    { kind: 'episodic',    label: 'Event' },
    { kind: 'status',      label: 'Current state' },
    { kind: 'opinion',     label: 'Preference or view' },
    { kind: 'rule',        label: 'Standing rule' },
    { kind: 'decision',    label: 'Decision' },
    { kind: 'procedure',   label: 'How-to' },
    { kind: 'prospective', label: 'Plan or to-do' },
    { kind: 'correction',  label: 'Correction' },
  ];

  // ── Main entry ──────────────────────────────────────────────────────────────

  function render(container) {
    if (!container) return;
    _injectCSS();
    var id = 'od-mem-' + Math.random().toString(36).slice(2, 8);
    // activeTab is reset explicitly: _scaffold always emits the Recall pane as the
    // active one, so leaving a stale value here (from a tab click before the user
    // navigated away and back) would desync state from what is actually on screen.
    _st = Object.assign({}, _st, {
      rootId: id, page: 0, category: null, sort: 'created', activeTab: 'recall',
      memories: [], catCounts: {}, cluLoaded: false, searchQ: null, requestSeq: 0,
    });
    // Clear per-id timeline cache for this render instance
    _tlLoaded = {};
    container.innerHTML = _scaffold(id);
    _wire(container, id);
    // Fire initial loads in parallel
    _loadMem(id);
    _loadCatCounts(id);
  }

  // ── Scaffold HTML ────────────────────────────────────────────────────────────

  function _scaffold(id) {
    return (
      '<div id="' + id + '">' +
        '<div class="page-head">' +
          '<h2>Everything you\'ve remembered</h2>' +
          '<p id="' + id + '-sub" style="color:var(--fg-2);margin-top:5px">Loading…</p>' +
        '</div>' +
        // Tab order is deliberate (owner request, 4.0.8): the two tabs that ANSWER
        // a question come first, the three that BROWSE data come after. Someone who
        // opens this page usually wants "what do you know about X" or "what did I do
        // today" — not a 3,600-row table sorted by insert time.
        '<div class="tabs" id="' + id + '-tabs">' +
          // Recall Lab — recall-lab.js is loaded in index.html and uses document-level
          // click/keydown delegation keyed on the IDs below, so it survives re-renders.
          '<button class="tab active" data-od-act="tab" data-tab="recall">Recall Lab</button>' +
          '<button class="tab" data-od-act="tab" data-tab="summary">Summaries</button>' +
          '<button class="tab" data-od-act="tab" data-tab="views">Saved views</button>' +
          '<button class="tab" data-od-act="tab" data-tab="all">' +
            'All memories <span class="cnt" id="' + id + '-cnt-all">…</span></button>' +
          '<button class="tab" data-od-act="tab" data-tab="timeline">Creation timeline</button>' +
          '<button class="tab" data-od-act="tab" data-tab="clusters">' +
            'Knowledge clusters <span class="cnt" id="' + id + '-cnt-clusters">…</span></button>' +
        '</div>' +
        '<div class="tabpane" id="' + id + '-pane-all">' + _allScaffold(id) + '</div>' +
        '<div class="tabpane" id="' + id + '-pane-timeline">' + _tlScaffold(id) + '</div>' +
        '<div class="tabpane" id="' + id + '-pane-clusters">' +
          '<div class="launch-grid" id="' + id + '-clu-grid">' +
            _loading('Loading clusters…') +
          '</div>' +
        '</div>' +
        '<div class="tabpane" id="' + id + '-pane-views">' +
          (window.ODViews ? window.ODViews.pane(id) : '') +
        '</div>' +
        '<div class="tabpane" id="' + id + '-pane-summary" style="padding-top:12px">' +
          '<p style="font-size:13px;color:var(--fg-2);margin-bottom:12px">'+
            'A readable view of what you recorded. Every summary states how much '+
            'of the underlying data it could actually cover.' +
          '</p>' +
          // Today / Yesterday / project and session pickers, and the output
          // area: od-summaries.js (loaded before this file).
          (window.ODSummaries ? window.ODSummaries.controls(id) : '') +
          // Knowledge Overview.
          //
          // The Today/Yesterday buttons above are a date filter over
          // atomic_facts and have never had any link to a cluster, so the
          // Summaries tab could not show what the store actually knows ABOUT
          // anything. This card fills that: cluster summaries, read from the
          // display-only table, which is where they belong now that they are
          // out of the retrieval corpus.
          '<div id="' + id + '-kover" style="margin-top:26px;' +
            'border-top:1px solid var(--border);padding-top:20px">' +
            _loading('Loading what your memory knows…') +
          '</div>' +
        '</div>' +
        // Recall Lab pane — exact IDs required by recall-lab.js:
        //   #recall-lab-query (input), #recall-lab-search (button — click check),
        //   #recall-lab-per-page (select, optional), #recall-lab-meta, #recall-lab-results.
        // Backend: POST /api/v3/recall/trace
        //
        // This is now the landing pane, so it has to explain itself to someone who
        // has never heard the word "recall" in this sense. An empty search box as a
        // first impression tells a non-technical user nothing about what SLM does.
        '<div class="tabpane active" id="' + id + '-pane-recall" style="padding-top:12px">' +
          '<div style="margin-bottom:14px">' +
            '<p style="font-size:13px;color:var(--fg-2);margin-bottom:10px">' +
              'Ask your memory a question and watch it answer. This shows you the ' +
              'same result an AI assistant gets — plus <em>why</em> each memory was ' +
              'chosen, how strongly it matched, and how long the lookup took.' +
            '</p>' +
            '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">' +
              '<input id="recall-lab-query" placeholder="Enter recall query…" autocomplete="off" ' +
                'style="flex:1;min-width:200px;padding:8px 12px;' +
                  'border:1px solid var(--border);border-radius:6px;font-size:13px;' +
                  'background:var(--bg-2);color:var(--fg)">' +
              '<select id="recall-lab-per-page" ' +
                'style="padding:7px 10px;border:1px solid var(--border);border-radius:6px;' +
                  'font-size:13px;background:var(--bg-2);color:var(--fg)">' +
                '<option value="5">5</option>' +
                '<option value="10" selected>10</option>' +
                '<option value="20">20</option>' +
                '<option value="50">50</option>' +
              '</select>' +
              // class="btn primary" (not an inline var(--accent) background —
              // --accent is not a defined token anywhere in this codebase, so
              // the background resolved to transparent and left white-on-white
              // text). `.btn.primary` is the design system's real primary-action
              // style already used elsewhere in the app; its own sizing matches
              // the input/select next to it.
              '<button id="recall-lab-search" class="btn primary" ' +
                'style="white-space:nowrap">Run Trace</button>' +
            '</div>' +
          '</div>' +
          // #recall-lab-meta — written by recall-lab.js before results (timing, count, etc.)
          '<div id="recall-lab-meta" ' +
            'style="font-size:12px;color:var(--fg-2);margin-bottom:8px"></div>' +
          // #recall-lab-results — written by recall-lab.js (result cards, pagination).
          // Seeded with a starter state because this pane loads first: a bare box
          // gives a first-time user no idea what to type. recall-lab.js replaces
          // the whole node on the first search, so this costs nothing afterwards.
          '<div id="recall-lab-results">' + _recallStarter() + '</div>' +
        '</div>' +
      '</div>'
    );
  }

  function _allScaffold(id) {
    return (
      '<div style="margin-bottom:16px;display:flex;gap:8px;align-items:center">' +
        '<input data-od-act="search" placeholder="Search all memories…" autocomplete="off" ' +
          'style="flex:1;padding:8px 12px;border:1px solid var(--border);' +
          'border-radius:var(--r-md);background:var(--card-2);color:var(--fg);' +
          'font-size:13.5px;outline:none">' +
        '<select data-od-act="window" title="Limit search to a time window" ' +
          'style="padding:8px 10px;border:1px solid var(--border);border-radius:var(--r-md);' +
          'background:var(--card-2);color:var(--fg);font-size:13px;outline:none">' +
          '<option value="">Any time</option>' +
          '<option value="24h">Last 24h</option>' +
          '<option value="7d">Last 7 days</option>' +
          '<option value="30d">Last 30 days</option>' +
          '<option value="90d">Last 90 days</option>' +
          '<option value="1y">Last year</option>' +
        '</select>' +
      '</div>' +
      // Filter bar: category chips (populated after cat-count fetch) + sort seg
      '<div id="' + id + '-cats" ' +
        'style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;align-items:center">' +
        '<div class="chip on" data-od-act="cat" data-cat="">All kinds</div>' +
        '<div style="flex:1"></div>' +
        // Scope view: Mine (this profile) / Shared with me / Global / Everything
        '<div class="seg" title="Which memories to show">' +
          '<button class="active" data-od-act="scope-view" data-scope="mine">Mine</button>' +
          '<button data-od-act="scope-view" data-scope="shared">Shared</button>' +
          '<button data-od-act="scope-view" data-scope="global">Global</button>' +
          '<button data-od-act="scope-view" data-scope="all">All</button>' +
        '</div>' +
        '<div class="seg">' +
          '<button class="active" data-od-act="sort" data-sort="created">Newest</button>' +
          '<button data-od-act="sort" data-sort="score">Score</button>' +
          '<button data-od-act="sort" data-sort="importance">Importance</button>' +
        '</div>' +
      '</div>' +
      '<div class="card" id="' + id + '-tbl-wrap">' + _loading('Loading memories…') + '</div>' +
      '<div id="' + id + '-pg" style="margin-top:10px"></div>'
    );
  }

  function _tlScaffold(id) {
    return (
      '<div class="card">' +
        '<div class="card-head">' +
          '<h3>Memory creation timeline</h3>' +
          '<span class="sub" id="' + id + '-tl-sub">facts stored per day</span>' +
          '<div style="flex:1"></div>' +
          '<div class="seg">' +
            '<button data-od-act="tl-range" data-range="7d">7d</button>' +
            '<button class="active" data-od-act="tl-range" data-range="30d">30d</button>' +
          '</div>' +
        '</div>' +
        '<div class="card-pad">' +
          '<div class="bars" id="' + id + '-bars" style="height:120px">' +
            _loading('Loading timeline…') +
          '</div>' +
          '<div style="display:flex;justify-content:space-between;margin-top:10px;' +
            'font-size:11.5px;color:var(--fg-3)">' +
            '<span id="' + id + '-tl-start"></span><span>today</span>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="card" style="margin-top:16px">' +
        '<div class="card-head"><h3>By fact type</h3></div>' +
        '<div class="card-pad" id="' + id + '-ftypes">' + _loading('Loading breakdown…') + '</div>' +
      '</div>'
    );
  }

  function _loading(msg) {
    return '<div style="padding:40px;text-align:center;color:var(--fg-2);font-size:13px">' +
      _esc(msg) + '</div>';
  }

  /* Starter state for Recall Lab.
   *
   * Recall Lab is the landing pane, so this is the first thing most people see in
   * the dashboard. It has to answer "what is this and what do I type" without
   * assuming the reader knows what a recall trace is.
   *
   * The examples are deliberately static. Seeding them from real memories was the
   * first idea and it is worse: stored content is things like
   * "[claude][SLM 4.0.8 CHECKPOINT] RELEASE STATE: local", which teaches exactly
   * the wrong query shape. These teach the shape; the store answers.
   */
  var RECALL_EGS = [
    'what did I decide about the database',
    'what am I working on',
    'what problems have I hit before',
  ];

  function _recallStarter() {
    var chips = RECALL_EGS.map(function (q) {
      return '<button class="tab" data-od-act="recall-eg" data-q="' + _esc(q) + '">' +
        _esc(q) + '</button>';
    }).join('');
    return (
      '<div id="od-recall-starter" style="border:1px dashed var(--border);' +
        'border-radius:var(--r-md);padding:22px;color:var(--fg-2);font-size:13px">' +
        '<div style="font-weight:600;color:var(--fg);margin-bottom:6px">' +
          'Ask a question above to search your memory</div>' +
        '<p style="margin:0 0 14px;line-height:1.55">' +
          'It searches by meaning, not keywords — asking “what database are we using” ' +
          'can surface a memory that says “switched to Postgres”, even though the two ' +
          'share no words. Try one:' +
        '</p>' +
        '<div style="display:flex;gap:8px;flex-wrap:wrap">' + chips + '</div>' +
      '</div>'
    );
  }

  // ── Tab switching ────────────────────────────────────────────────────────────

  function _switchTab(id, tab) {
    _st = Object.assign({}, _st, { activeTab: tab });
    var root = document.getElementById(id);
    if (!root) return;
    root.querySelectorAll('#' + id + '-tabs .tab').forEach(function (b) {
      b.classList.toggle('active', b.dataset.tab === tab);
    });
    root.querySelectorAll('.tabpane').forEach(function (p) { p.classList.remove('active'); });
    var pane = document.getElementById(id + '-pane-' + tab);
    if (pane) pane.classList.add('active');
    if (tab === 'timeline') _loadTimeline(id);
    if (tab === 'clusters')  _loadClusters(id);
    // The Summaries and Saved views panes live in od-summaries.js / od-views.js.
    if (tab === 'summary') {
      if (window.ODSummaries) window.ODSummaries.onShow(id);
      _loadKnowledgeOverview(id);
    }
    if (tab === 'views' && window.ODViews) window.ODViews.onShow(id);
  }

  // ── Kind counts (pre-fetch real totals) ──────────────────────────────────────

  function _loadCatCounts(id) {
    // One request classifies every visible row by its DISPLAYED kind
    // (server-side kind_fields, same precedence as `slm list`/recall) —
    // not nine (or, before 4.1.21, five) separate ?category=X round trips,
    // which could only ever count legacy fact_type buckets.
    fetch('/api/memories/kind-counts')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        _st = Object.assign({}, _st, { catCounts: d.counts || {} });
        _rebuildCatBar(id);
      })
      .catch(function () {
        _st = Object.assign({}, _st, { catCounts: {} });
        _rebuildCatBar(id);
      });
  }

  // Rebuild the full filter bar (chips + sort) from current _st
  function _rebuildCatBar(id) {
    var bar = document.getElementById(id + '-cats');
    if (!bar) return;
    var active = _st.category || '';
    var html = '<div class="chip' + (!active ? ' on' : '') +
      '" data-od-act="cat" data-cat="">All kinds</div>';
    KNOWN_KINDS.forEach(function (k) {
      var count = _st.catCounts[k.kind] || 0;
      if (count <= 0) return;
      html += '<div class="chip' + (_st.category === k.kind ? ' on' : '') +
        '" data-od-act="cat" data-cat="' + _esc(k.kind) + '">' +
        _esc(k.label) + ' <span class="cnt">' +
        count.toLocaleString() + '</span></div>';
    });
    var sv = _st.scopeView || 'mine';
    html += '<div style="flex:1"></div>' +
      '<div class="seg" title="Which memories to show">' +
        ['mine', 'shared', 'global', 'all'].map(function (s) {
          var label = { mine: 'Mine', shared: 'Shared', global: 'Global', all: 'All' }[s];
          return '<button class="' + (sv === s ? 'active' : '') +
            '" data-od-act="scope-view" data-scope="' + s + '">' + label + '</button>';
        }).join('') +
      '</div>' +
      '<div class="seg">' +
        '<button class="' + (_st.sort === 'created' ? 'active' : '') +
          '" data-od-act="sort" data-sort="created">Newest</button>' +
        '<button class="' + (_st.sort === 'score' ? 'active' : '') +
          '" data-od-act="sort" data-sort="score">Score</button>' +
        '<button class="' + (_st.sort === 'importance' ? 'active' : '') +
          '" data-od-act="sort" data-sort="importance">Importance</button>' +
      '</div>';
    bar.innerHTML = html;
  }

  // ── Memories fetch & render ──────────────────────────────────────────────────

  function _loadMem(id) {
    var url = '/api/memories?limit=' + _st.pageSize +
      '&offset=' + (_st.page * _st.pageSize);
    if (_st.category) url += '&kind=' + encodeURIComponent(_st.category);
    if (_st.scopeView && _st.scopeView !== 'mine') {
      url += '&scope=' + encodeURIComponent(_st.scopeView);
    }
    var requestSeq = _st.requestSeq + 1;
    _st = Object.assign({}, _st, { requestSeq: requestSeq });
    var wrap = document.getElementById(id + '-tbl-wrap');
    if (wrap) wrap.innerHTML = _loading('Loading memories…');

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        if (_st.rootId !== id || _st.requestSeq !== requestSeq) return;
        _st = Object.assign({}, _st, { memories: d.memories || [], total: d.total || 0 });
        _renderTable(id, d);
      })
      .catch(function (err) {
        if (_st.rootId !== id || _st.requestSeq !== requestSeq) return;
        var w = document.getElementById(id + '-tbl-wrap');
        if (w) w.innerHTML = '<div class="card-pad" style="color:var(--danger);text-align:center">' +
          'Failed to load memories: ' + _esc(err.message) + '</div>';
      });
  }

  function _renderTable(id, d) {
    var mems   = d.memories || [];
    var total  = d.total || 0;
    var wrap   = document.getElementById(id + '-tbl-wrap');
    var cntEl  = document.getElementById(id + '-cnt-all');
    var subEl  = document.getElementById(id + '-sub');

    if (cntEl) cntEl.textContent = total.toLocaleString();
    if (subEl) {
      var _projs = mems.reduce(function (s, m) {
        if (m.project_name) s.add(m.project_name);
        return s;
      }, new Set()).size;
      var _pStr = _projs > 0
        ? ' across ' + _projs + ' project' + (_projs === 1 ? '' : 's')
        : '';
      subEl.textContent = total.toLocaleString() + ' memories' + _pStr +
        '. Every memory shows its provenance — why it was remembered and how strongly it scored.';
    }
    if (!wrap) return;

    if (mems.length === 0) {
      wrap.innerHTML = '<div class="card-pad" style="text-align:center;padding:40px;' +
        'color:var(--fg-2)">No memories found. Try a different filter.</div>';
      _renderPag(id, d);
      return;
    }

    var rows = mems.map(function (m, idx) {
      var imp      = _imp(m.importance);
      var impCls   = imp >= 8 ? 'ok' : imp >= 5 ? 'warn' : 'neutral';
      var scorePct = Math.round((parseFloat(m.importance) || 0) * 100);
      var scoreCls = _scoreCls(m.importance);
      // memory_kind_label (server: storage/memory_kinds.py kind_fields(),
      // the same confirmed/suggested/legacy-fallback precedence `slm list`
      // and recall use) over the raw legacy category — a v2 store with no
      // kind support at all still falls back to m.category.
      var kind     = m.memory_kind || m.category || 'semantic';
      var catLabel = m.memory_kind_label || m.category || 'semantic';
      var preview  = (m.content || '').substring(0, 120);
      if ((m.content || '').length > 120) preview += '…';

      return '<tr class="row" data-od-act="drawer" data-idx="' + idx + '" style="cursor:pointer">' +
        '<td style="max-width:420px">' +
          '<b class="mono dim" style="font-size:11px">#' +
            _esc((m.id || '').substring(0, 8)) +
          '</b>' +
          '<div style="margin-top:2px">' + _esc(preview) + '</div>' +
        '</td>' +
        '<td><span class="badge ' + _esc(_catCls(kind)) + '">' + _esc(catLabel) + '</span></td>' +
        '<td class="mono dim" style="font-size:12px">' + _esc(m.project_name || '—') + '</td>' +
        '<td><b class="num">' + imp + '</b>/10</td>' +
        '<td><span class="badge ' + _esc(scoreCls) + '">' + scorePct + '%</span></td>' +
        '<td class="mono dim" style="font-size:12px">' + _esc(_fmt(m.created_at)) + '</td>' +
      '</tr>';
    }).join('');

    wrap.innerHTML =
      '<table class="tbl">' +
        '<thead><tr>' +
          '<th>Memory</th><th>Kind</th><th>Project</th>' +
          '<th>Importance</th><th>Score</th><th>Created</th>' +
        '</tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>';

    _renderPag(id, d);
  }

  function _renderPag(id, d) {
    var el   = document.getElementById(id + '-pg');
    if (!el) return;
    var tot  = d.total || 0;
    var lim  = d.limit || _st.pageSize;
    var off  = d.offset || (_st.page * _st.pageSize);
    var pg   = Math.floor(off / lim);
    var last = Math.max(0, Math.ceil(tot / lim) - 1);
    var show = Math.min(off + lim, tot);
    var from = tot === 0 ? 0 : off + 1;

    el.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:center;' +
        'font-size:12.5px;color:var(--fg-2)">' +
        '<span>Showing ' + from + '–' + show + ' of ' + tot.toLocaleString() + '</span>' +
        '<div class="seg">' +
          '<button ' + (pg <= 0 ? 'disabled style="opacity:.4"' :
            'data-od-act="pg" data-page="' + (pg - 1) + '"') + '>← Prev</button>' +
          '<button disabled style="opacity:.7;cursor:default">Page ' +
            (pg + 1) + ' / ' + (last + 1) + '</button>' +
          '<button ' + (pg >= last ? 'disabled style="opacity:.4"' :
            'data-od-act="pg" data-page="' + (pg + 1) + '"') + '>Next →</button>' +
        '</div>' +
      '</div>';
  }

  // ── Sort (client-side on current page) ──────────────────────────────────────

  function _doSort(id, by) {
    _st = Object.assign({}, _st, { sort: by });
    var mems = _st.memories.slice();
    // 'score' and 'importance' both rank by the same underlying importance float
    if (by === 'importance' || by === 'score') {
      mems.sort(function (a, b) {
        return (parseFloat(b.importance) || 0) - (parseFloat(a.importance) || 0);
      });
    } else {
      mems.sort(function (a, b) {
        return String(b.created_at || '').localeCompare(String(a.created_at || ''));
      });
    }
    _st = Object.assign({}, _st, { memories: mems });
    _renderTable(id, {
      memories: mems, total: _st.total,
      limit: _st.pageSize, offset: _st.page * _st.pageSize,
    });
  }

  // ── Search — POST /api/search (full corpus, 0.5–1.4 s warm) ─────────────────

  var _searchTimer = null;
  function _doSearch(id, q) {
    clearTimeout(_searchTimer);
    var requestSeq = _st.requestSeq + 1;
    // /api/search is a full-corpus semantic query. It cannot truthfully retain
    // the list endpoint's category/scope selection, so clear those controls
    // before the request rather than leaving the table and chips disagreeing.
    _st = Object.assign({}, _st, {
      searchQ: q || null,
      category: q ? null : _st.category,
      scopeView: q ? 'mine' : _st.scopeView,
      page: 0,
      requestSeq: requestSeq,
    });
    if (q) _rebuildCatBar(id);
    _searchTimer = setTimeout(function () {
      if (_st.rootId !== id || _st.requestSeq !== requestSeq) return;
      if (!q) { _loadMem(id); return; }
      var wrap = document.getElementById(id + '-tbl-wrap');
      if (wrap) wrap.innerHTML = _loading('Searching…');
      fetch('/api/search', { method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        slmInvalidatesCache: false,
        slmRequiresWriteAuth: false,
        body: JSON.stringify({ query: q, limit: 50, window: _st.window || '' })
      }).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (d) {
        if (_st.rootId !== id || _st.requestSeq !== requestSeq) return;
        var mems = (d.results || []).map(_normSearch);
        _st = Object.assign({}, _st, { memories: mems, total: mems.length });
        _renderTable(id, { memories: mems, total: mems.length, limit: mems.length, offset: 0 });
      }).catch(function (err) {
        if (_st.rootId !== id || _st.requestSeq !== requestSeq) return;
        var w = document.getElementById(id + '-tbl-wrap');
        if (w) w.innerHTML = '<div class="card-pad" style="color:var(--danger);text-align:center;' +
          'padding:24px">Search failed: ' + _esc(err.message) + '</div>';
      });
    }, 350);
  }

  // ── Timeline fetch & render ──────────────────────────────────────────────────

  var _tlLoaded = {};
  function _loadTimeline(id) {
    var range = _st.tlRange;
    if (_tlLoaded[id + '-' + range]) return;
    var barsEl  = document.getElementById(id + '-bars');
    var ftEl    = document.getElementById(id + '-ftypes');
    var startEl = document.getElementById(id + '-tl-start');

    fetch('/api/v3/timeline/?range=' + range + '&group_by=category&limit=1000')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var events = d.events || [];
        _tlLoaded[id + '-' + range] = true;
        if (events.length === 0) {
          if (barsEl) barsEl.innerHTML =
            '<div style="display:grid;place-items:center;width:100%;height:100%;' +
            'color:var(--fg-3);font-size:13px">No memory events in this range.</div>';
          if (ftEl) ftEl.innerHTML =
            '<div style="color:var(--fg-3);font-size:13px">No category data available.</div>';
          return;
        }
        _renderBars(id, events, range, barsEl, startEl);
        _renderFtypes(id, events, ftEl);
      })
      .catch(function (err) {
        if (barsEl) barsEl.innerHTML =
          '<div style="color:var(--fg-3);font-size:13px;text-align:center;padding:16px">' +
          'Timeline unavailable: ' + _esc(err.message) + '</div>';
        // TODO: endpoint returns 404 when timeline feature is disabled
      });
  }

  function _renderBars(id, events, range, barsEl, startEl) {
    if (!barsEl) return;
    var days   = range === '30d' ? 30 : 7;
    var today  = new Date();
    var counts = new Array(days).fill(0);
    events.forEach(function (ev) {
      var diff = Math.floor((today - new Date(ev.timestamp)) / 86400000);
      if (diff >= 0 && diff < days) counts[days - 1 - diff]++;
    });
    if (typeof slmBars === 'function') {
      slmBars(barsEl, counts);
    } else {
      var maxVal = Math.max.apply(null, counts) || 1;
      barsEl.innerHTML = counts.map(function (v) {
        return '<i style="height:' + Math.max(4, (v / maxVal) * 100) + '%' +
          '" title="' + v + ' events"></i>';
      }).join('');
    }
    if (startEl) startEl.textContent = days + 'd ago';
    var subEl = document.getElementById(id + '-tl-sub');
    if (subEl) subEl.textContent = events.length.toLocaleString() + ' events · last ' + days + ' days';
  }

  function _renderFtypes(id, events, ftEl) {
    if (!ftEl) return;
    var catMap = {};
    events.forEach(function (ev) {
      catMap[ev.category || 'unknown'] = (catMap[ev.category || 'unknown'] || 0) + 1;
    });
    var total  = events.length;
    var colors = { temporal:'var(--ok)', semantic:'var(--violet)', episodic:'var(--cyan)', opinion:'var(--warn)' };
    var sorted = Object.keys(catMap).sort(function (a, b) { return catMap[b] - catMap[a]; });
    ftEl.innerHTML = sorted.map(function (cat) {
      var pct   = Math.round(catMap[cat] / total * 100);
      var color = colors[cat] || 'var(--fg-3)';
      return '<div style="margin-bottom:13px">' +
        '<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">' +
          '<span class="mono">' + _esc(cat) + '</span><b>' + pct + '%</b>' +
        '</div>' +
        '<div class="meter"><i style="width:' + pct + '%;background:' + color + '"></i></div>' +
      '</div>';
    }).join('');
  }

  // ── Clusters fetch & render ──────────────────────────────────────────────────

  /* Knowledge Overview — cluster summaries, plus an honest health line.
   *
   * Two rules this card exists to obey.
   *
   * IT READS ONLY THE DISPLAY TABLE. These summaries were in the retrieval
   * corpus until 4.0.10, where they out-ranked the user's own words. Moving
   * them out is only worth anything if exactly one surface shows them, and
   * this is that surface.
   *
   * IT DOES NOT DRESS UP JUNK AS INSIGHT. Some stored summaries are a model's
   * non-answer -- "Unfortunately, there is no information available about
   * 'State'..." -- and rendering those as a heading called Knowledge Overview
   * makes the product look broken. The endpoint labels each row's quality; the
   * usable ones are shown and the rest are counted in one plain sentence. Not
   * hidden: hiding them would hide the very problem the release fixes, and a
   * reader who is told "3 could not be generated" trusts the other 47 more.
   */
  function _loadKnowledgeOverview(id) {
    var box = document.getElementById(id + '-kover');
    if (!box) return;
    // Guarded on the element, not a module flag — same reason as the
    // summary pickers: a stale "already loaded" boolean outlives the DOM
    // it described and leaves a permanent spinner after a re-render.
    if (box.dataset.loaded === '1') return;

    Promise.all([
      fetch('/api/v3/abstraction/consolidated?limit=24')
        .then(function (r) { return r.ok ? r.json() : { summaries: [], unusable: 0 }; })
        .catch(function () { return { summaries: [], unusable: 0 }; }),
      fetch('/api/v3/abstraction/health')
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { return null; })
    ]).then(function (res) {
      box.dataset.loaded = '1';
      box.innerHTML = _koverHtml(res[0] || {}, res[1]);
    });
  }

  function _koverHtml(data, health) {
    var all = data.summaries || [];
    var usable = all.filter(function (s) { return s.quality === 'ok'; });
    var unusable = typeof data.unusable === 'number' ? data.unusable
                 : (all.length - usable.length);

    var html = '<h3 style="font-size:15px;margin:0 0 4px">What your memory knows</h3>' +
      '<p style="font-size:12px;color:var(--fg-2);margin:0 0 14px">' +
        'Written by the summarizer from groups of your memories. These are a ' +
        'view of your memory, not memories themselves — they are never returned ' +
        'as answers.' +
      '</p>';

    if (health) html += _koverHealth(health);

    if (usable.length === 0) {
      html += '<div style="padding:20px;color:var(--fg-2);font-size:13px">' +
        (all.length === 0
          ? 'No summaries yet. They are written in the background as related ' +
            'memories accumulate.'
          : 'None of the ' + all.length + ' stored summaries came back usable. ' +
            'That usually means the summarizer had nothing in common to merge.') +
        '</div>';
      return html;
    }

    html += '<div style="display:grid;gap:12px;' +
      'grid-template-columns:repeat(auto-fill,minmax(280px,1fr))">';
    usable.forEach(function (s) { html += _koverCard(s); });
    html += '</div>';

    var notes = [];
    if (unusable > 0) {
      notes.push(unusable + (unusable === 1 ? ' summary' : ' summaries') +
        ' could not be generated and are not shown. The memories they were ' +
        'built from are unaffected.');
    }
    if (data.near_duplicates > 0) {
      notes.push(data.near_duplicates + ' near-identical ' +
        (data.near_duplicates === 1 ? 'summary was' : 'summaries were') +
        ' collapsed into the ones above.');
    }
    if (notes.length) {
      html += '<p style="font-size:12px;color:var(--fg-2);margin-top:12px">' +
        notes.map(_esc).join(' ') + '</p>';
    }
    return html;
  }

  function _koverCard(s) {
    // Every field is escaped. Anything in a memory can reach this screen: the
    // summarizer merges fact content verbatim, so a memory containing markup
    // would otherwise render as markup.
    var body = _koverPlain(s.content);
    var clipped = body.length > 320;
    var shown = clipped ? body.slice(0, 320).replace(/\s+\S*$/, '') + '…' : body;

    var meta = [];
    if (s.source_count) {
      meta.push(s.source_count + (s.source_count === 1 ? ' memory' : ' memories'));
    }
    var span = _koverSpan(s.source_earliest, s.source_latest);
    if (span) meta.push(span);

    return '<div style="background:var(--card-2);border:1px solid var(--border);' +
        'border-radius:var(--r-md);padding:14px">' +
      (s.entity_name
        ? '<div style="font-size:13px;font-weight:600;margin-bottom:6px">' +
            _esc(s.entity_name) + '</div>'
        : '') +
      '<div style="font-size:13px;line-height:1.5">' + _esc(shown) + '</div>' +
      (meta.length
        ? '<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">' +
            meta.map(_koverChip).join('') +
          '</div>'
        : '') +
    '</div>';
  }

  /* Flatten a summary to plain prose for display.
   *
   * Mode B/C summaries come back with markdown in them — "**Audit Round**",
   * "**Findings**:" — because a chat-tuned model formats for a chat window.
   * Rendered as text those asterisks read as broken output to exactly the
   * reader this card is for. Escaping is still what protects the page; this
   * only removes the emphasis marks that carry no meaning here.
   */
  function _koverPlain(text) {
    return String(text || '')
      .replace(/```[\s\S]*?```/g, ' ')     // fenced blocks: no room for them
      .replace(/`([^`]*)`/g, '$1')          // inline code marks
      .replace(/\*\*([^*]+)\*\*/g, '$1')   // bold
      .replace(/(^|\s)\*([^*\s][^*]*)\*/g, '$1$2')  // italic, not bullets
      .replace(/^#{1,6}\s+/gm, '')          // headings
      .replace(/^\s*[-*+]\s+/gm, '')        // list bullets
      .replace(/\s+/g, ' ')                 // collapse whitespace + newlines
      .trim();
  }

  /* A chip, not a keyword dump.
   *
   * The neighbouring community-summary API returns its `summary` field as
   * "Topics: CCQ, Resume, P07…", which printed verbatim reads as a broken
   * page. Counts and date spans are rendered as discrete labels instead.
   */
  function _koverChip(text) {
    return '<span style="font-size:11px;padding:2px 8px;border-radius:999px;' +
      'background:var(--card);border:1px solid var(--border);color:var(--fg-2)">' +
      _esc(text) + '</span>';
  }

  function _koverSpan(earliest, latest) {
    var a = String(earliest || '').slice(0, 10);
    var b = String(latest || '').slice(0, 10);
    if (!a && !b) return '';
    if (!a || !b || a === b) return a || b;
    return a + ' to ' + b;
  }

  /* Whether the memories behind all this can actually be found.
   *
   * The one question no screen answered before. A machine ran for months with
   * 43.7% of its store unreachable by meaning while every status line it showed
   * said healthy, because nothing counted the reachable share.
   */
  function _koverHealth(h) {
    var lines = h.summary || [];
    if (!lines.length) return '';
    var bad = h.healthy === false;
    return '<div style="margin:0 0 16px;padding:12px 14px;border-radius:var(--r-md);' +
        'background:var(--card-2);border-left:3px solid ' +
        (bad ? 'var(--warn)' : 'var(--ok)') + '">' +
      lines.map(function (l) {
        return '<div style="font-size:12px;color:var(--fg-2);line-height:1.5">' +
          _esc(l) + '</div>';
      }).join('') +
    '</div>';
  }

  function _loadClusters(id) {
    if (_st.cluLoaded) return;
    var grid = document.getElementById(id + '-clu-grid');
    if (!grid) return;
    fetch('/api/clusters')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        _st = Object.assign({}, _st, { cluLoaded: true });
        var clus  = d.clusters || [];
        var cntEl = document.getElementById(id + '-cnt-clusters');
        if (cntEl) cntEl.textContent = clus.length.toLocaleString();
        if (clus.length === 0) {
          grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:40px;' +
            'color:var(--fg-2)">No clusters yet. Clusters form as memories accumulate.</div>';
          return;
        }
        _renderClusters(id, clus, grid);
      })
      .catch(function (err) {
        if (grid) grid.innerHTML =
          '<div style="grid-column:1/-1;text-align:center;padding:24px;' +
          'color:var(--danger);font-size:13px">Failed to load clusters: ' +
          _esc(err.message) + '</div>';
      });
  }

  var _PAL = ['var(--violet)', 'var(--cyan)', 'var(--warn)', 'var(--ok)', 'var(--danger)'];

  function _renderClusters(id, clus, grid) {
    grid.innerHTML = clus.map(function (c, i) {
      var color   = _PAL[i % _PAL.length];
      var summary = c.summary || c.categories || '';
      var preview = summary.length > 80 ? summary.substring(0, 80) + '…' : summary;
      var cid     = String(c.cluster_id || '');
      return '<div class="launch-card card" style="cursor:pointer"' +
        ' data-od-act="expand-cluster" data-cid="' + _esc(cid) + '">' +
        '<div class="ic" style="background:color-mix(in srgb,' + color + ' 15%,transparent);color:' + color + '">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="21" height="21">' +
            '<path d="M12 3l1.9 4.6L18 9l-3.5 3 1 5-3.5-2.5L8.5 17l1-5L6 9l4.1-1.4z"/>' +
          '</svg>' +
        '</div>' +
        '<h3>' + _esc(preview) + '</h3>' +
        '<p><b class="num">' + _esc(String(c.member_count)) + '</b> memories · expand →</p>' +
        '<div style="display:none;margin-top:14px;border-top:1px solid var(--border);' +
          'padding-top:12px;font-size:13px" id="' + id + '-clu-' + _esc(cid) + '"></div>' +
      '</div>';
    }).join('');
  }

  function _expandCluster(id, cid) {
    var detEl = document.getElementById(id + '-clu-' + cid);
    if (!detEl) return;
    if (detEl.style.display !== 'none') { detEl.style.display = 'none'; return; }
    detEl.style.display = 'block';
    if (detEl.dataset.loaded) return;
    detEl.dataset.loaded = '1';
    detEl.textContent = 'Loading members…';
    fetch('/api/clusters/' + encodeURIComponent(cid) + '?limit=5')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var members = d.members || [];
        if (members.length === 0) { detEl.textContent = 'No members found.'; return; }
        detEl.innerHTML = members.map(function (m, i) {
          var txt = m.content || m.summary || '';
          return '<div style="margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid var(--border)">' +
            '<div style="font-size:12.5px">' + (i + 1) + '. ' +
              _esc(txt.substring(0, 120)) + '</div></div>';
        }).join('');
      })
      .catch(function () { detEl.textContent = 'Failed to load members.'; });
  }

  // ── Provenance drawer ────────────────────────────────────────────────────────

  function _ensureDrawer() {
    if (document.getElementById('od-mem-drawer')) return;
    var scrim = document.createElement('div');
    scrim.id = 'od-mem-drawer-scrim';
    scrim.className = 'od-drawer-scrim';
    scrim.dataset.odAct = 'close-drawer';
    document.body.appendChild(scrim);
    var drawer = document.createElement('aside');
    drawer.id = 'od-mem-drawer';
    drawer.className = 'od-drawer';
    document.body.appendChild(drawer);
  }

  function _openDrawer(idx) {
    var mem = _st.memories[idx];
    if (!mem) return;
    _ensureDrawer();
    var drawer = document.getElementById('od-mem-drawer');
    var scrim  = document.getElementById('od-mem-drawer-scrim');
    if (!drawer || !scrim) return;

    var imp      = _imp(mem.importance);
    var kind     = mem.memory_kind || mem.category || 'semantic';
    var catLabel = mem.memory_kind_label || mem.category || 'semantic';
    var scorePct = Math.round((parseFloat(mem.importance) || 0) * 100);

    drawer.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:flex-start">' +
        '<div>' +
          '<div style="font-size:12px;color:var(--fg-2)">Memory</div>' +
          '<h2 style="font-size:19px;margin-top:2px">' +
            _esc('#' + (mem.id || '').substring(0, 12)) + '</h2>' +
        '</div>' +
        '<button class="btn icon ghost" data-od-act="close-drawer">✕</button>' +
      '</div>' +
      '<div class="od-prov" style="font-size:14px;line-height:1.7">' +
        _esc(mem.content || '') +
      '</div>' +
      '<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">' +
        '<span class="badge ' + _esc(_catCls(kind)) + '">' + _esc(catLabel) + '</span>' +
        '<span class="badge neutral">' + _esc(mem.project_name || 'no project') + '</span>' +
        '<span class="badge ' + (mem.scope === 'global' ? 'success' : mem.scope === 'shared' ? 'warn' : 'neutral') +
          '">' + _esc(mem.scope || 'personal') + '</span>' +
      '</div>' +
      // Scope & sharing control (C2) — set personal/shared/global from the UI
      '<h3 style="font-size:13px;margin:20px 0 8px">Scope &amp; sharing</h3>' +
      '<div class="od-prov" style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">' +
        '<select id="od-scope-sel" style="padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
          ['personal', 'shared', 'global'].map(function (s) {
            return '<option value="' + s + '"' + ((mem.scope || 'personal') === s ? ' selected' : '') + '>' + s + '</option>';
          }).join('') +
        '</select>' +
        '<input id="od-scope-shared" placeholder="share with profiles (comma-sep)" ' +
          'value="' + _esc(_sharedToStr(mem.shared_with)) + '" ' +
          'style="flex:1;min-width:140px;padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
        '<button data-od-act="set-scope" data-mid="' + _esc(mem.id) + '" ' +
          'style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Apply</button>' +
      '</div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Why this was remembered</h3>' +
      '<div class="od-prov">' +
        '<div style="font-size:13px;color:var(--fg-2);line-height:1.6">' +
          'Importance: <b>' + imp + '/10</b>. ' +
          'Score: <b>' + scorePct + '%</b>. ' +
          'Recalled <b>' + _esc(String(mem.access_count || 0)) + '</b> time' +
          (mem.access_count === 1 ? '' : 's') + '. ' +
          'Stored ' + _esc(_ago(mem.created_at)) + '.' +
        '</div>' +
        // TODO: /api/memories/{id}/score-breakdown — Semantic/BM25/EntityGraph/Temporal
        //       decomposition is not yet exposed by the daemon; show when available.
      '</div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Atomic facts</h3>' +
      '<div id="od-drawer-facts" style="color:var(--fg-3);font-size:13px">Loading facts…</div>' +
      '<div id="od-drawer-code"></div>' +
      '<div style="display:flex;gap:8px;margin-top:20px;padding-top:12px;border-top:1px solid var(--border)">' +
        '<button data-od-act="edit-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Edit</button>' +
        '<button data-od-act="del-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--danger);border-radius:6px;background:transparent;color:var(--danger);cursor:pointer;font-size:13px">Delete</button>' +
      '</div><div id="od-mem-act" style="margin-top:10px"></div>';

    drawer.classList.add('open');
    scrim.classList.add('on');
    // A row in this pane is an ATOMIC FACT, not a source memory: /api/memories
    // reports total 3603 on a store holding 830 memories and 3603 facts, and
    // every row id resolves through /api/facts/{id}. The row carries its parent
    // separately as memory_id.
    //
    // "Atomic facts" therefore needs the PARENT id — it lists the other facts
    // extracted from the same source memory. Passing mem.id asked
    // /api/memories/{fact_id}/facts, which has no children to return, so this
    // section read "No atomic facts recorded for this memory" for every memory
    // on every store. Not an empty state: the wrong identifier.
    if (mem.memory_id) {
      _loadFacts(mem.memory_id);
    } else {
      // No parent row: this record was stored directly as a fact rather than
      // extracted from a longer memory (198 of 3,608 on a real store). Saying
      // "no atomic facts recorded" there is wrong in a way that matters — it
      // reads as data missing, when the fact IS the record.
      var fx = document.getElementById('od-drawer-facts');
      if (fx) fx.textContent = 'Stored directly — this record is itself the atomic fact.';
    }
    // Code links hang off the FACT, so this one takes the row id.
    if (mem.id) _loadCodeLinks(mem.id);
  }

  function _closeDrawer() {
    var d = document.getElementById('od-mem-drawer');
    var s = document.getElementById('od-mem-drawer-scrim');
    if (d) d.classList.remove('open');
    if (s) s.classList.remove('on');
  }

  function _loadFacts(memId) {
    var el = document.getElementById('od-drawer-facts');
    if (!el) return;
    fetch('/api/memories/' + encodeURIComponent(memId) + '/facts')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var facts = d.facts || [];
        if (facts.length === 0) {
          el.textContent = 'No atomic facts recorded for this memory.'; return;
        }
        el.innerHTML = facts.map(function (f) {
          return '<div class="od-prov" style="margin-bottom:8px">' +
            '<span class="badge neutral" style="font-size:10px">' +
              _esc(f.fact_type || 'fact') +
            '</span>' +
            '<div style="font-size:13px;line-height:1.6;margin-top:4px">' +
              _esc((f.content || '').substring(0, 200)) +
            '</div></div>';
        }).join('');
      })
      .catch(function () { if (el) el.textContent = 'Could not load facts.'; });
  }

  /* Code this memory refers to (4.0.8).
   *
   * Renders into the memory drawer — the panel actually used to inspect a
   * memory. 4.0.7 put this in js/fact-detail.js, which only binds to
   * `.fact-result-item` rows in the search-results view, so the feature was
   * invisible to anyone browsing Memories.
   *
   * KEYED ON THE ROW'S OWN ID, which is an ATOMIC FACT id. The Memories pane
   * lists atomic facts — /api/memories returns total 3603 on a store with 830
   * memories and 3603 facts, and each row id resolves through /api/facts/{id}.
   * (That mismatch is also why the drawer's "Atomic facts" block always reads
   * "No atomic facts recorded": it asks /api/memories/{fact_id}/facts, which
   * has no children to return. Left alone here — separate defect, separate fix.)
   *
   * Reads /api/facts/{id}, which serves links from code_graph.db. Recall never
   * opens that database, so nothing here can affect retrieval.
   *
   * textContent throughout: every value is a qualified name or path read out of
   * an indexed repository, i.e. untrusted input as far as this panel goes.
   */
  function _loadCodeLinks(factId) {
    var host = document.getElementById('od-drawer-code');
    if (!host || !factId) return;
    fetch('/api/facts/' + encodeURIComponent(factId))
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var links = (d && d.code_links) || [];
        if (!links.length) return;   // silent when there is nothing to show

        var wrap = document.createElement('div');
        wrap.style.cssText = 'margin-top:6px;padding-top:6px;border-top:1px dashed var(--border)';

        var label = document.createElement('div');
        label.style.cssText = 'font-size:11px;color:var(--fg-3);margin-bottom:4px';
        label.textContent = 'Code referenced (' + links.length + ')';
        wrap.appendChild(label);

        links.slice(0, 6).forEach(function (l) {
          var row = document.createElement('div');
          row.style.cssText = 'font-size:12px;line-height:1.5';

          var code = document.createElement('code');
          code.style.cssText = 'color:var(--violet)';
          code.textContent = l.qualified_name || l.name || '?';
          row.appendChild(code);

          if (l.kind) {
            var kind = document.createElement('span');
            kind.style.cssText = 'color:var(--fg-3);font-size:10px;margin-left:6px';
            kind.textContent = l.kind;
            row.appendChild(kind);
          }
          // A stale link points at code that has moved or gone. Saying so is the
          // entire value of tracking staleness.
          if (l.is_stale) {
            var stale = document.createElement('span');
            stale.className = 'badge';
            stale.style.cssText = 'font-size:9px;margin-left:6px';
            stale.textContent = 'code changed';
            row.appendChild(stale);
          }
          wrap.appendChild(row);
        });

        if (links.length > 6) {
          var more = document.createElement('div');
          more.style.cssText = 'font-size:11px;color:var(--fg-3);margin-top:2px';
          more.textContent = '… and ' + (links.length - 6) + ' more';
          wrap.appendChild(more);
        }
        host.appendChild(wrap);
      })
      .catch(function () { /* display extra: never surface an error here */ });
  }

  function _startEdit(mid) { // show inline edit form in drawer
    var m = _st.memories.filter(function (x) { return x.id === mid; })[0];
    var el = document.getElementById('od-mem-act'); if (!el || !m) return;
    el.innerHTML = '<textarea id="od-ea" rows="4" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);font-size:13px">' + _esc(m.content) + '</textarea><button data-od-act="save-edit" data-mid="' + _esc(mid) + '" style="display:block;margin-top:6px;padding:6px 14px;border:none;border-radius:6px;background:var(--violet);color:#fff;cursor:pointer">Save</button>';
  }
  function _startDel(mid) { // show delete confirmation in drawer
    var el = document.getElementById('od-mem-act'); if (!el) return;
    el.innerHTML = '<p style="font-size:13px;color:var(--fg-2);margin-bottom:10px">Permanently delete this memory? This cannot be undone.</p><button data-od-act="confirm-del" data-mid="' + _esc(mid) + '" style="padding:6px 14px;border-radius:6px;background:var(--danger);color:#fff;border:none;cursor:pointer;margin-right:8px">Confirm delete</button><button data-od-act="cancel-act" style="padding:6px 14px;border-radius:6px;background:var(--card-2);border:1px solid var(--border);color:var(--fg);cursor:pointer">Cancel</button>';
  }

  // ── Event delegation ─────────────────────────────────────────────────────────
  function _wire(container, id) {
    container.addEventListener('click', function (e) {
      var el  = e.target.closest('[data-od-act]');
      if (!el) return;
      var act = el.dataset.odAct;

      // Example query chip in the Recall Lab starter state. recall-lab.js owns the
      // search itself and listens on #recall-lab-search, so fill the box and let it
      // run — duplicating its fetch here would be a second, divergent code path.
      if (act === 'recall-eg') {
        var qBox = document.getElementById('recall-lab-query');
        var qBtn = document.getElementById('recall-lab-search');
        if (qBox) qBox.value = el.dataset.q || '';
        if (qBtn) qBtn.click();
        return;
      }
      if (act === 'tab') {
        _switchTab(id, el.dataset.tab);
        return;
      }
      if (act === 'cat') {
        // Toggle on/off chip states without a full re-render
        var bar = document.getElementById(id + '-cats');
        if (bar) bar.querySelectorAll('[data-od-act="cat"]').forEach(function (c) {
          c.classList.toggle('on', c.dataset.cat === el.dataset.cat);
        });
        clearTimeout(_searchTimer);
        _st = Object.assign({}, _st, {
          category: el.dataset.cat || null, page: 0, searchQ: null,
          requestSeq: _st.requestSeq + 1,
        });
        var searchInput = document.querySelector('#' + id + ' [data-od-act="search"]');
        if (searchInput) searchInput.value = '';
        _loadMem(id);
        return;
      }
      if (act === 'sort') {
        // Update active state on sort buttons in-place
        var root = document.getElementById(id);
        if (root) root.querySelectorAll('[data-od-act="sort"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.sort === el.dataset.sort);
        });
        _doSort(id, el.dataset.sort);
        return;
      }
      if (act === 'scope-view') {
        var sroot = document.getElementById(id);
        if (sroot) sroot.querySelectorAll('[data-od-act="scope-view"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.scope === el.dataset.scope);
        });
        clearTimeout(_searchTimer);
        _st = Object.assign({}, _st, {
          scopeView: el.dataset.scope, page: 0, searchQ: null,
          requestSeq: _st.requestSeq + 1,
        });
        var scopedSearchInput = document.querySelector('#' + id + ' [data-od-act="search"]');
        if (scopedSearchInput) scopedSearchInput.value = '';
        _loadMem(id);
        return;
      }
      if (act === 'drawer') {
        var row = el.closest('tr[data-od-act]');
        var idx = parseInt((row || el).dataset.idx, 10);
        if (!isNaN(idx)) _openDrawer(idx);
        return;
      }
      if (act === 'close-drawer') { _closeDrawer(); return; }
      if (act === 'pg') {
        var pg = parseInt(el.dataset.page, 10);
        if (!isNaN(pg)) {
          clearTimeout(_searchTimer);
          _st = Object.assign({}, _st, {
            page: pg, searchQ: null, requestSeq: _st.requestSeq + 1,
          });
          var pagedSearchInput = document.querySelector('#' + id + ' [data-od-act="search"]');
          if (pagedSearchInput) pagedSearchInput.value = '';
          _loadMem(id);
        }
        return;
      }
      if (act === 'expand-cluster') { _expandCluster(id, el.dataset.cid); return; }
      if (act === 'tl-range') {
        var root2 = document.getElementById(id);
        if (root2) root2.querySelectorAll('[data-od-act="tl-range"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.range === el.dataset.range);
        });
        _tlLoaded[id + '-' + _st.tlRange] = false; // invalidate cache
        _st = Object.assign({}, _st, { tlRange: el.dataset.range });
        _loadTimeline(id);
        return;
      }
    });

    // Search — input event
    container.addEventListener('change', function (e) {
      if (e.target.dataset.odAct === 'window') {
        _st = Object.assign({}, _st, { window: e.target.value || '' });
        var wq = (_st.searchQ || '').trim();
        if (wq) _doSearch(id, wq);   // re-run active search with the new window
      }
    });
    container.addEventListener('input', function (e) {
      if (e.target.dataset.odAct === 'search') _doSearch(id, e.target.value.trim());
    });

    // Drawer actions — document-level (drawer is outside container), guarded once
    if (!window._slmMemDrawerWired) {
      window._slmMemDrawerWired = true;
      document.addEventListener('click', function (e) {
        var el = e.target.closest('[data-od-act]');
        if (!el) return;
        var act = el.dataset.odAct;
        if (act === 'close-drawer') { _closeDrawer(); return; }
        if (act === 'edit-mem') { _startEdit(el.dataset.mid); return; }
        if (act === 'del-mem')  { _startDel(el.dataset.mid); return; }
        if (act === 'cancel-act') { var ac = document.getElementById('od-mem-act'); if (ac) ac.innerHTML = ''; return; }
        if (act === 'set-scope') {
          var smid = el.dataset.mid;
          var scopeVal = (document.getElementById('od-scope-sel') || {}).value || 'personal';
          var sharedVal = ((document.getElementById('od-scope-shared') || {}).value || '').trim();
          if (scopeVal === 'shared' && !sharedVal) {
            showToast('Enter at least one profile to share with'); return;
          }
          _getMutToken().then(function (tok) {
            if (!tok) { showToast('Auth unavailable — reload the page'); return; }
            fetch('/api/memories/' + encodeURIComponent(smid) + '/scope', {
              method: 'PATCH', credentials: 'same-origin',
              headers: { 'Content-Type': 'application/json', 'X-Install-Token': tok },
              body: JSON.stringify({ scope: scopeVal, shared_with: sharedVal }),
            })
              .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
              .then(function () { showToast('Scope updated → ' + scopeVal); _closeDrawer(); _loadMem(_st.rootId); })
              .catch(function (e) { showToast('Scope update failed: ' + e.message); });
          });
          return;
        }
        if (act === 'save-edit' || act === 'confirm-del') {
          var rid = _st.rootId, isDel = act === 'confirm-del', mid = el.dataset.mid;
          var content = isDel ? '' : ((document.getElementById('od-ea') || {}).value || '').trim();
          if (!isDel && !content) { showToast('Content cannot be empty'); return; }
          _getMutToken().then(function (tok) {
            if (!tok) { showToast('Auth unavailable — reload the page'); return; }
            var opts = isDel
              ? {method:'DELETE',credentials:'same-origin',headers:{'X-Install-Token':tok}}
              : {method:'PATCH',credentials:'same-origin',headers:{'Content-Type':'application/json','X-Install-Token':tok},body:JSON.stringify({content:content})};
            fetch('/api/memories/' + encodeURIComponent(mid), opts)
              .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
              .then(function () { showToast(isDel ? 'Memory deleted' : 'Memory updated'); _closeDrawer(); _loadMem(rid); })
              .catch(function (e) { showToast((isDel ? 'Delete' : 'Edit') + ' failed: ' + e.message); });
          });
          return;
        }
      });
    }
  }

  // ── Public API + auto-init ───────────────────────────────────────────────────

  window.odRenderMemories = render;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      var p = document.getElementById('memories-pane');
      if (p) render(p);
    });
  } else {
    var _p = document.getElementById('memories-pane');
    if (_p) render(_p);
  }

}());
