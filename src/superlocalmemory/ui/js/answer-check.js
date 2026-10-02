/* answer-check.js — "Answer check" settings section (4.1.18)
 * Lets a non-technical user turn on: the on-device Laya judge, the hosted
 * Jev judge, or neither — and, inside the Jev option only, an opt-in
 * "Also use Jev to reorder results" with its own notice.
 *
 * This file owns the FEATURE (API calls, validation, confirms); it does not
 * own the surrounding chrome. od-settings.js builds the group card exactly
 * like every other settings section (makeGrp/makeRow, same typography and
 * colours) and calls window.SLMAnswerCheck.renderInto(mount) to hand this
 * file one empty container to fill — so the section lives and updates
 * inside the real Settings hub, not as a second design next to it.
 *
 * Styling matches the hub's own design tokens (css/design-system.css:
 * --card-2, --border, --fg, --r-md, the .btn/.switch classes) rather than
 * Bootstrap, because that is what the surrounding hub actually renders.
 *
 * Auth: plain fetch() is enough — core.js's global fetch patch attaches
 * X-Install-Token to every same-origin mutating request, and the server
 * refuses any answer-check change without it (even from this Mac). MANAGE
 * permission is checked client-side (via /api/rbac/whoami) only to decide
 * whether to show the controls as usable; the server is the real gate and
 * answers 403 regardless of what this script does.
 *
 * "auto" is the starting state nobody chose: it is shown as what it does
 * (status.active), never offered as a fourth option. Setting up the model
 * and checking an existing install are background jobs; this file polls the
 * status until they finish, and stops polling if the daemon stops answering.
 *
 * API: GET/POST /api/v3/answer-check[...] — see
 * src/superlocalmemory/server/routes/answer_check.py for the exact contract.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0-or-later
 */
(function () {
  'use strict';

  var API = '/api/v3/answer-check';
  var canManage = true; // optimistic until /whoami says otherwise
  var pollTimer = null;
  var pollFailures = 0;
  var POLL_MS = 2000;
  var POLL_MAX_FAILURES = 5;   // ~10 s of silence, then stop and say so
  var adoptPending = false;    // a "Use this install" check is being waited on
  var lastStatus = null;       // the last status read, to re-render on a failed read
  var mountEl = null;

  /* ── Small helpers (duplicated on purpose — this file does not reach
     into od-settings.js's private closure; these mirror its factories so
     the two sections look identical) ─────────────────────────────────── */

  function el(tag, attrs, css) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]);
    });
    if (css) Object.assign(e.style, css);
    return e;
  }
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function acSel(opts) {
    var s = el('select');
    Object.assign(s.style, { height:'34px', borderRadius:'var(--r-md)',
      border:'1px solid var(--border)', background:'var(--card-2)',
      color:'var(--fg)', padding:'0 10px', fontSize:'13px',
      minWidth:'140px', outline:'none' });
    opts.forEach(function (o) {
      var opt = document.createElement('option');
      opt.value = o.v; opt.textContent = o.l;
      if (o.selected) opt.selected = true;
      s.appendChild(opt);
    });
    return s;
  }
  function acTin(ph, type) {
    var i = el('input', { type: type || 'text', placeholder: ph || '' });
    Object.assign(i.style, { height:'34px', borderRadius:'var(--r-md)',
      border:'1px solid var(--border)', background:'var(--card-2)',
      color:'var(--fg)', padding:'0 10px', fontSize:'13px',
      minWidth:'160px', maxWidth:'260px', outline:'none' });
    return i;
  }
  function acBtn(txt, pri) {
    var b = el('button', { type:'button' });
    b.className = 'btn sm ' + (pri ? 'primary' : 'ghost');
    b.textContent = txt;
    return b;
  }
  function acRow() {
    var d = el('div', null, { display:'flex', gap:'8px', alignItems:'center', flexWrap:'wrap' });
    Array.from(arguments).forEach(function (c) { d.appendChild(c); });
    return d;
  }
  function acNote(text) {
    return el('span', { text: text }, { fontSize:'12px', color:'var(--fg-2)' });
  }

  function notify(message, isError) {
    if (typeof window.showToast === 'function') { window.showToast(message); return; }
    if (isError) console.error(message); else console.log(message);
  }

  /* null when the read failed, so a failure is never mistaken for a status. */
  function getJSON(url) {
    return fetch(url, { credentials: 'same-origin' }).then(function (r) {
      if (!r.ok) return null;
      return r.json().catch(function () { return null; });
    }).catch(function () { return null; });
  }

  function mutate(url, method, body) {
    return fetch(url, {
      method: method,
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) {
        return { ok: response.ok, status: response.status, data: data };
      });
    }).catch(function () {
      return { ok: false, status: 0,
               data: { error: 'Could not reach SLM. Try again in a moment.' } };
    });
  }

  function errorMessage(result) {
    if (result.status === 403) return 'Ask an admin to change this.';
    return (result.data && (result.data.error || result.data.detail)) ||
      'That did not work. Try again in a moment.';
  }

  /* ── Render ───────────────────────────────────────────────── */

  /* What the radios show: the explicit choice, or — for the never-chosen
     "auto" — what it is actually doing right now. */
  function effectiveMode(status) {
    if (status.mode === 'laya' || status.mode === 'jev' || status.mode === 'off') return status.mode;
    return status.active === 'laya' ? 'laya' : 'off';
  }

  function render(status) {
    lastStatus = status;
    mountEl.innerHTML = '';
    mountEl.appendChild(el('p', { text:
      'Lets your memory say "I don’t have that" instead of guessing.' },
      { fontSize:'12.5px', color:'var(--fg-2)', margin:'0 0 14px' }));

    var bannerHost = el('div');
    mountEl.appendChild(bannerHost);
    renderBanner(bannerHost, status);

    if (!canManage) {
      mountEl.appendChild(el('div', { text: 'Ask an admin to change this.' },
        { fontSize:'12px', color:'var(--fg-2)', marginBottom:'10px' }));
    }

    mountEl.appendChild(buildModes(status));
    mountEl.appendChild(buildLaya(status));
    mountEl.appendChild(buildJev(status));
  }

  function renderBanner(host, status) {
    var layaUnset = status.mode !== 'laya' && status.laya.state === 'not_installed';
    if (!(status.apple_silicon && layaUnset)) { host.innerHTML = ''; return; }
    var banner = el('div', null, {
      display:'flex', alignItems:'center', justifyContent:'space-between', gap:'12px',
      background:'var(--violet-soft)', border:'1px solid var(--violet-line)',
      borderRadius:'var(--r-md)', padding:'10px 14px', marginBottom:'14px',
      fontSize:'13px',
    });
    banner.appendChild(el('span', { text:
      'New: answer check. Set it up — about 1.1 GB, stays on this Mac.' }));
    var closeBtn = el('button', { type:'button', text:'×', 'aria-label':'Dismiss' },
      { background:'transparent', border:'0', fontSize:'16px', lineHeight:'1',
        color:'var(--fg-2)', cursor:'pointer' });
    closeBtn.addEventListener('click', function () { host.innerHTML = ''; });
    banner.appendChild(closeBtn);
    host.innerHTML = '';
    host.appendChild(banner);
  }

  function confirmSwitch(targetLabel, otherLabel) {
    if (otherLabel === targetLabel) return true;
    return window.confirm('Turning this on turns ' + otherLabel + ' off.');
  }

  var MODE_NAMES = { laya: 'the on-device model', jev: 'Jev', off: 'answer check' };

  function buildModes(status) {
    var wrap = el('div', null, { marginBottom:'16px' });
    var current = effectiveMode(status);
    var layaDisabled = !status.apple_silicon;

    [
      { v:'laya', l:'On this Mac — private (recommended)',
        disabled: layaDisabled, reason: 'Needs a Mac with Apple Silicon' },
      { v:'jev', l:'Online with Jev — needs a key', disabled:false, reason:'' },
      { v:'off', l:'Off', disabled:false, reason:'' },
    ].forEach(function (opt) {
      var row = el('label', null, {
        display:'flex', alignItems:'center', gap:'8px', marginBottom:'8px',
        fontSize:'13.5px', cursor: opt.disabled ? 'not-allowed' : 'pointer',
        color: opt.disabled ? 'var(--fg-3)' : 'var(--fg)',
      });
      var input = el('input', { type:'radio', name:'ac-mode', value: opt.v });
      input.checked = current === opt.v;
      input.disabled = opt.disabled || !canManage;
      row.appendChild(input);
      row.appendChild(document.createTextNode(opt.l));
      if (opt.reason && opt.disabled) {
        row.appendChild(el('span', { text: ' (' + opt.reason + ')' },
          { fontSize:'11.5px', color:'var(--danger)', marginLeft:'4px' }));
      }
      input.addEventListener('change', function () {
        var next = this.value;
        if (current !== 'off' && next !== current &&
            !confirmSwitch(MODE_NAMES[next], MODE_NAMES[current])) {
          this.checked = false;
          var running = wrap.querySelector('input[value="' + current + '"]');
          if (running) running.checked = true;
          return;
        }
        var radios = wrap.querySelectorAll('input[name="ac-mode"]');
        Array.prototype.forEach.call(radios, function (r) { r.disabled = true; });
        mutate(API + '/mode', 'POST', { mode: next }).then(function (result) {
          if (!result.ok) notify(errorMessage(result), true);
          else notify(result.data.message || 'Answer check updated.');
          refresh(); // always re-read truth; a failed read re-renders the last one
        });
      });
      wrap.appendChild(row);
    });
    if (status.mode === 'auto') {
      wrap.appendChild(acNote(current === 'laya'
        ? 'Turned on by itself: the on-device model is set up and checked.'
        : 'Not chosen yet. It turns on by itself once the on-device model is set up.'));
    }
    return wrap;
  }

  /* ── Laya panel ───────────────────────────────────────────── */

  function layaStatusText(laya) {
    switch (laya.state) {
      case 'ready': return 'Ready.';
      case 'installing': return (laya.step || 'Setting up') + '…';
      case 'failed': return 'Setup failed: ' + (laya.error || 'unknown reason');
      case 'unsupported': return 'Needs a Mac with Apple Silicon.';
      default: return 'Not set up yet.';
    }
  }

  function buildProgressBar(progress) {
    var track = el('div', null, { height:'8px', borderRadius:'var(--r-pill)',
      background:'var(--card-2)', border:'1px solid var(--border)',
      overflow:'hidden', margin:'6px 0 10px', maxWidth:'280px' });
    var fill = el('div', null, { height:'100%',
      width: Math.round((progress || 0) * 100) + '%',
      background:'var(--violet)', transition:'width .3s ease' });
    track.setAttribute('aria-live', 'polite');
    track.appendChild(fill);
    return track;
  }

  function buildLaya(status) {
    var laya = status.laya;
    var box = el('div', null, { border:'1px solid var(--border)',
      borderRadius:'var(--r-md)', padding:'12px 14px', marginBottom:'14px' });
    box.appendChild(el('h4', { text:'On this Mac' },
      { fontSize:'13px', fontWeight:'640', margin:'0 0 6px' }));
    box.appendChild(el('p', { text: layaStatusText(laya) },
      { fontSize:'12.5px', color:'var(--fg-2)', margin:'0 0 8px' }));
    if (laya.state === 'installing') box.appendChild(buildProgressBar(laya.progress));

    var setupBtn = acBtn('Set up', true);
    setupBtn.disabled = laya.state === 'installing' || !status.apple_silicon || !canManage;
    var removeBtn = acBtn('Remove', false);
    removeBtn.disabled = laya.state === 'not_installed' || !canManage;
    box.appendChild(acRow(setupBtn, removeBtn));

    var details = el('details', null, { marginTop:'10px' });
    var summary = el('summary', { text: 'Advanced — use an existing install' },
      { fontSize:'12px', color:'var(--fg-2)', cursor:'pointer' });
    details.appendChild(summary);
    var pyInp = acTin('Python interpreter path');
    var hfInp = acTin('HF_HOME (optional)');
    var modelInp = acTin('Model path (optional)');
    var adoptBtn = acBtn('Use this install', false);
    adoptBtn.disabled = !canManage;
    var advRow = el('div', null, { display:'flex', gap:'8px', flexWrap:'wrap', marginTop:'8px' });
    [pyInp, hfInp, modelInp, adoptBtn].forEach(function (c) { advRow.appendChild(c); });
    details.appendChild(advRow);
    box.appendChild(details);

    setupBtn.addEventListener('click', function () {
      mutate(API + '/laya/setup', 'POST').then(function (result) {
        if (!result.ok) { notify(errorMessage(result), true); refresh(); return; }
        schedulePoll();
        refresh();
      });
    });
    removeBtn.addEventListener('click', function () {
      if (!window.confirm('Remove the on-device model? You can set it up again later.')) return;
      mutate(API + '/laya/remove', 'POST').then(function (result) {
        if (!result.ok) notify(errorMessage(result), true);
        refresh();
      });
    });
    adoptBtn.addEventListener('click', function () {
      var python = pyInp.value.trim();
      if (!python) { notify('A Python interpreter path is required.', true); return; }
      adoptBtn.disabled = true;
      mutate(API + '/laya/adopt', 'POST', {
        python: python, hf_home: hfInp.value.trim(), model_path: modelInp.value.trim(),
      }).then(function (result) {
        if (!result.ok) { notify(errorMessage(result), true); refresh(); return; }
        // The check loads the model, so it runs in the background; the poll
        // reports how it went.
        adoptPending = true;
        notify('Checking that install…');
        schedulePoll();
        refresh();
      });
    });

    return box;
  }

  /* ── Jev panel ────────────────────────────────────────────── */

  var CONSENT_TEXT = 'When on, each recall sends your question and its top ' +
    '3 memories to {provider}. Don’t use this for client, confidential ' +
    'or personal material.';

  /* Exactly the approved wording; {k} and {provider} come from the server's
     status, so the notice names the count and the provider actually used. */
  var RERANK_CONSENT_TEXT =
    "When on, each recall sends your question and its top {k} memories to {provider} to choose the best order. Don't use this for client, confidential or personal material.";

  function providerLabel(p) { return p === 'openrouter' ? 'OpenRouter' : 'TypeSafe'; }

  function buildJev(status) {
    var jev = status.jev || {};
    var box = el('div', null, { border:'1px solid var(--border)',
      borderRadius:'var(--r-md)', padding:'12px 14px' });
    box.appendChild(el('h4', { text:'Online with Jev' },
      { fontSize:'13px', fontWeight:'640', margin:'0 0 8px' }));

    var providerSel = acSel([
      { v:'typesafe', l:'TypeSafe', selected: jev.provider === 'typesafe' },
      { v:'openrouter', l:'OpenRouter', selected: jev.provider === 'openrouter' },
    ]);
    var keyInput = acTin(jev.has_key ? jev.key_hint : 'Paste a key', 'password');
    var saveBtn = acBtn('Save key', true);
    var testBtn = acBtn('Test', false);
    testBtn.disabled = !jev.has_key || !canManage;
    var removeBtn = acBtn('Remove key', false);
    // A key file changed outside SLM must stay removable: that is the repair.
    removeBtn.disabled = !(jev.has_key || jev.key_problem) || !canManage;
    var testResult = acNote('');

    [providerSel, keyInput, saveBtn].forEach(function (c) { c.disabled = !canManage; });

    box.appendChild(acRow(providerSel, keyInput, saveBtn));
    if (jev.key_problem) {
      box.appendChild(el('div', { text: jev.key_problem },
        { fontSize:'12px', color:'var(--danger)', marginTop:'6px' }));
    }
    box.appendChild(el('div', null, { height:'8px' }));
    box.appendChild(acRow(testBtn, removeBtn, testResult));
    box.appendChild(el('div', { text:
      'Each test sends one request to your provider and uses a little of your credit.' },
      { fontSize:'11.5px', color:'var(--fg-3)', marginTop:'4px' }));

    var consentWrap = el('label', null, { display:'flex', gap:'8px',
      alignItems:'flex-start', marginTop:'10px', fontSize:'12px',
      color:'var(--fg-2)', cursor: canManage ? 'pointer' : 'not-allowed' });
    var consentBox = el('input', { type:'checkbox' }, { marginTop:'2px' });
    consentBox.checked = !!jev.consent;
    consentBox.disabled = !canManage;
    var consentLabel = el('span', { text:
      CONSENT_TEXT.replace('{provider}', providerLabel(jev.provider)) });
    consentWrap.appendChild(consentBox);
    consentWrap.appendChild(consentLabel);
    box.appendChild(consentWrap);

    providerSel.addEventListener('change', function () {
      consentLabel.textContent = CONSENT_TEXT.replace('{provider}', providerLabel(this.value));
    });

    saveBtn.addEventListener('click', function () {
      var key = keyInput.value.trim();
      if (!key) { notify('Paste a key first.', true); return; }
      mutate(API + '/jev/key', 'POST', {
        provider: providerSel.value, key: key,
      }).then(function (result) {
        keyInput.value = '';
        if (!result.ok) { notify(errorMessage(result), true); return; }
        notify('Key saved.');
        refresh();
      });
    });
    testBtn.addEventListener('click', function () {
      testBtn.disabled = true;
      testResult.textContent = 'Testing…';
      mutate(API + '/jev/test', 'POST', { provider: providerSel.value }).then(function (result) {
        testBtn.disabled = !canManage;
        testResult.textContent = result.ok
          ? (result.data.message || (result.data.ok ? 'Reached.' : 'Could not reach it.'))
          : errorMessage(result);
      });
    });
    removeBtn.addEventListener('click', function () {
      if (!window.confirm('Remove the saved key for ' + providerLabel(providerSel.value) + '?')) return;
      mutate(API + '/jev/key', 'DELETE', { provider: providerSel.value }).then(function (result) {
        if (!result.ok) notify(errorMessage(result), true);
        refresh();
      });
    });
    consentBox.addEventListener('change', function () {
      var accepted = this.checked;
      this.disabled = true;
      mutate(API + '/jev/consent', 'POST', {
        accepted: accepted, provider: providerSel.value,
      }).then(function (result) {
        if (!result.ok) notify(errorMessage(result), true);
        refresh();
      });
    });

    box.appendChild(buildRerank(status));
    return box;
  }

  /* ── "Also use Jev to reorder results" — its own toggle and notice ── */

  function rerankText(rerank, provider) {
    return RERANK_CONSENT_TEXT
      .replace('{k}', String(rerank.k))
      .replace('{provider}', providerLabel(provider));
  }

  function buildRerank(status) {
    var jev = status.jev || {};
    var rerank = jev.rerank || { enabled: false, active: false, k: 20 };
    // Turning it on needs the online check running; turning it off never
    // needs anything, so a stale "on" can always be switched off.
    var jevRunning = status.mode === 'jev' && status.active === 'jev' &&
      !!jev.consent && !!jev.has_key;
    var usable = canManage && (jevRunning || rerank.enabled);

    var wrap = el('div', null, { marginTop:'12px', paddingTop:'10px',
      borderTop:'1px solid var(--border)' });
    var row = el('label', null, { display:'flex', gap:'8px', alignItems:'flex-start',
      fontSize:'12.5px', cursor: usable ? 'pointer' : 'not-allowed' });
    var toggle = el('input', { type:'checkbox' }, { marginTop:'2px' });
    toggle.checked = !!rerank.enabled;
    toggle.disabled = !usable;

    var words = el('span');
    words.appendChild(el('span', { text: 'Also use Jev to reorder results' },
      { display:'block', color:'var(--fg)' }));
    words.appendChild(el('span', { text: rerankText(rerank, jev.provider) },
      { display:'block', color:'var(--fg-2)', fontSize:'12px', marginTop:'2px' }));
    if (!jevRunning && !rerank.enabled) {
      words.appendChild(el('span', { text: 'Turn on "Online with Jev" first.' },
        { display:'block', color:'var(--fg-3)', fontSize:'11.5px', marginTop:'2px' }));
    }
    row.appendChild(toggle);
    row.appendChild(words);
    wrap.appendChild(row);

    toggle.addEventListener('change', function () {
      var enabled = this.checked;
      this.disabled = true;
      // Ticking the box next to the notice is accepting it, as for Jev itself.
      mutate(API + '/jev/rerank', 'POST', {
        enabled: enabled, accepted: enabled, provider: jev.provider,
      }).then(function (result) {
        if (!result.ok) notify(errorMessage(result), true);
        else notify(result.data.message || 'Saved.');
        refresh(); // always re-read truth, never trust the optimistic click
      });
    });
    return wrap;
  }

  /* ── Polling while a setup or a check is in progress ─────── */

  function busy(status) {
    return status.laya.state === 'installing' || !!(status.adopt && status.adopt.running);
  }

  function stopPoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function reportAdopt(status) {
    if (!adoptPending || !status.adopt || status.adopt.running) return;
    adoptPending = false;
    if (status.adopt.state === 'ready') notify('Using that install.');
    else notify(status.adopt.error || 'That install could not be used.', true);
  }

  function lostContact() {
    stopPoll();
    if (!mountEl) return;
    mountEl.appendChild(el('div', { text:
      'Lost contact with SLM. Reload the page to see the latest.' },
      { fontSize:'12px', color:'var(--danger)', marginTop:'10px' }));
  }

  function schedulePoll() {
    if (pollTimer) return;
    pollFailures = 0;
    pollTimer = setInterval(function () {
      getJSON(API).then(function (status) {
        if (!mountEl || !pollTimer) return;
        if (!status || !status.laya) {
          pollFailures += 1;
          if (pollFailures >= POLL_MAX_FAILURES) lostContact();
          return;
        }
        pollFailures = 0;
        render(status);
        reportAdopt(status);
        if (!busy(status)) stopPoll();
      });
    }, POLL_MS);
  }

  /* ── Bootstrap ────────────────────────────────────────────── */

  function refresh() {
    if (!mountEl) return;
    getJSON(API).then(function (status) {
      if (!status || !status.laya) {
        // GET failed: show the last good read again, so nothing stays disabled.
        if (lastStatus) render(lastStatus);
        return;
      }
      render(status);
      reportAdopt(status);
      if (busy(status)) schedulePoll();
    });
  }

  function applyPermission(whoami) {
    var permissions = (whoami && whoami.permissions) || [];
    canManage = permissions.indexOf('manage') !== -1;
  }

  /** Called by od-settings.js with the empty mount it built inside its own
   * group card. Re-entrant: od-settings.js re-mounts the whole hub only
   * when the Settings pane is first opened, so this only runs once per
   * pane build, but guards anyway in case that ever changes. */
  function renderInto(container) {
    if (!container) return;
    mountEl = container;
    getJSON('/api/rbac/whoami').then(applyPermission).catch(function () {}).then(refresh);
  }

  // effectiveMode is exposed for the UI tests (tests/test_ui).
  window.SLMAnswerCheck = { renderInto: renderInto, effectiveMode: effectiveMode };
}());
