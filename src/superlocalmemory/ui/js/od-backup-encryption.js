/* od-backup-encryption.js — cloud backup encryption panel for the Backup page.
 * Exposes:
 *   window.odRenderBackupEncryption(authMutation, toast) -> HTMLElement
 * Wired endpoints:
 *   GET  /api/backup/encryption               key-free status + notices
 *   POST /api/backup/recovery-key             show the recovery key (install token)
 *   POST /api/backup/encryption/acknowledge   dismiss the old-backups notice
 * The recovery key is only ever written with textContent and never stored.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  function node(tag, text, css) {
    var e = document.createElement(tag);
    if (text) e.textContent = text;
    if (css) Object.keys(css).forEach(function (k) { e.style[k] = css[k]; });
    return e;
  }

  function button(label, primary) {
    var b = node('button', label);
    b.type = 'button';
    b.className = primary ? 'btn sm primary' : 'btn sm';
    return b;
  }

  function odRenderBackupEncryption(authMutation, toast) {
    var box = node('div', null, { margin: '0 0 18px', padding: '16px 20px' });
    box.className = 'card';
    box.hidden = true;

    var summary = node('p', null, { margin: '0 0 8px', fontWeight: '600' });
    var notes = node('div');
    var keyOut = node('p', null, {
      fontFamily: 'var(--mono, monospace)', wordBreak: 'break-all', margin: '10px 0 4px',
    });
    var advice = node('p', null, { fontSize: '12.5px', color: 'var(--fg-3)', margin: '0' });
    var actions = node('div', null, { display: 'flex', gap: '8px', marginTop: '10px' });
    var showBtn = button('Show recovery key', true);
    var dismissBtn = button('Dismiss old-backups notice', false);
    actions.appendChild(showBtn);
    actions.appendChild(dismissBtn);
    [summary, notes, keyOut, advice, actions].forEach(function (e) { box.appendChild(e); });

    function refresh() {
      return fetch('/api/backup/encryption', { credentials: 'same-origin' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (s) {
          if (!s) return;
          var notices = s.notices || [];
          box.hidden = !(s.enabled || notices.length);
          summary.textContent = s.enabled
            ? 'Cloud copies are encrypted on this computer before upload (key ' + s.key_id + ').'
            : 'Cloud copies will be encrypted on this computer before upload.';
          notes.replaceChildren();
          notices.forEach(function (text) {
            notes.appendChild(node('p', text, { margin: '6px 0', fontSize: '13px' }));
          });
          showBtn.hidden = !s.enabled;
          dismissBtn.hidden = !s.legacy_plaintext_uploads;
        })
        .catch(function () { /* status is advisory; the backup page still works */ });
    }

    showBtn.addEventListener('click', function () {
      return authMutation('/api/backup/recovery-key', 'POST')
        .then(function (r) { return r.json(); })
        .then(function (d) {
          keyOut.textContent = d.recovery_key;
          advice.textContent = d.advice || '';
          return refresh();
        })
        .catch(function () { toast('Could not show the recovery key', true); });
    });

    dismissBtn.addEventListener('click', function () {
      return authMutation('/api/backup/encryption/acknowledge', 'POST')
        .then(refresh)
        .catch(function () { toast('Could not dismiss the notice', true); });
    });

    box.refresh = refresh;
    refresh();
    return box;
  }

  window.odRenderBackupEncryption = odRenderBackupEncryption;
}());
