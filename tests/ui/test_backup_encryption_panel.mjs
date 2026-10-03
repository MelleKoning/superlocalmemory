import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

const root = new URL('../..', import.meta.url);

function fakeDocument() {
  function element(tag) {
    const el = {
      tag, children: [], style: {}, hidden: false, textContent: '', listeners: {},
      appendChild(child) { this.children.push(child); return child; },
      replaceChildren(...kids) { this.children = kids; },
      addEventListener(name, fn) { this.listeners[name] = fn; },
    };
    return el;
  }
  return { createElement: element };
}

async function loadPanel(status, reveal) {
  const source = await readFile(
    new URL('src/superlocalmemory/ui/js/od-backup-encryption.js', root), 'utf8');
  const calls = [];
  const context = {
    window: {}, document: fakeDocument(), Object, Promise,
    fetch: async (url) => { calls.push(['GET', url]); return { ok: true, json: async () => status }; },
  };
  vm.runInNewContext(source, context);
  const authMutation = async (url, method) => {
    calls.push([method, url]);
    return { json: async () => reveal };
  };
  const toasts = [];
  const box = context.window.odRenderBackupEncryption(authMutation, (m) => toasts.push(m));
  await box.refresh();
  return { box, calls, toasts };
}

function allText(el) {
  return [el.textContent, ...el.children.map(allText)].join(' ');
}

function findButton(el, label) {
  if (el.tag === 'button' && el.textContent === label) return el;
  for (const child of el.children) {
    const hit = findButton(child, label);
    if (hit) return hit;
  }
  return null;
}

describe('backup encryption panel', function () {
  it('shows the upgrade notices and reveals the key only on request', async function () {
    const key = 'SLMBK1-AAAA-BBBB';
    const { box, calls } = await loadPanel(
      { enabled: true, key_id: 'abcd', notices: ['Cloud backups are now encrypted.', 'Old backups.'],
        legacy_plaintext_uploads: true },
      { recovery_key: key, advice: 'Keep it safe.' },
    );
    assert.equal(box.hidden, false);
    assert.match(allText(box), /Cloud backups are now encrypted\./);
    assert.match(allText(box), /key abcd/);
    assert.doesNotMatch(allText(box), /SLMBK1/);
    assert.ok(!calls.some(([, url]) => url === '/api/backup/recovery-key'));

    await findButton(box, 'Show recovery key').listeners.click();
    assert.ok(calls.some(([m, url]) => m === 'POST' && url === '/api/backup/recovery-key'));
    assert.match(allText(box), /SLMBK1-AAAA-BBBB/);
  });

  it('stays hidden when there is no key and nothing to say', async function () {
    const { box } = await loadPanel({ enabled: false, notices: [] }, {});
    assert.equal(box.hidden, true);
  });

  it('dismisses the old-backups notice through an authenticated POST', async function () {
    const { box, calls } = await loadPanel(
      { enabled: true, key_id: 'abcd', notices: ['Old backups.'], legacy_plaintext_uploads: true }, {});
    await findButton(box, 'Dismiss old-backups notice').listeners.click();
    assert.ok(calls.some(([m, url]) => m === 'POST' && url === '/api/backup/encryption/acknowledge'));
  });
});
