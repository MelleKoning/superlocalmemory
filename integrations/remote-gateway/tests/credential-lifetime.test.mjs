import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEVICE_CREDENTIAL_TTL_MS, RENEWAL_WINDOW_MS } from '../src/credential-lifetime.ts';

test('the laptop renews in the same window the gateway accepts', () => {
  const python = readFileSync(new URL('../../../src/superlocalmemory/remote_connections/renewal.py', import.meta.url), 'utf8');
  const laptop = Number(/^RENEWAL_WINDOW_MS = ([0-9* ]+)$/m.exec(python)?.[1].split('*').reduce((a, b) => a * Number(b.trim()), 1));
  assert.equal(RENEWAL_WINDOW_MS, DEVICE_CREDENTIAL_TTL_MS / 2);
  assert.equal(laptop, RENEWAL_WINDOW_MS);
});
