/**
 * tests/ui/test_od_ops_health_erasure.mjs — Ops Health table explains erasures.
 * Runner: npm test   (scripts/run-ui-tests.mjs)
 * Requires: jsdom
 *
 * `slm ops list` prints each exhausted entry's kind, its recorded reason and
 * what happened. The dashboard table showed none of that: an unconfirmed
 * deletion appeared as a bare "Exhausted" row with an empty Error cell, so
 * the user could not tell a deletion from a failed save, or what to do.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

const REASON = 'deletion not confirmed: its erasure receipt does not list which memories it covered';
const EXPLANATION = 'A deletion could not be confirmed: the memory or one of its search '
    + 'indexes still holds it. Delete the memory again, then use Reconcile; or Cancel to dismiss.';

const PAYLOAD = {
    dead_letter: [],
    degraded_manifests: [],
    exhausted_obligations: [
        {
            category: 'exhausted_obligation', operation_id: 'erase-1', kind: 'erase',
            attempts: 1, profile_id: 'default', error: REASON, what_happened: EXPLANATION,
        },
        {
            category: 'exhausted_obligation', operation_id: 'save-1', kind: 'apply',
            attempts: 10, profile_id: 'default',
            what_happened: 'Background sync failed after maximum retries — use Force Re-sync or Cancel.',
        },
    ],
    total: 2,
};

async function renderedTable() {
    const h = buildHarness(['od-ops-tabs'], { ok: true, status: 200, json: PAYLOAD });
    evalModule(h.window, 'od-ops-health.js');
    h.window.OpsHealth.load();
    await flushPromises();
    return h.document.getElementById('oh-table-wrap');
}

function rowFor(table, opId) {
    return Array.from(table.querySelectorAll('tbody tr'))
        .find(tr => tr.textContent.includes(opId));
}

describe('Ops Health table — erasure entries', function () {
    it('has a Kind column naming each entry the way the CLI does', async function () {
        const table = await renderedTable();
        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent);
        const kindCol = headers.indexOf('Kind');
        assert.ok(kindCol >= 0, 'headers: ' + headers.join(', '));
        const kindOf = opId => rowFor(table, opId).querySelectorAll('td')[kindCol].textContent;
        assert.equal(kindOf('erase-1'), 'erase');
        assert.equal(kindOf('save-1'), 'apply');
    });

    it('shows the full recorded reason, not a 60-character stub', async function () {
        const row = rowFor(await renderedTable(), 'erase-1');
        assert.ok(row.textContent.includes(REASON), row.textContent);
    });

    it('shows what happened and what to do', async function () {
        const row = rowFor(await renderedTable(), 'erase-1');
        assert.ok(row.textContent.includes(EXPLANATION), row.textContent);
    });
});
