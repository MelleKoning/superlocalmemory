/**
 * tests/ui/test_memories_kind_label_render.mjs
 *
 * The Memories -> All memories table's "Kind" column (formerly "Category")
 * must show the memory_kind_label the server computes (storage/memory_kinds
 * .py kind_fields(), the same confirmed/suggested/legacy-fallback precedence
 * `slm list` and recall use) instead of the raw legacy category string, and
 * the filter chip bar must enumerate the nine memory kinds (with real
 * counts from GET /api/memories/kind-counts) rather than five legacy
 * fact_type buckets.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function response(body) {
    return { ok: true, status: 200, json: function () { return Promise.resolve(body); } };
}

describe('od-memories.js — Kind column and chip bar', function () {
    it('shows the memory_kind_label, not the raw legacy category', async function () {
        const h = buildHarness(['root'], { ok: true, status: 200, json: {} });
        const pending = [];
        h.window.fetch = function (url) {
            const p = Promise.resolve(
                url.includes('kind-counts')
                    ? response({ counts: { decision: 1 }, truncated: false })
                    : response({
                        memories: [{
                            id: 'fact-confirmed',
                            content: 'We ship Tuesdays.',
                            category: 'episodic',          // legacy fact_type
                            memory_kind: 'decision',        // DISPLAYED kind
                            memory_kind_label: 'Decision',
                            memory_kind_state: 'confirmed',
                            importance: 0.8,
                            project_name: '',
                            created_at: '2026-01-01T00:00:00Z',
                        }],
                        total: 1, limit: 50, offset: 0, has_more: false,
                    }),
            );
            pending.push(url);
            return p;
        };
        evalModule(h.window, 'od-memories.js');
        h.window.odRenderMemories(h.document.getElementById('root'));
        await flushPromises();

        const text = h.document.getElementById('root').textContent;
        assert.match(text, /Decision/, 'the Kind column must show the memory_kind_label');
        assert.doesNotMatch(
            // "episodic" as a bare word — not as a substring of something else.
            text, /\bepisodic\b/,
            'the Kind column must not show the raw legacy category once a ' +
            'memory_kind_label is available',
        );
    });

    it('renders filter chips from the nine kinds, labelled, with server-provided counts', async function () {
        const h = buildHarness(['root'], { ok: true, status: 200, json: {} });
        h.window.fetch = function (url) {
            if (url.includes('kind-counts')) {
                return Promise.resolve(response({
                    counts: { semantic: 3, decision: 1 }, truncated: false,
                }));
            }
            return Promise.resolve(response({ memories: [], total: 0, limit: 50, offset: 0 }));
        };
        evalModule(h.window, 'od-memories.js');
        h.window.odRenderMemories(h.document.getElementById('root'));
        await flushPromises();

        const chips = Array.from(
            h.document.querySelectorAll('[data-od-act="cat"]'),
        ).map(function (el) { return el.textContent.trim(); });

        assert.ok(chips.some(function (c) { return /^All kinds/.test(c); }), chips.join(' | '));
        assert.ok(chips.some(function (c) { return c.indexOf('Fact') === 0; }), chips.join(' | '));
        assert.ok(chips.some(function (c) { return c.indexOf('Decision') === 0; }), chips.join(' | '));
        // A kind with a zero count must not clutter the bar with an empty chip.
        assert.ok(!chips.some(function (c) { return c.indexOf('Event') === 0; }), chips.join(' | '));
    });
});
