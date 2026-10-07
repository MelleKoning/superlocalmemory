/**
 * tests/ui/test_quick_actions_changed_this_week_kind.mjs
 *
 * Q5 (2026-10-06): the "Changed This Week" quick-insight row must show the
 * memory_kind_label the server now computes (storage/memory_kinds.py
 * kind_fields(), the same precedence the Memories table's "Kind" column
 * uses — see test_memories_kind_label_render.mjs), not the raw legacy
 * fact_type string.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule } from './harness.mjs';

describe('quick-actions.js — Changed This Week badge', function () {
    it('shows the memory_kind_label, not the raw legacy fact_type', function () {
        const h = buildHarness(['insight-results'], { ok: true, status: 200, json: {} });
        evalModule(h.window, 'quick-actions.js');

        const container = h.document.getElementById('insight-results');
        h.window.renderChangedThisWeek({
            items: [{
                fact_id: 'f1',
                content: 'We ship Tuesdays.',
                fact_type: 'episodic',          // legacy
                memory_kind: 'decision',
                memory_kind_label: 'Decision',   // DISPLAYED kind
                memory_kind_state: 'confirmed',
                created_at: '2026-10-06T00:00:00Z',
            }],
        }, container);

        const text = container.textContent;
        assert.match(text, /Decision/, 'the badge must show the memory_kind_label');
        assert.doesNotMatch(
            text, /\bepisodic\b/,
            'the badge must not show the raw legacy fact_type once a ' +
            'memory_kind_label is available',
        );
    });

    it('falls back to the raw fact_type on a legacy store with no kind label', function () {
        const h = buildHarness(['insight-results'], { ok: true, status: 200, json: {} });
        evalModule(h.window, 'quick-actions.js');

        const container = h.document.getElementById('insight-results');
        h.window.renderChangedThisWeek({
            items: [{
                fact_id: 'f2',
                content: 'Old store fact.',
                fact_type: 'semantic',
                created_at: '2026-10-06T00:00:00Z',
            }],
        }, container);

        assert.match(container.textContent, /semantic/, 'the legacy fallback must still show something');
    });
});
