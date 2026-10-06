/**
 * tests/ui/test_contrast_token_sweep.mjs
 *
 * Pins the contrast fixes from the exhaustive dashboard sweep (both themes,
 * every pane, via real Chromium — not reproducible in jsdom, which cannot
 * resolve CSS custom properties in getComputedStyle, verified empirically
 * earlier in this work). What IS practical and durable here:
 *
 *   1. literal (non-var()) color values this round introduced, checked with
 *      the same WCAG relative-luminance formula used throughout this file's
 *      sibling test (test_recall_lab_badge_contrast.mjs);
 *   2. the two "a blanket low-specificity-looking rule actually outranks a
 *      component's own color" regressions this round found and fixed —
 *      .ng-dark .btn-primary (background) was NOT the bug here; the bug
 *      class is ".ng-dark a" (0,1,1) beating ".star-cta" (0,1,0) — same
     shape as the .msg.q regression from the previous round, caught again
 *      by the same :where()-zero-specificity technique.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const neuralGlass = readFileSync(join(UI, 'css/neural-glass.css'), 'utf8');
const designSystem = readFileSync(join(UI, 'css/design-system.css'), 'utf8');

function luminance([r, g, b]) {
    const lin = (c) => {
        const v = c / 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}
function contrast(a, b) {
    const la = luminance(a), lb = luminance(b);
    const [hi, lo] = la >= lb ? [la, lb] : [lb, la];
    return (hi + 0.05) / (lo + 0.05);
}
const WHITE = [255, 255, 255];

describe('Coordinator item 1 — light-theme Bootstrap .btn-primary', function () {
    it('replaces #667eea (3.66:1) with a literal darker hex that passes AA', function () {
        const m = designSystem.match(/\[data-bs-theme="light"\] \.btn-primary \{[^}]*\}/)
            || neuralGlass.match(/\[data-bs-theme="light"\] \.btn-primary \{[^}]*\}/);
        assert.ok(m, '[data-bs-theme="light"] .btn-primary rule not found');
        assert.doesNotMatch(m[0], /#667eea/, 'still uses the failing #667eea background');
        const hex = m[0].match(/background:\s*(#[0-9a-fA-F]{6})/);
        assert.ok(hex, 'no background hex found in the rule');
        const rgb = [1, 3, 5].map((i) => parseInt(hex[1].slice(i, i + 2), 16));
        assert.ok(contrast(WHITE, rgb) >= 4.5,
            `white on ${hex[1]} is ${contrast(WHITE, rgb).toFixed(2)}:1, still under AA`);
    });
});

describe('Coordinator item 2 — self-colored badge text tokens', function () {
    const CASES = [
        { name: '--violet-text (light)', re: /--violet-text:\s*hsl\(262 83% (\d+)%\)/, light: true },
        { name: '--violet-text (dark)', re: /--violet-text:\s*hsl\(262 83% (\d+)%\)/, light: false },
    ];

    it('.badge.violet points its color at --violet-text, not --violet', function () {
        assert.match(designSystem, /\.badge\.violet\s*\{\s*color:\s*var\(--violet-text\)/,
            '.badge.violet must read --violet-text (the dedicated text token), not --violet');
    });

    it('.ng-dark .badge.bg-primary points its color at --ng-accent-text, not --ng-accent', function () {
        assert.match(neuralGlass, /\.ng-dark \.badge\.bg-primary\s*\{[^}]*color:\s*var\(--ng-accent-text\)/,
            '.ng-dark .badge.bg-primary must read --ng-accent-text, not --ng-accent');
    });

    it('--violet-text and --ng-accent-text are both actually defined', function () {
        assert.match(designSystem, /--violet-text:\s*hsl\(/);
        assert.match(neuralGlass, /--ng-accent-text:\s*#[0-9a-fA-F]{6}/);
    });
});

describe('Regression: a blanket rule must not outrank a specific component by accident', function () {
    it('.ng-dark a excludes .star-cta via :where() (zero specificity), not a bare :not()', function () {
        const m = neuralGlass.match(/\.ng-dark a[^{,\n]*\{/);
        assert.ok(m, '.ng-dark a rule not found');
        assert.match(m[0], /:not\(:where\(\.star-cta\)\)/,
            '.ng-dark a must exclude .star-cta via :not(:where(.star-cta)) — a bare ' +
            ':not(.star-cta) would add specificity and risk silently outranking some ' +
            'OTHER .star-cta-like component\'s own color, the same way the un-:where()\'d ' +
            'badge/progress-bar exclusion once broke .msg.q');
    });

    it('#dashboard-pane no longer pins .muted to the less-accessible --fg-3', function () {
        const odBridge = readFileSync(join(UI, 'css/od-bridge.css'), 'utf8');
        assert.doesNotMatch(odBridge, /#dashboard-pane\s*\.muted\s*\{\s*color:\s*var\(--fg-3\)/,
            '#dashboard-pane .muted must not override to --fg-3 (2.97:1/3.89:1, under AA) — ' +
            'every other pane\'s .muted uses --fg-2 (design-system.css), which passes');
    });
});
