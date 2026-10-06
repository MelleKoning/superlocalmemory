/**
 * tests/ui/test_fg3_classification.mjs
 *
 * Coordinator decision: --fg-3 is kept for genuinely decorative glyphs
 * only ("—", "…", separators). Every informational use (a status message,
 * a stat's label, a config key, a count, a chart subtitle — anything a
 * user needs to read) moved to --fg-2 or the new .muted-info class.
 * <code> and danger-red close-misses (4.27-4.37:1) were fixed in the
 * product's own CSS, scoped to .app (the dashboard root), not by editing
 * Bootstrap. This pins the structural facts a full-Chromium audit can't
 * be re-verified by in CI (jsdom can't resolve CSS var() in
 * getComputedStyle — confirmed empirically earlier in this work).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const designSystem = readFileSync(join(UI, 'css/design-system.css'), 'utf8');
const neuralGlass = readFileSync(join(UI, 'css/neural-glass.css'), 'utf8');

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

describe('.muted-info exists for informational text that is not --fg-3', function () {
    it('is defined as --fg-2', function () {
        assert.match(designSystem, /\.muted-info\s*\{\s*color:\s*var\(--fg-2\)/,
            '.muted-info must read --fg-2, the token proven to pass AA in both themes');
    });

    it('.dim (--fg-3) still exists, unchanged, for decorative glyphs', function () {
        assert.match(designSystem, /\.dim\s*\{\s*color:\s*var\(--fg-3\)/,
            '.dim must keep --fg-3 — it is still the right token for decorative ' +
            'placeholders ("—" in od-optimize.js\'s od-opt-inr-saved is the one ' +
            'deliberately-kept instance; see the audit report)');
    });
});

describe('.cnt counts (chip and tab) are informational, not decorative', function () {
    it('.chip .cnt and .tab .cnt both read --fg-2', function () {
        assert.match(designSystem, /\.chip \.cnt\s*\{\s*color:\s*var\(--fg-2\)/,
            'a node-budget / filter-chip count is informational');
        assert.match(designSystem, /\.tab \.cnt\s*\{[^}]*color:\s*var\(--fg-2\)/,
            'a tab count ("All memories 97") is informational');
    });
});

describe('<code> and .text-danger: fixed in our own CSS, scoped to .app', function () {
    it('.app .text-danger overrides Bootstrap\'s own color, using --danger-text', function () {
        assert.match(neuralGlass, /\.app \.text-danger\s*\{\s*color:\s*var\(--danger-text\)/,
            '.app .text-danger must exist and read --danger-text, not Bootstrap\'s ' +
            'own --bs-danger (#dc3545, measured 4.3:1/4.37:1 — under WCAG AA 4.5:1)');
    });

    it('[data-bs-theme="light"] .app code overrides with a literal AA-passing hex', function () {
        const m = neuralGlass.match(/\[data-bs-theme="light"\] \.app code\s*\{\s*color:\s*(#[0-9a-fA-F]{6})/);
        assert.ok(m, '[data-bs-theme="light"] .app code rule not found');
        const rgb = [1, 3, 5].map((i) => parseInt(m[1].slice(i, i + 2), 16));
        assert.ok(contrast(rgb, [248, 249, 250]) >= 4.5,
            `${m[1]} on a near-white page background is ${contrast(rgb, [248, 249, 250]).toFixed(2)}:1, under AA`);
    });

    it('.ng-dark code (already dashboard-scoped) uses a literal AA-passing hex, not --ng-accent', function () {
        const m = neuralGlass.match(/\.ng-dark code \{[^}]*color:\s*(#[0-9a-fA-F]{6})/);
        assert.ok(m, '.ng-dark code color not found or still a var()');
        const rgb = [1, 3, 5].map((i) => parseInt(m[1].slice(i, i + 2), 16));
        // Checked against the tightest real background this renders on
        // (loops-pane's bare code chip, --ng-bg-glass-active over a dark card).
        assert.ok(contrast(rgb, [36, 36, 38]) >= 4.5,
            `${m[1]} on (36,36,38) is ${contrast(rgb, [36, 36, 38]).toFixed(2)}:1, under AA`);
    });
});
