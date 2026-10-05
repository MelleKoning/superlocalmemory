/**
 * tests/ui/test_recall_lab_badge_contrast.mjs
 *
 * WHY THIS EXISTS
 * ---------------
 * Recall Lab's channel-status chips (recall-lab.js's buildChannelHealth —
 * "bm25 . ok", "hopfield . off", etc.) and the per-result Semantic/BM25/
 * Entity/Temporal score bars (buildChannelBar) are plain Bootstrap
 * `.badge`/`.progress-bar` elements. neural-glass.css had a light-mode rule,
 * meant to fix near-white dark-theme text leaking onto plain prose in light
 * mode:
 *
 *   [data-bs-theme="light"] p, span, div, label, td, th { color: inherit; }
 *
 * Because that selector's specificity (0,1,1 per branch) beats Bootstrap's
 * own `.badge`/`.progress-bar` rules (0,1,0), it also overrode THEIR
 * explicit, already-correct text color with whatever happened to be
 * inherited from an ancestor — on a .bg-success/.bg-secondary chip that
 * inherited color was frequently near-identical to the chip's own
 * background (measured directly off docs/screenshots/4.1.21/02-recall-lab.png:
 * the "ok" chips rendered #65758B-ish text on #198754, and one "no hits"/
 * "off" chip rendered #65758B-ish text on a #6c757d background it is
 * barely distinguishable from).
 *
 * jsdom does not resolve CSS custom properties in getComputedStyle (verified
 * empirically: it returns the literal "var(--bs-badge-color)" string, not a
 * color), so a real cascade/contrast assertion has to run in an actual
 * browser engine — that was done directly (Playwright/Chromium) while
 * building this fix. What IS practical and durable in this jsdom suite:
 *
 *   1. pin that the light-mode `color: inherit` reset still excludes the
 *      Bootstrap components it must not touch, so nobody "simplifies" the
 *      selector back to the broken, unscoped form;
 *   2. compute WCAG AA contrast with the exact literal (non-var()) colors
 *      this fix introduced for the two hues (.bg-warning yellow,
 *      .bg-info cyan) where Bootstrap's own flat white/near-white default
 *      text fails AA regardless of theme.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CSS_PATH = join(__dirname, '../../src/superlocalmemory/ui/css/neural-glass.css');
const css = readFileSync(CSS_PATH, 'utf8');

// WCAG 2.x relative luminance + contrast ratio, sRGB 0-255 inputs.
function luminance([r, g, b]) {
    const lin = (c) => {
        const v = c / 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a, b) {
    const la = luminance(a);
    const lb = luminance(b);
    const [hi, lo] = la >= lb ? [la, lb] : [lb, la];
    return (hi + 0.05) / (lo + 0.05);
}

const WHITE = [255, 255, 255];
const DARK_NAVY = [0x11, 0x18, 0x27]; // #111827, this file's existing light-mode text color

describe('neural-glass.css — light-mode text-color reset does not swallow badges/bars', function () {
    it('excludes .badge from the span color:inherit reset', function () {
        const m = css.match(
            /\[data-bs-theme="light"\]\s*p,\s*\n\[data-bs-theme="light"\]\s*span([^,]*),/,
        );
        assert.ok(m, 'could not find the light-mode span color:inherit rule at all');
        assert.match(m[1], /:not\(:where\(\.badge\)\)/,
            'the light-mode span reset no longer excludes .badge — Bootstrap badges ' +
            '(Recall Lab\'s channel-status chips) will lose their own text color to ' +
            'whatever they inherit, which is frequently near-invisible against their ' +
            'own colored background. (Must be :not(:where(.badge)), not a bare ' +
            ':not(.badge) — the bare form adds specificity that broke .msg.q\'s own ' +
            'white chat-bubble text; see the comment above this rule in neural-glass.css.)');
    });

    it('excludes .progress-bar from the div color:inherit reset', function () {
        const m = css.match(/\[data-bs-theme="light"\]\s*div([^,]*),/);
        assert.ok(m, 'could not find the light-mode div color:inherit rule at all');
        assert.match(m[1], /:not\(:where\(\.progress-bar\)\)/,
            'the light-mode div reset no longer excludes .progress-bar — the per-result ' +
            'Semantic/BM25/Entity/Temporal score bars in Recall Lab will lose their own ' +
            'text color the same way the badges did');
    });

    it('the exclusion costs ZERO specificity, so it never outranks a two-class component', function () {
        // Real regression, found while verifying an unrelated dark-theme token
        // fix: a bare `:not(.badge)` has specificity (0,1,1) per branch — ONE
        // MORE than a plain element selector's (0,1,0) — because :not()'s
        // specificity is that of its argument. That extra point pushed this
        // whole rule from (0,1,1) to (0,2,1), which started beating
        // .msg.q { color: #fff } (two classes, (0,2,0)) — a component this
        // rule was never meant to touch. :where(.badge) has ZERO specificity
        // by spec, so :not(:where(.badge)) keeps the overall rule at its
        // original (0,1,1): exactly as weak as it always was for every
        // two-class (or higher) component already relying on that, while
        // still excluding .badge/.progress-bar functionally.
        assert.doesNotMatch(css.slice(css.indexOf('[data-bs-theme="light"] p,'),
                                      css.indexOf('[data-bs-theme="light"] p,') + 400),
            /:not\((?!:where\()/,
            'a :not(...) in this rule is not wrapped in :where(...) — it will ' +
            'cost specificity and can silently outrank a two-class component ' +
            '(e.g. .msg.q) that this rule was never meant to touch');
    });
});

describe('Bootstrap badge/progress-bar text on light hues meets WCAG AA (4.5:1)', function () {
    it('white-on-success/secondary/danger (Bootstrap default, untouched by this fix) passes', function () {
        // These three already passed before this fix and must keep passing —
        // this fix must not have regressed them.
        assert.ok(contrast(WHITE, [25, 135, 84]) >= 4.5, 'white on .bg-success');
        assert.ok(contrast(WHITE, [108, 117, 125]) >= 4.5, 'white on .bg-secondary');
        assert.ok(contrast(WHITE, [220, 53, 69]) >= 4.5, 'white on .bg-danger');
    });

    it('white-on-warning FAILS (why .badge.bg-warning/.progress-bar.bg-warning need an override)', function () {
        const ratio = contrast(WHITE, [255, 193, 7]);
        assert.ok(ratio < 4.5,
            `expected white-on-#ffc107 to fail AA (it is ${ratio.toFixed(2)}:1) — if ` +
            'Bootstrap ever changes this default, the override below may no longer be needed');
    });

    it('white-on-info FAILS (why .progress-bar.bg-info needs an override)', function () {
        const ratio = contrast(WHITE, [13, 202, 240]);
        assert.ok(ratio < 4.5,
            `expected white-on-#0dcaf0 to fail AA (it is ${ratio.toFixed(2)}:1)`);
    });

    it('this fix\'s dark-navy override on warning/info passes AA in both directions', function () {
        assert.ok(contrast(DARK_NAVY, [255, 193, 7]) >= 4.5, '#111827 on .bg-warning (yellow)');
        assert.ok(contrast(DARK_NAVY, [13, 202, 240]) >= 4.5, '#111827 on .bg-info (cyan)');
    });

    it('neural-glass.css actually declares the dark-navy overrides for both hues', function () {
        assert.match(css, /\[data-bs-theme="light"\]\s*\.badge\.bg-warning\s*\{[^}]*color:\s*#111827/,
            'missing (or reworded past matching) the light-mode .badge.bg-warning override');
        assert.match(css, /\.progress-bar\.bg-info,\s*\n\.progress-bar\.bg-warning\s*\{[^}]*color:\s*#111827/,
            'missing (or reworded past matching) the .progress-bar.bg-info/.bg-warning override');
    });
});
