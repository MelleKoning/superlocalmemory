// Regression test for GitHub issue #112, part 2.
//
// The dashboard's one-click Mode C button (`.mode-btn[data-mode="c"]`) calls
// PUT /api/v3/mode. When the server refuses — nothing configured, neither a
// cloud key nor a custom endpoint — the error toast must show the server's
// plain next step, not just "requires a cloud API key" (which used to be
// shown even when a keyless custom endpoint was already configured and the
// switch should have succeeded).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule } from './harness.mjs';

describe('dashboard Mode C quick-switch button', function() {
    it('surfaces the server refusal naming both a key and an endpoint as the next step', async function() {
        const h = buildHarness([], { ok: true, status: 200, json: {} });
        h.document.body.innerHTML = `
            <button class="mode-btn btn-sm" data-mode="c">C</button>
        `;
        // dashboard.js also wires an auto-refresh (DOMContentLoaded timer,
        // focus/visibilitychange listeners) that calls other endpoints —
        // stub those harmlessly so only the mode-switch call is asserted.
        h.window.fetch = async function(url, options) {
            if (url !== '/api/v3/mode') {
                return { ok: false, status: 404, json: async function() { return {}; } };
            }
            assert.equal(options.method, 'PUT');
            assert.deepEqual(JSON.parse(options.body), { mode: 'c' });
            return {
                ok: false,
                status: 400,
                json: async function() {
                    return {
                        error: (
                            'Mode C needs a cloud API key, or a configured custom endpoint ' +
                            '(llama.cpp, vLLM, or any other OpenAI-compatible server). ' +
                            'Configure one in Settings → Step 2 (uses POST /api/v3/mode/set), ' +
                            'or run `slm provider set`.'
                        ),
                        code: 'mode_c_requires_api_key',
                    };
                },
            };
        };
        evalModule(h.window, 'core.js');
        evalModule(h.window, 'dashboard.js');

        h.document.querySelector('.mode-btn[data-mode="c"]').click();
        await new Promise(function(resolve) { setTimeout(resolve, 20); });

        const toastText = h.document.body.textContent;
        assert.match(toastText, /cloud API key/);
        assert.match(toastText, /custom endpoint/);
    });
});
