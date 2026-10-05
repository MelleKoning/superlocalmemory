/**
 * tests/ui/test_od_graph_label_clipping.mjs
 *
 * WHY THIS EXISTS
 * ---------------
 * docs/screenshots/4.1.21/03-knowledge-graph.png shows entity labels like
 * "webhook signing secret" and "mutable account balances" overlapping their
 * neighbours (and running past the canvas edge) in a dense cluster — od-graph
 * .js's draw() drew every node's full label at a fixed offset below the node
 * with no width check at all.
 *
 * Full pairwise label-collision avoidance (every visible label's measured
 * box checked against every other, every animated frame) is a real layout
 * feature, not a small fix, for a live force simulation — out of scope here.
 * What ships instead: clipLabel() binary-searches ctx.measureText() to clip
 * any label wider than LABEL_MAX_WIDTH to an ellipsis, which removes most
 * overlap for free. This test pins that a label wider than the budget comes
 * out shorter with a trailing ellipsis, and that a short label is left
 * untouched (no false-positive clipping of names that never caused the bug).
 *
 * Reuses the canvas-stub + synchronous-rAF harness pattern from
 * test_od_graph_draw_calls.mjs (same file, same stubbing approach) rather
 * than driving a real mousemove-based hover, which would require
 * reimplementing od-graph.js's private W2S() screen-space math outside the
 * module just to predict a hit-test — the hover/selected "always show the
 * full label" branch is exercised by code review, not by this test.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { JSDOM } from 'jsdom';

const __dirname = dirname(fileURLToPath(import.meta.url));
const odGraphSrc = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/od-graph.js'),
  'utf8',
);

const LONG_NAME = 'webhook signing secret rotation policy';
const SHORT_NAME = 'SDK';

const FAKE_GRAPH = { nodes: [], links: [] };
const FAKE_ENTITIES = {
  entities: [
    { entity_id: 'e-long', name: LONG_NAME, type: 'concept', confidence: 0.8, fact_count: 3 },
    { entity_id: 'e-short', name: SHORT_NAME, type: 'concept', confidence: 0.8, fact_count: 1 },
  ],
};

describe('od-graph.js — entity label clipping (docs/screenshots/4.1.21/03-knowledge-graph.png)', function () {
  it('clips a label past the width budget, with an ellipsis, and leaves a short label untouched', async function () {
    const STAGE_W = 640, STAGE_H = 660;
    const drawnText = []; // every fillText(text, ...) call, in draw() order

    const stubCtx = {
      arc() {},
      clearRect() {},
      fillText(text) { drawnText.push(text); },
      fill() {},
      stroke() {},
      beginPath() {},
      moveTo() {},
      lineTo() {},
      save() {},
      restore() {},
      setTransform() {},
      setLineDash() {},
      // A monospace-like stand-in: proportional to character count, same
      // shape as a real font's metrics (not exact px-for-px, but enough for
      // clipLabel's binary search to behave the same way a real font would —
      // wider text measures wider, clipping converges, short text is always
      // under the budget).
      measureText(s) { return { width: String(s || '').length * 7 }; },
      get strokeStyle() { return ''; }, set strokeStyle(_) {},
      get fillStyle() { return ''; }, set fillStyle(_) {},
      get globalAlpha() { return 1; }, set globalAlpha(_) {},
      get lineWidth() { return 1; }, set lineWidth(_) {},
      get font() { return ''; }, set font(_) {},
      get textAlign() { return ''; }, set textAlign(_) {},
      get textBaseline() { return ''; }, set textBaseline(_) {},
    };

    const rafQueue = [];
    function flushRaf(limit) {
      let n = 0;
      while (rafQueue.length > 0 && n++ < (limit ?? 600)) rafQueue.shift()(Date.now());
    }

    let capturedRo = null;
    let stageW = STAGE_W, stageH = STAGE_H;

    const dom = new JSDOM(
      '<!DOCTYPE html><html><head></head><body></body></html>',
      { url: 'http://localhost:8799', runScripts: 'dangerously' },
    );
    const { window } = dom;
    const { document } = window;

    Object.defineProperty(window.HTMLCanvasElement.prototype, 'getContext', {
      value(type) { return type === '2d' ? stubCtx : null; },
      configurable: true,
    });
    Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', {
      get() { return this.id === 'odg-stage' ? stageW : 0; }, configurable: true,
    });
    Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', {
      get() { return this.id === 'odg-stage' ? stageH : 0; }, configurable: true,
    });
    window.requestAnimationFrame = (fn) => { rafQueue.push(fn); return rafQueue.length; };
    window.cancelAnimationFrame = () => {};
    window.ResizeObserver = class {
      constructor(cb) { capturedRo = cb; }
      observe() {}
      disconnect() { capturedRo = null; }
    };
    window.fetch = (url) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(url.includes('/api/entity') ? FAKE_ENTITIES : FAKE_GRAPH),
    });
    window.getComputedStyle = () => ({ getPropertyValue: () => '' });
    window.devicePixelRatio = 2;

    const script = document.createElement('script');
    script.textContent = odGraphSrc;
    document.head.appendChild(script);

    const container = document.createElement('div');
    document.body.appendChild(container);
    window.odRenderGraph(container);

    await new Promise((resolve) => setTimeout(resolve, 0));
    flushRaf(600);
    if (capturedRo) capturedRo([{ contentRect: { width: STAGE_W, height: STAGE_H } }]);
    flushRaf(600);

    assert.ok(drawnText.length > 0, 'draw() never called fillText at all');

    const longDrawn = drawnText.find((t) => t.indexOf(LONG_NAME.slice(0, 10)) === 0);
    assert.ok(longDrawn, `the long label was never drawn (got: ${JSON.stringify(drawnText)})`);
    assert.ok(longDrawn.length < LONG_NAME.length,
      `expected the long label to be clipped shorter than its ${LONG_NAME.length} chars, got ${JSON.stringify(longDrawn)}`);
    assert.match(longDrawn, /…$/, 'a clipped label must end with an ellipsis');

    const shortDrawn = drawnText.find((t) => t.indexOf(SHORT_NAME) === 0);
    assert.equal(shortDrawn, SHORT_NAME,
      `a label already under the width budget must not be touched (got: ${JSON.stringify(shortDrawn)})`);
  });
});
