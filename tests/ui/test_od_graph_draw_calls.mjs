/**
 * Deliverable C — draw-call integration test for the cold-load blank-canvas bug.
 *
 * Architecture:
 *   - jsdom provides a realistic browser DOM (innerHTML, querySelectorAll, MutationObserver).
 *   - requestAnimationFrame is replaced by a synchronous queue flushed on demand.
 *     This removes any dependency on real timer scheduling; the bug was first
 *     "measured" in a hidden browser tab where rAF never fires, so a test that
 *     relies on real rAF would be as unreliable as that measurement.
 *   - A stub 2D context intercepts every canvas call and records arc() coordinates.
 *
 * KEY ASSERTION (Assertion B): after the stage transitions from 0×0 to a real
 * size, at least one arc() call must land INSIDE [0, W] × [0, H].
 *
 * Why coordinates matter:
 *   On the pre-fix build the renderer issued 16,362 arc calls across the settle
 *   budget — every one at x ≈ −2,265.  fit() had been running against W=0/H=0
 *   and parked the camera off-screen.  A test that only counts arc calls would
 *   pass on the broken build; checking that coordinates are in-bounds catches
 *   the actual failure mode.
 *
 * Expected outcomes:
 *   PRE-FIX  (original 4.0.5, no ResizeObserver, loop burns settle budget while
 *             stage is 0×0 and stops):
 *       - After stage gains size, no new arc calls occur because the loop is dead
 *         and nothing restarts it.  Assertion A ("arc > 0") fails.
 *   POST-FIX (loop idles when W=0/H=0 instead of burning settle frames; ResizeObserver
 *             resets fitFrames and calls wake() when stage gains size):
 *       - Loop picks up with W=640, fit() computes correct camera, draw() issues
 *         arc() calls inside [0, 640] × [0, 660].  Both assertions pass.
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

/* ---- Minimal fake payloads ---- */
/* 3 memories, 2 community clusters → buildGraph() produces:
   2 tier-1 (community), 1 tier-2 (entity), 3 tier-3 (episode) = 6 nodes total. */
const FAKE_GRAPH = {
  nodes: [
    { id: 'm1', content: 'Alpha memory',  importance: 0.9, community_id: 0,
      category: 'semantic', entities: ['e1'] },
    { id: 'm2', content: 'Beta memory',   importance: 0.7, community_id: 0,
      category: 'semantic', entities: []    },
    { id: 'm3', content: 'Gamma memory',  importance: 0.5, community_id: 1,
      category: 'episodic', entities: []    },
  ],
  links: [],
};
const FAKE_ENTITIES = {
  entities: [
    { entity_id: 'e1', name: 'TestConcept', type: 'concept', confidence: 0.8, fact_count: 3 },
  ],
};

describe('od-graph draw-call integration — Deliverable C', function () {
  it(
    'arc() coordinates fall inside canvas bounds after stage transitions from 0×0 to real size',
    async function () {
      /* Canvas bounds we grant the stage once it "becomes visible". */
      const STAGE_W = 640, STAGE_H = 660;

      /* ── Arc recording ─────────────────────────────────────────────────── */
      let recording = false;   /* only capture arcs after stage gains size */
      const recorded = [];     /* { x, y, r } */
      const calls = { arc: 0, clearRect: 0, fillText: 0, fill: 0, stroke: 0 };

      /* ── Stub 2D context ───────────────────────────────────────────────── */
      const stubCtx = {
        arc(x, y, r)   { calls.arc++; if (recording) recorded.push({ x, y, r }); },
        clearRect()    { calls.clearRect++; },
        fillText()     { calls.fillText++;  },
        // od-graph.js's label-clipping (clipLabel) measures text before
        // drawing it — a fixed per-character width is close enough for a
        // stub that only needs to return *something* numeric.
        measureText(s) { return { width: String(s || '').length * 6 }; },
        fill()         { calls.fill++;      },
        stroke()       { calls.stroke++;    },
        beginPath()    {},
        moveTo()       {},
        lineTo()       {},
        save()         {},
        restore()      {},
        setTransform() {},
        setLineDash()  {},
        /* property stubs — od-graph.js assigns to these */
        get strokeStyle()  { return ''; }, set strokeStyle(_)  {},
        get fillStyle()    { return ''; }, set fillStyle(_)    {},
        get globalAlpha()  { return 1;  }, set globalAlpha(_)  {},
        get lineWidth()    { return 1;  }, set lineWidth(_)    {},
        get font()         { return ''; }, set font(_)         {},
        get textAlign()    { return ''; }, set textAlign(_)    {},
        get textBaseline() { return ''; }, set textBaseline(_) {},
      };

      /* ── Synchronous rAF queue ─────────────────────────────────────────── */
      const rafQueue = [];
      function flushRaf(limit) {
        let n = 0;
        while (rafQueue.length > 0 && n++ < (limit ?? 600)) {
          rafQueue.shift()(Date.now());
        }
      }

      /* ── ResizeObserver handle — fired manually by the test ────────────── */
      let capturedRo = null;

      /* ── Stage dimensions — controlled by the test ─────────────────────── */
      let stageW = 0, stageH = 0;

      /* ── Build jsdom arena ─────────────────────────────────────────────── */
      const dom = new JSDOM(
        '<!DOCTYPE html><html><head></head><body></body></html>',
        { url: 'http://localhost:8799', runScripts: 'dangerously' },
      );
      const { window } = dom;
      const { document } = window;

      /* canvas.getContext('2d') always returns our stub */
      Object.defineProperty(window.HTMLCanvasElement.prototype, 'getContext', {
        value(type) { return type === '2d' ? stubCtx : null; },
        configurable: true,
      });

      /* clientWidth / clientHeight on #odg-stage reports our controlled dims */
      Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', {
        get() { return this.id === 'odg-stage' ? stageW : 0; },
        configurable: true,
      });
      Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', {
        get() { return this.id === 'odg-stage' ? stageH : 0; },
        configurable: true,
      });

      /* Synchronous rAF — callbacks execute when flushRaf() is called */
      window.requestAnimationFrame  = fn => { rafQueue.push(fn); return rafQueue.length; };
      window.cancelAnimationFrame   = () => {};

      /* ResizeObserver mock */
      window.ResizeObserver = class {
        constructor(cb) { capturedRo = cb; }
        observe()       {}
        disconnect()    { capturedRo = null; }
      };

      /* fetch mock — resolves with fake payloads */
      window.fetch = url => Promise.resolve({
        ok: true,
        json: () => Promise.resolve(
          url.includes('/api/entity') ? FAKE_ENTITIES : FAKE_GRAPH,
        ),
      });

      /* getComputedStyle — empty CSS vars; readPalette() applies hardcoded defaults */
      window.getComputedStyle = () => ({ getPropertyValue: () => '' });

      /* devicePixelRatio */
      window.devicePixelRatio = 2;

      /* ── Inject od-graph.js — IIFE runs immediately in jsdom window scope ── */
      const script = document.createElement('script');
      script.textContent = odGraphSrc;
      document.head.appendChild(script);
      assert.ok(
        typeof window.odRenderGraph === 'function',
        'od-graph.js must expose window.odRenderGraph after the IIFE executes',
      );

      /* ── Mount: call odRenderGraph with the stage starting at 0×0 ───────── */
      const container = document.createElement('div');
      document.body.appendChild(container);
      window.odRenderGraph(container);

      /* ── Drain the fetch microtask chain ────────────────────────────────── */
      /* Promise.resolve() callbacks are microtasks; they resolve before a
         setTimeout callback fires, so after this await buildGraph() has run
         and NODES is populated. */
      await new Promise(resolve => setTimeout(resolve, 0));

      /* Sanity: buildGraph() must have called updateCount(), which writes DOM. */
      const countEl = container.querySelector('#odg-count');
      assert.ok(
        countEl && /\d+\s*nodes/.test(countEl.textContent),
        `#odg-count should show "N nodes" after data loads; got: "${countEl?.textContent}"`,
      );

      /* ── Phase 1: run the loop while stage is still 0×0 ─────────────────── */
      /* Post-fix: each iteration hits the !W||!H guard, idles (reschedules
         rAF), and returns — settle budget is never spent.
         Pre-fix:  each iteration ticks, fits against W=0, draws off-screen.
         After SETTLE_MAX_FRAMES (180) the pre-fix loop sets running=false. */
      flushRaf(600);   /* generous cap: SETTLE_MAX_FRAMES=180, so 600 covers 3× */

      /* ── Phase 2: reset counters and start recording ─────────────────────── */
      calls.arc = 0;
      recorded.length = 0;
      recording = true;

      /* ── Stage acquires real pixel dimensions ─────────────────────────────── */
      stageW = STAGE_W;
      stageH = STAGE_H;

      /* ── Fire ResizeObserver (simulates tab becoming visible) ─────────────── */
      /* Post-fix: fires resize() (commits canvas to 1280×1320 at DPR 2) and
         resets fitFrames=0 so the camera re-fits against the real W and H.
         Pre-fix with no ResizeObserver: capturedRo is null; nothing fires. */
      if (capturedRo) {
        capturedRo([{ contentRect: { width: STAGE_W, height: STAGE_H } }]);
      }

      /* ── Run the loop that the observer (should have) started / kept alive ── */
      flushRaf(600);

      /* ── ASSERTION A: arc calls were issued after stage gained size ────────── */
      assert.ok(
        calls.arc > 0,
        `Expected arc() calls after stage transitioned from 0×0 to ${STAGE_W}×${STAGE_H}. ` +
        `Got 0. ` +
        `Pre-fix diagnosis: ${capturedRo === null
          ? 'no ResizeObserver registered (capturedRo is null) — the loop died during Phase 1 and nothing restarted it.'
          : 'ResizeObserver was registered but the loop may have stopped without being restarted.'}`,
      );

      /* ── ASSERTION B: at least one arc falls INSIDE the canvas bounds ──────── */
      /* This distinguishes the real fix from the broken state.
         Pre-fix (hypothetical): even if the loop were restarted, fit() had
         computed scale/ox/oy against W=0, parking the camera off-screen at
         x ≈ −2,265.  A woken loop faithfully redraws nothing visible.
         Post-fix: ResizeObserver resets fitFrames=0 before wake(); the loop
         recomputes fit() against W=640 and maps all nodes into [0,640]×[0,660]. */
      const inBounds = recorded.filter(
        ({ x, y }) => x >= 0 && x <= STAGE_W && y >= 0 && y <= STAGE_H,
      );

      const sample = recorded.slice(0, 6)
        .map(({ x, y }) => `(${x.toFixed(0)},${y.toFixed(0)})`)
        .join(' ');

      assert.ok(
        inBounds.length > 0,
        `Expected ≥1 arc() with center coordinates inside [0,${STAGE_W}]×[0,${STAGE_H}]. ` +
        `Got ${inBounds.length} in-bounds out of ${recorded.length} total. ` +
        `Sample arc centers: ${sample || '(none recorded)'}. ` +
        `Likely cause: fit() computed the camera transform against W=0 and the ` +
        `settle budget was not reset when the stage gained real dimensions.`,
      );
    },
  );
});
