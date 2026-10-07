/* od-decay-preview.js — the Forgetting group's decay preview (moved out of
 * od-settings.js in 4.1.22, unchanged in behaviour).
 *
 * Three curves, one per source-trust level. Shows what the kappa setting
 * actually does before it is saved: the same memory strength fading at three
 * different rates depending on where the memory came from. At kappa 0 the
 * three curves coincide, which is the honest picture of trust-modulation
 * being off.
 *
 * Exposes window.odDecayPreview: build() -> element, draw(kappa).
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var CANVAS_ID = 'od-s-forg-decay';
  var DECAY_TRUSTS = [
    { trust: 0.0, color: '#e0245e', label: 'low trust' },
    { trust: 0.5, color: '#f59e0b', label: 'medium' },
    { trust: 1.0, color: '#10b981', label: 'trusted' }
  ];

  /* The decay rate is per HOUR, and the window is chosen so the three curves
     are actually distinguishable. At a strong memory's strength (100, the
     configured maximum) retention over sixty days is indistinguishable from
     zero for every trust level, so a two-month preview would draw three flat
     lines along the bottom and say nothing. Three days separates them. */
  var DECAY_HOURS = 72;
  var DECAY_STRENGTH = 100.0;

  function build() {
    var wrap = document.createElement('div');
    Object.assign(wrap.style, { display:'flex', flexDirection:'column', gap:'6px' });
    var cvs = document.createElement('canvas');
    cvs.id = CANVAS_ID;
    cvs.width = 300; cvs.height = 96;
    Object.assign(cvs.style, {
      border:'1px solid var(--border)', borderRadius:'4px', display:'block'
    });
    wrap.appendChild(cvs);
    var key = document.createElement('div');
    Object.assign(key.style, { display:'flex', gap:'10px' });
    DECAY_TRUSTS.forEach(function (t) {
      var span = document.createElement('span');
      span.textContent = t.label;
      Object.assign(span.style, { fontSize:'10.5px', color: t.color, fontFamily:'var(--font-mono)' });
      key.appendChild(span);
    });
    wrap.appendChild(key);
    return wrap;
  }

  function retentionAt(hours, kappa, trust) {
    // Mirrors math/ebbinghaus.trust_modulated_retention exactly:
    //   lambda_eff = (1 / S) * (1 + kappa * (1 - trust))
    //   retention  = exp(-lambda_eff * hours)
    var lambdaEff = (1.0 / DECAY_STRENGTH) * (1.0 + kappa * (1.0 - trust));
    return Math.exp(-lambdaEff * hours);
  }

  function draw(kappa) {
    var cvs = document.getElementById(CANVAS_ID);
    if (!cvs || !cvs.getContext) return;
    var ctx = cvs.getContext('2d');
    if (!ctx) return;
    var W = cvs.width, H = cvs.height, STEPS = 72;
    ctx.clearRect(0, 0, W, H);
    DECAY_TRUSTS.forEach(function (t) {
      ctx.beginPath();
      for (var i = 0; i <= STEPS; i++) {
        var hours = (i / STEPS) * DECAY_HOURS;
        var retention = retentionAt(hours, kappa, t.trust);
        var x = (i / STEPS) * (W - 2) + 1;
        var y = H - 2 - retention * (H - 4);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    });
  }

  window.odDecayPreview = { build: build, draw: draw, retentionAt: retentionAt };
}());
