/**
 * Memory activity must key by the browser's LOCAL calendar day (audit 4.1.20 M5).
 *
 * The chart built local midnights and keyed them with toISOString(), which is
 * the UTC date. East of UTC (India, +05:30) local midnight is the previous UTC
 * day, so today's facts never matched and the chart said there was no history.
 * The server and the browser now agree: the browser sends its offset, the
 * server buckets by local day, and the browser keys by local YYYY-MM-DD.
 */

// Must be set before any Date is created in this process.
process.env.TZ = 'Asia/Kolkata';

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dashboardSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/dashboard.js'),
  'utf8',
);

function localKey(d) {
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0')
    + '-' + String(d.getDate()).padStart(2, '0');
}

describe('dashboard memory activity date keys', function () {
  it('counts today east of UTC and sends the offset to /api/timeline', async function () {
    const dom = new JSDOM(`<!doctype html><html><body>
      <div id="sp-big"></div><div id="k-avg-day"></div>
    </body></html>`, { runScripts: 'dangerously', url: 'http://localhost:8765/' });
    const { window } = dom;
    assert.equal(new window.Date(2026, 9, 4).getTimezoneOffset(), -330,
      'the test must run at +05:30');

    const now = new Date();
    const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
    const timeline = [
      { period: localKey(now), count: 3 },
      { period: localKey(yesterday), count: 2 },
    ];
    const urls = [];
    window.fetch = function (url) {
      urls.push(String(url));
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.resolve({ timeline }); },
      });
    };
    let plotted = null;
    window.slmSpark = function (counts) { plotted = counts.slice(); return '<svg></svg>'; };

    const script = window.document.createElement('script');
    script.textContent = dashboardSource;
    window.document.head.appendChild(script);
    await window.loadDashboardTimeline();

    const timelineUrl = urls.find((u) => u.startsWith('/api/timeline'));
    assert.ok(timelineUrl, 'timeline was requested');
    assert.match(timelineUrl, /[?&]tz_offset_minutes=330(&|$)/);
    assert.ok(plotted, 'the chart was drawn instead of "No dated memory history"');
    assert.equal(plotted.length, 365);
    assert.equal(plotted[364], 3, 'today is the last point');
    assert.equal(plotted[363], 2, 'yesterday is the one before');
    dom.window.close();
  });
});
