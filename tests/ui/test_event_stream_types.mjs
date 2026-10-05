/**
 * The live event stream's filter offers only events the product emits, and the
 * page subscribes to each of them.
 *
 * The filter offered "Memory created" (memory.created), an event nothing
 * emits: the event bus refuses that type. A capture is emitted as
 * memory.captured, which the page did not even subscribe to, so captures never
 * appeared in the stream under any filter.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import vm from 'vm';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '../..');
const UI = join(ROOT, 'src/superlocalmemory/ui');

function validEventTypes() {
  const source = readFileSync(join(ROOT, 'src/superlocalmemory/infra/event_bus.py'), 'utf8');
  const block = source.match(/VALID_EVENT_TYPES = frozenset\(\[([\s\S]*?)\]\)/);
  assert.ok(block, 'VALID_EVENT_TYPES not found in event_bus.py');
  return new Set([...block[1].matchAll(/"([a-z_.]+)"/g)].map((m) => m[1]));
}

function filterOptions() {
  const html = readFileSync(join(UI, 'index.html'), 'utf8');
  const select = html.match(/<select[^>]*id="event-type-filter"[\s\S]*?<\/select>/);
  assert.ok(select, 'the event type filter is missing from index.html');
  return [...select[0].matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
    .map((m) => ({ value: m[1], label: m[2].trim() }))
    .filter((o) => o.value);
}

function subscribedTypes() {
  const listened = [];
  class FakeEventSource {
    constructor() { this.onopen = null; this.onmessage = null; this.onerror = null; }
    addEventListener(type) { listened.push(type); }
  }
  const context = { EventSource: FakeEventSource, console,
                    document: { getElementById: () => null } };
  vm.createContext(context);
  vm.runInContext(readFileSync(join(UI, 'js/events.js'), 'utf8'), context);
  context.initEventStream();
  return new Set(listened);
}

describe('live event stream filter', function () {
  const options = filterOptions();
  const valid = validEventTypes();
  const subscribed = subscribedTypes();

  it('offers only event types the product emits', function () {
    for (const { value } of options) {
      assert.ok(valid.has(value), `filter option ${value} is not an event the product emits`);
    }
  });

  it('subscribes to every event type it offers', function () {
    for (const { value } of options) {
      assert.ok(subscribed.has(value), `the page never subscribes to ${value}`);
    }
  });

  it('offers captured memories as "Memory captured"', function () {
    const captured = options.find((o) => o.value === 'memory.captured');
    assert.ok(captured, 'no filter option for memory.captured');
    assert.equal(captured.label, 'Memory captured');
    assert.ok(!options.some((o) => o.value === 'memory.created'));
  });
});
