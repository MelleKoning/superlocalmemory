/**
 * The live event stream's filter offers only events the product emits, and the
 * page subscribes to each of them.
 *
 * The filter offered "Memory created" (memory.created), an event nothing
 * emits: the event bus refuses that type. A capture is emitted as
 * memory.captured and a save as memory.stored, neither of which the page
 * subscribed to, so neither ever appeared in the stream. "Memory recalled" and
 * "Agent connected" were offered too, and nothing emits either.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'fs';
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

/** Every event type a product module (not the bus itself) passes to an emit. */
function emittedEventTypes(valid) {
  const found = new Set();
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) { walk(path); continue; }
      if (!name.endsWith('.py') || path.endsWith(join('infra', 'event_bus.py'))) continue;
      const source = readFileSync(path, 'utf8');
      for (const type of valid) {
        if (source.includes(`"${type}"`)) found.add(type);
      }
    }
  };
  walk(join(ROOT, 'src/superlocalmemory'));
  return found;
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
    const emitted = emittedEventTypes(valid);
    for (const { value } of options) {
      assert.ok(valid.has(value), `filter option ${value} is not an event the bus accepts`);
      assert.ok(emitted.has(value), `nothing in the product emits ${value}`);
    }
  });

  it('subscribes to every event type it offers', function () {
    for (const { value } of options) {
      assert.ok(subscribed.has(value), `the page never subscribes to ${value}`);
    }
  });

  it('offers saved, captured, corrected and deleted memories', function () {
    assert.deepEqual(options, [
      { value: 'memory.stored', label: 'Memory saved' },
      { value: 'memory.captured', label: 'Memory captured' },
      { value: 'memory.updated', label: 'Memory corrected' },
      { value: 'memory.deleted', label: 'Memory deleted' },
    ]);
  });
});
