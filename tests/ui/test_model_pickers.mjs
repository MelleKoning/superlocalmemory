/**
 * tests/ui/test_model_pickers.mjs — Settings model/embedding suggestions (4.1.22).
 *
 * The dashboard's model fields used to be a bare text box ("e.g. llama3.2").
 * They now offer the installed Ollama models and SLM's own recommendations,
 * each with a one-line quality/speed hint, via GET /api/v3/models/catalog —
 * free typing still works, the catalog is advisory only.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function catalogFixture() {
  return {
    local_recommendations: [
      { model: 'gemma3:4b', reason: '1 of 102 test sentences gave a changed fact.',
        installed: true, fits: true, label: 'Gemma 3 4B' },
      { model: 'llama3.2', reason: '36 of 102 test sentences gave a changed fact.',
        installed: true, fits: true, label: 'Llama 3.2 3B' },
    ],
    hosted_llms: [
      { id: 'openai/gpt-6-luna', label: 'GPT-6 Luna', price: '$0.10 / $0.50',
        advice: 'Lowest cost per memory.' },
    ],
    local_embedders: [
      { id: 'nomic-ai/nomic-embed-text-v1.5', label: 'Nomic', advice: 'Default.',
        dimension: 768 },
    ],
    hosted_embedders: [],
    ollama: { installed: ['gemma3:4b', 'llama3.2:latest'], reachable: true },
    best_local_llm: 'gemma3:4b',
  };
}

async function mount({ catalogOk = true, mode = {} } = {}) {
  const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
  const modeBody = Object.assign({ mode: 'b', provider: 'ollama', model: '' }, mode);

  h.window.fetch = async function (url) {
    if (url === '/api/v3/models/catalog') {
      if (!catalogOk) return { ok: false, status: 500, json: async function () { return {}; } };
      return { ok: true, status: 200, json: async function () { return catalogFixture(); } };
    }
    if (url === '/api/v3/mode') {
      return { ok: true, status: 200, json: async function () { return modeBody; } };
    }
    // Every other GET used by loadAll() for the other settings groups —
    // not under test here, must not make odRenderSettings throw.
    return { ok: true, status: 200, json: async function () { return {}; } };
  };

  evalModule(h.window, 'od-settings.js');
  const container = h.document.getElementById('mount');
  h.window.odRenderSettings(container);
  // loadMode() and loadModelCatalog() race; flush twice so whichever settles
  // last (the one that must win — see od-settings.js fillModelSuggestions)
  // has definitely run.
  await flushPromises();
  await flushPromises();
  return h;
}

function ids(h) {
  return {
    provider: h.document.getElementById('od-s-provider'),
    model: h.document.getElementById('od-s-model'),
    modelList: h.document.getElementById('od-s-model-list'),
    modelHint: h.document.getElementById('od-s-model-hint'),
    embModel: h.document.getElementById('od-s-emb-model'),
    embModelHint: h.document.getElementById('od-s-emb-model-hint'),
  };
}

describe('dashboard model & embedding suggestions', function () {
  it('(a) LLM datalist has the local recommendations, installed first', async function () {
    const h = await mount();
    const { modelList } = ids(h);
    assert.equal(modelList.children.length, 2);
    assert.equal(modelList.children[0].value, 'gemma3:4b');
    assert.equal(modelList.children[1].value, 'llama3.2');
  });

  it('(b) empty saved model is pre-filled with best_local_llm and shows its reason', async function () {
    const h = await mount({ mode: { model: '' } });
    const { model, modelHint } = ids(h);
    assert.equal(model.value, 'gemma3:4b');
    assert.match(modelHint.textContent, /1 of 102 test sentences/);
  });

  it('(c) switching provider to openrouter replaces options with the hosted id', async function () {
    const h = await mount();
    const { provider, modelList } = ids(h);
    provider.value = 'openrouter';
    provider.dispatchEvent(new h.window.Event('change'));
    assert.equal(modelList.children.length, 1);
    assert.equal(modelList.children[0].value, 'openai/gpt-6-luna');
  });

  it('(d) typing an uninstalled ollama model shows the pull hint', async function () {
    const h = await mount();
    const { model, modelHint } = ids(h);
    model.value = 'mistral:7b';
    model.dispatchEvent(new h.window.Event('input'));
    assert.equal(modelHint.textContent, 'Not installed. Pull it first: ollama pull mistral:7b');
  });

  it('(e) a saved non-empty model is NOT overwritten', async function () {
    const h = await mount({ mode: { model: 'llama3.2' } });
    const { model } = ids(h);
    assert.equal(model.value, 'llama3.2');
  });

  it('(f) a failing catalog fetch leaves the inputs usable and raises no exception', async function () {
    const h = await mount({ catalogOk: false });
    const { model, modelHint, modelList } = ids(h);
    assert.equal(model.disabled, false);
    assert.equal(model.value, ''); // no catalog => no prefill, but still editable
    assert.equal(modelHint.textContent, '');
    assert.equal(modelList.children.length, 0);
    // Free typing still works with no catalog loaded.
    model.value = 'anything-i-want';
    model.dispatchEvent(new h.window.Event('input'));
    assert.equal(model.value, 'anything-i-want');
  });

  it('(g) the embedding hint includes the dimension', async function () {
    const h = await mount();
    const { embModel, embModelHint } = ids(h);
    embModel.value = 'nomic-ai/nomic-embed-text-v1.5';
    embModel.dispatchEvent(new h.window.Event('input'));
    assert.match(embModelHint.textContent, /\(dimension 768\)/);
  });
});
