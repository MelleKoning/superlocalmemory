# Dependency audit exceptions

The release dependency audit is fail-closed except for the exact advisories
listed below. Each exception must name the transitive dependency path, explain
why the vulnerable API is not reachable from untrusted input, and be removed
as soon as a stable patched release is available.

**Status as of 2026-10-02:** one exception remains, and it has no patched
release upstream. The PyTorch and setuptools exceptions are retired: SLM pins
PyTorch 2.13.0, whose setuptools constraint permits the patched 83.0.0 release.

## GHSA-8mgp-746c-j5xp — NLTK 3.10.3

- **Dependency path:** `superlocalmemory -> llmlingua 0.2.2 -> nltk`.
  SuperLocalMemory pins `nltk==3.10.3` directly so the resolver cannot pick an
  older release that carries the fixed advisories.
- **Advisory:** several model-artifact APIs (`TransitionParser.train`,
  `TransitionParser.parse`, `AveragedPerceptron.save`/`load`,
  `PerceptronTagger.save_to_json`, `save_maxent_params`) bypass NLTK's own
  `pathsec` sandbox when an application enables `pathsec` enforcement and lets
  untrusted input choose a model path. GitHub lists `<= 3.10.3` as affected
  with no patched release; 3.10.3 is the newest release on PyPI.
- **Exposure:** none. SuperLocalMemory never imports NLTK. LLMLingua calls only
  `nltk.sent_tokenize`, which is not one of the affected APIs. Nothing in the
  installed runtime calls the affected APIs or enables `pathsec`, so no model
  path reaches them from any CLI, HTTP, dashboard or MCP input.
- **Removal condition:** an NLTK release that fixes the advisory. Pin it and
  remove the `--ignore-vuln` line from `.github/workflows/test.yml`.
- **Review deadline:** 2026-11-02.

## Retired in 4.1.18: GHSA-rrmf-rvhw-rf47 — PyTorch 2.11.0

Memory corruption in `torch.jit.script()`. SuperLocalMemory never called
TorchScript compilation, and 4.1.18 moves to PyTorch 2.13.0, the first release
outside the affected range. The move was benchmarked against 2.11.0 on the
cached embedding and reranker models: embeddings are unchanged, reranker
orderings are unchanged, and the embedding worker's peak memory fell. The CI
suppression is removed.

## Retired in 4.1.18: PYSEC-2026-3447 — setuptools 81.0.0

A Unicode-normalization exclusion bypass while building an sdist on macOS. It
was reachable only through the `setuptools<82` constraint of PyTorch 2.11.0.
PyTorch 2.13.0 accepts setuptools 83.0.0, the fixed release, and 4.1.18
declares that as a floor so an existing install is upgraded too. The CI
suppression is removed.

---

## Why these stay published

Naming an advisory, proving the vulnerable API is unreachable from untrusted
input, and committing to a removal condition is more useful to a reader than
silence. Ignoring the advisories and saying nothing would be worse.

The obligation that comes with publishing is that the entries must stay true.
The 2026-08-09 revision existed because three "no patch available upstream"
statements had all been overtaken by stable releases, and one deferral
rationale blamed a crash that had since been root-caused to an unrelated
dependency (a `pydantic-core` garbage-collection crash, unrelated to PyTorch).

## Retired exceptions

- `GHSA-rrmf-rvhw-rf47` (PyTorch) and `PYSEC-2026-3447` (setuptools) were
  retired on 2026-10-02 by the 4.1.18 dependency update.
- `PYSEC-2026-597` (NLTK) was retired on 2026-08-08 after V4 pinned NLTK
  3.10.0, verified LLMLingua compatibility, and removed the CI suppression.
