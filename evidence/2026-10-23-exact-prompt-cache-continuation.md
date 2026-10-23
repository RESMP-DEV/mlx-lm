# Exact prompt-cache continuation regression, 2026-10-23

## Change

Commit `de15a0e` repairs two exact prompt-cache-hit paths in
`mlx_lm/server.py`. When `LRUPromptCache.fetch_nearest_cache` returns an exact
hit, the server previously consumed every token segment (batched path) or
created an empty `rest` prompt (single path). `GenerationThread` then died with
`Sequence 1 has an empty prompt`. The fix re-prefills the full prompt, sets the
cache to `None`, and reports zero cached tokens for that request.

## Root-cause proof

`tests/test_server.py::TestExactPromptCacheHit` sends a real completion, then
sends `prompt + response` as the second prompt. Before the fix, the test failed
by reproducing the exact server error:

`ValueError: Sequence 1 has an empty prompt.`

The generation thread then became unavailable and the HTTP connection closed
without a response. The temporary pre-fix run exited 1.

## Regression coverage and validation

The new dedicated HTTP test:

- receives HTTP 200 on the initial completion;
- receives HTTP 200 on the exact cached continuation;
- gets non-empty continuation text;
- keeps `ResponseGenerator.generation_available()` true.

After the fix:

- dedicated regression test: 1 passed;
- full `tests/test_server.py`: 42 passed;
- `ruff format` on changed files: unchanged;
- `ruff check` on changed files: 0 findings;
- `git diff --check`: clean.

Commands were run in `/Users/kearm/mlx-lm` with the repository's uv
environment.

## Non-claims

This fixes the exact-cache continuation path that blocked the Flint entropy
controller. It does not establish a general agentic benchmark win or new token
savings. Chat-marker task recalibration remains separate.
